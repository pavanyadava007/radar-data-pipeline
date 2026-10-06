import numpy as np
import pytest

from rdp import config as C
from rdp import dsp


def tone(f_hz, amp=1.0, n=C.N_SLOW):
    t = np.arange(n) / C.PRF_HZ
    return amp * np.exp(2j * np.pi * f_hz * t)


@pytest.mark.parametrize("k", [-100, -37, 0, 5, 64, 127])
def test_tone_lands_in_expected_bin(k):
    f = k * C.PRF_HZ / C.N_SLOW
    x = np.zeros((1, 5, C.N_SLOW), complex)
    x[0, 2] = tone(f)
    p = dsp.range_doppler_power(x)
    assert p[0, 2].argmax() == k + C.N_SLOW // 2
    assert dsp.doppler_axis_hz()[p[0, 2].argmax()] == pytest.approx(f)


def test_window_scaling_unit_tone_is_0_db():
    x = np.zeros((1, 5, C.N_SLOW), complex)
    x[0, 2] = tone(10 * C.PRF_HZ / C.N_SLOW)
    p = dsp.range_doppler_power(x)
    assert dsp.to_db(p)[0, 2].max() == pytest.approx(0.0, abs=1e-4)


def test_velocity_axis():
    v = dsp.doppler_axis_mps()
    lam = C.C_MPS / C.FC_HZ
    assert v[-1] == pytest.approx((C.PRF_HZ / 2 - C.PRF_HZ / C.N_SLOW) * lam / 2)
    assert abs(v[0]) == pytest.approx(C.PRF_HZ / 2 * lam / 2)  # about 16.5 m/s
    assert 16.0 < abs(v[0]) < 17.0


def test_noise_floor_matches_theory():
    rng = np.random.default_rng(0)
    sigma2 = 2.0
    x = np.sqrt(sigma2 / 2) * (rng.standard_normal((400, 5, 256)) + 1j * rng.standard_normal((400, 5, 256)))
    w = dsp.hann(256)
    expected = sigma2 * (w**2).sum() / w.sum() ** 2
    nf = dsp.noise_floor(dsp.range_doppler_power(x))
    assert nf.mean() == pytest.approx(expected, rel=0.05)


def test_cfar_alpha_formula():
    n, pfa = 32, 1e-4
    assert dsp.cfar_alpha(n, pfa) == pytest.approx(n * (pfa ** (-1 / n) - 1))


def test_cfar_false_alarm_rate_on_noise():
    rng = np.random.default_rng(1)
    pfa = 1e-3
    p = rng.exponential(1.0, size=(4000, 256))
    det, _ = dsp.ca_cfar(p, guard=4, train=16, pfa=pfa)
    rate = det.mean()
    assert pfa / 2 < rate < pfa * 2, rate


def test_cfar_finds_injected_target():
    rng = np.random.default_rng(2)
    p = rng.exponential(1.0, size=(200, 256))
    p[:, 77] += 200.0  # 23 dB target
    det, _ = dsp.ca_cfar(p)
    assert det[:, 77].mean() > 0.99
    assert det[:, np.r_[0:70, 85:256]].mean() < 1e-3


def test_spectrogram_tone_and_shape():
    f = 2000.0
    s = dsp.spectrogram(tone(f)[None])
    frames = (C.N_SLOW - C.STFT_NPERSEG) // C.STFT_HOP + 1
    assert s.shape == (1, frames, C.STFT_NPERSEG)
    fs = dsp.doppler_axis_hz(C.STFT_NPERSEG)
    assert abs(fs[s[0].mean(0).argmax()] - f) <= C.PRF_HZ / C.STFT_NPERSEG


def test_features_known_answers():
    rng = np.random.default_rng(3)
    noise = 0.01 * (rng.standard_normal((3, 5, 256)) + 1j * rng.standard_normal((3, 5, 256)))
    x = noise.copy()
    x[0, 2] += 1.0  # pure zero-Doppler (clutter-like) line
    x[1, 2] += tone(3000)
    t = np.arange(256) / C.PRF_HZ
    x[2, 2] += np.exp(1j * (2 * np.pi * 3000 * t + 4 * np.sin(2 * np.pi * 500 * t)))  # micro-Doppler FM
    f, rd = dsp.features(x)
    assert rd.shape == (3, 5, 256) and rd.dtype == np.float16
    assert f["zero_doppler_ratio"][0] > 0.9
    assert f["peak_doppler_hz"][1] == pytest.approx(3000, abs=C.PRF_HZ / 256)
    assert f["doppler_centroid_hz"][1] == pytest.approx(3000, abs=100)
    assert f["md_bandwidth_hz"][2] > 2 * f["md_bandwidth_hz"][1]
    assert f["doppler_spread_hz"][2] > f["doppler_spread_hz"][1]
    assert f["snr_db"][1] > 30
    assert f["centre_power_frac"][1] > 0.9
