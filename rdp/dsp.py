"""Signal processing steps for one batch of segments x[B, 5, 256] (complex).

Conventions
- Slow-time FFT with a periodic Hann window over the 256 samples of each range cell,
  fftshift so bin 128 is 0 Hz. Power is normalised by sum(w)^2 so a unit-amplitude
  complex tone at a bin centre gives 0 dB.
- Doppler axis: f_k = (k - 128) * PRF / 256 Hz, velocity v = f * lambda / 2.
- CA-CFAR along Doppler (circular), square-law detector, N = 2 * CFAR_TRAIN reference
  cells, alpha = N * (Pfa^(-1/N) - 1) (exact for i.i.d. exponential noise).
- Noise floor: median RD power over all 5 x 256 bins divided by ln 2 (mean of an
  exponential distribution with that median). Robust to a few target / clutter bins.
"""

from __future__ import annotations

import numpy as np

from rdp import config as C


def hann(n: int) -> np.ndarray:
    return (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)).astype(np.float32)


def doppler_axis_hz(n: int = C.N_SLOW, prf: float = C.PRF_HZ) -> np.ndarray:
    return (np.arange(n) - n // 2) * prf / n


def doppler_axis_mps(n: int = C.N_SLOW, prf: float = C.PRF_HZ) -> np.ndarray:
    return doppler_axis_hz(n, prf) * C.WAVELENGTH_M / 2


def range_doppler_power(x: np.ndarray) -> np.ndarray:
    """Linear RD power [B, 5, 256] (window-normalised)."""
    w = hann(x.shape[-1])
    spec = np.fft.fftshift(np.fft.fft(x * w, axis=-1), axes=-1)
    return (np.abs(spec) ** 2 / w.sum() ** 2).astype(np.float32)


def to_db(p: np.ndarray) -> np.ndarray:
    return (10 * np.log10(np.maximum(p, 1e-20))).astype(np.float32)


def cfar_alpha(n_ref: int, pfa: float) -> float:
    return n_ref * (pfa ** (-1.0 / n_ref) - 1.0)


def ca_cfar(p: np.ndarray, guard: int = C.CFAR_GUARD, train: int = C.CFAR_TRAIN, pfa: float = C.CFAR_PFA):
    """Cell-averaging CFAR along the last axis (circular). Returns (detections bool, threshold)."""
    n = p.shape[-1]
    half = guard + train
    pad = np.concatenate([p[..., n - half :], p, p[..., :half]], axis=-1).astype(np.float64)
    cs = np.concatenate([np.zeros(p.shape[:-1] + (1,)), np.cumsum(pad, axis=-1)], axis=-1)
    idx = np.arange(n) + half  # index of the cell under test in pad
    # leading window [i-half, i-guard), lagging window (i+guard, i+half]
    lead = cs[..., idx - guard] - cs[..., idx - half]
    lag = cs[..., idx + half + 1] - cs[..., idx + guard + 1]
    noise = (lead + lag) / (2 * train)
    thr = cfar_alpha(2 * train, pfa) * noise
    return p > thr, thr.astype(np.float32)


def noise_floor(p: np.ndarray) -> np.ndarray:
    """Per-segment noise power estimate from RD power [B, 5, 256] -> [B]."""
    return np.median(p.reshape(p.shape[0], -1), axis=1) / np.log(2)


def spectrogram(x1: np.ndarray, nperseg: int = C.STFT_NPERSEG, hop: int = C.STFT_HOP) -> np.ndarray:
    """Micro-Doppler spectrogram of a slow-time signal x1[B, 256] -> power [B, frames, nperseg]."""
    w = hann(nperseg)
    frames = np.lib.stride_tricks.sliding_window_view(x1, nperseg, axis=-1)[:, ::hop]
    s = np.fft.fftshift(np.fft.fft(frames * w, axis=-1), axes=-1)
    return (np.abs(s) ** 2 / w.sum() ** 2).astype(np.float32)


FEATURES = (
    "snr_db",
    "peak_doppler_hz",
    "doppler_centroid_hz",
    "doppler_spread_hz",
    "spectral_entropy",
    "md_bandwidth_hz",
    "n_cfar",
    "zero_doppler_ratio",
    "centre_power_frac",
    "noise_floor_db",
    "envelope_cv",
    # generic slow-time signal statistics (all 5 range cells), used as QA features
    "amp_kurtosis",
    "crest_factor_db",
    "lag1_coherence",
    "envelope_roughness",
    "iq_power_ratio_db",
)


def features(x: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Compute per-segment features and the log RD map (dB re noise floor) for x[B,5,256]."""
    b = x.shape[0]
    f = doppler_axis_hz(x.shape[-1])
    p = range_doppler_power(x)
    nf = noise_floor(p)  # [B]
    pc = p[:, C.CENTRE_CELL]  # [B, 256]
    peak = pc.max(axis=1)
    out: dict[str, np.ndarray] = {}
    out["snr_db"] = 10 * np.log10(np.maximum(peak, 1e-20) / np.maximum(nf, 1e-20))
    out["peak_doppler_hz"] = f[pc.argmax(axis=1)]
    sig = np.maximum(pc - nf[:, None], 0.0)
    tot = sig.sum(axis=1) + 1e-20
    cen = (sig * f).sum(axis=1) / tot
    out["doppler_centroid_hz"] = cen
    out["doppler_spread_hz"] = np.sqrt(np.maximum((sig * (f - cen[:, None]) ** 2).sum(axis=1) / tot, 0.0))
    q = pc / pc.sum(axis=1, keepdims=True)
    out["spectral_entropy"] = -(q * np.log(np.maximum(q, 1e-30))).sum(axis=1) / np.log(pc.shape[1])
    det, _ = ca_cfar(pc)
    out["n_cfar"] = det.sum(axis=1).astype(np.float32)
    zb = np.abs(np.arange(pc.shape[1]) - pc.shape[1] // 2) <= C.ZERO_DOPPLER_BINS
    out["zero_doppler_ratio"] = pc[:, zb].sum(axis=1) / (pc.sum(axis=1) + 1e-20)
    pr = p.sum(axis=2)  # [B, 5]
    out["centre_power_frac"] = pr[:, C.CENTRE_CELL] / (pr.sum(axis=1) + 1e-20)
    out["noise_floor_db"] = 10 * np.log10(np.maximum(nf, 1e-20))
    env = np.abs(x[:, C.CENTRE_CELL])
    out["envelope_cv"] = env.std(axis=1) / (env.mean(axis=1) + 1e-20)
    a = np.abs(x).reshape(b, -1)
    am = a.mean(axis=1, keepdims=True)
    out["amp_kurtosis"] = ((a - am) ** 4).mean(axis=1) / (((a - am) ** 2).mean(axis=1) ** 2 + 1e-30)
    out["crest_factor_db"] = 20 * np.log10(a.max(axis=1) / (np.sqrt((a**2).mean(axis=1)) + 1e-20) + 1e-20)
    lag = (x[:, :, 1:] * np.conj(x[:, :, :-1])).mean(axis=(1, 2))
    out["lag1_coherence"] = np.abs(lag) / ((np.abs(x) ** 2).mean(axis=(1, 2)) + 1e-20)
    ea = np.abs(x)
    out["envelope_roughness"] = np.abs(np.diff(ea, axis=-1)).mean(axis=(1, 2)) / (ea.mean(axis=(1, 2)) + 1e-20)
    out["iq_power_ratio_db"] = 10 * np.log10(((x.real**2).mean(axis=(1, 2)) + 1e-20) / ((x.imag**2).mean(axis=(1, 2)) + 1e-20))
    # micro-Doppler bandwidth from the STFT of the centre range cell
    s = spectrogram(x[:, C.CENTRE_CELL])  # [B, T, F]
    fs = doppler_axis_hz(s.shape[-1])
    snf = np.median(s.reshape(b, -1), axis=1) / np.log(2)
    above = s > snf[:, None, None] * 10 ** (C.MD_THRESHOLD_DB / 10)
    any_f = above.any(axis=2)
    fmax = np.where(any_f, np.max(np.where(above, fs, -np.inf), axis=2), 0.0)
    fmin = np.where(any_f, np.min(np.where(above, fs, np.inf), axis=2), 0.0)
    bw = np.where(any_f, fmax - fmin + (fs[1] - fs[0]), 0.0)
    out["md_bandwidth_hz"] = bw.mean(axis=1)
    out = {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}
    rd_db = to_db(p) - to_db(nf)[:, None, None]
    return out, rd_db.astype(np.float16)
