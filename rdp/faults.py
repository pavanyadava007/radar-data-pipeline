"""Physically motivated fault injection (SYNTHETIC faults on real clean segments).

Every injected fault is labelled as injected/synthetic in all outputs. Amplitudes are
set relative to the segment's own noise level so faults are neither trivial nor
invisible by construction.
"""

from __future__ import annotations

import numpy as np

FAULTS = (
    "dropped_block",  # 32 consecutive sweeps lost (zeros), e.g. DMA / buffer overrun
    "adc_clipping",  # I and Q clipped at 30 % of the segment's peak (receiver saturation, approximated post range-FFT)
    "interference_burst",  # 3 short bursts (2 sweeps) of broadband energy in all range cells (mutual FMCW interference)
    "dc_leakage",  # constant complex offset identical in all range cells (TX-RX leakage / DC bias)
    "phase_jump",  # LO phase step of 90-180 degrees at a random sweep, all cells
    "range_offset",  # range cells shifted by +-1 or +-2 (range-gate / timing misalignment)
    "frozen_block",  # sample-and-hold of one sweep for 32 sweeps (stuck ADC / stale buffer)
    "duplicate_segment",  # exact copy of another catalogued segment (pipeline duplication)
    "gain_drift",  # linear gain ramp of +-10 dB across the segment (AGC / temperature drift)
)


def _noise_rms(x: np.ndarray) -> float:
    p = np.abs(x) ** 2
    return float(np.sqrt(np.quantile(p, 0.25) / 0.2877))


def inject(x: np.ndarray, fault: str, rng: np.random.Generator, donor: np.ndarray | None = None) -> np.ndarray:
    """Return a faulted copy of one segment x[5, 256]."""
    y = x.astype(np.complex64).copy()
    n = y.shape[-1]
    if fault == "dropped_block":
        k = int(rng.integers(0, n - 32))
        y[:, k : k + 32] = 0
    elif fault == "adc_clipping":
        lvl = 0.3 * max(np.abs(y.real).max(), np.abs(y.imag).max())
        y = (np.clip(y.real, -lvl, lvl) + 1j * np.clip(y.imag, -lvl, lvl)).astype(np.complex64)
    elif fault == "interference_burst":
        s = _noise_rms(y)
        for _ in range(3):
            k = int(rng.integers(0, n - 2))
            burst = (rng.standard_normal((5, 2)) + 1j * rng.standard_normal((5, 2))) * 10 * s / np.sqrt(2)
            y[:, k : k + 2] += burst.astype(np.complex64)
    elif fault == "dc_leakage":
        s = _noise_rms(y)
        y += np.complex64(3 * s * np.exp(1j * rng.uniform(0, 2 * np.pi)))
    elif fault == "phase_jump":
        k = int(rng.integers(32, n - 32))
        phi = rng.uniform(np.pi / 2, np.pi) * rng.choice([-1, 1])
        y[:, k:] *= np.complex64(np.exp(1j * phi))
    elif fault == "range_offset":
        y = np.roll(y, int(rng.choice([-2, -1, 1, 2])), axis=0)
    elif fault == "frozen_block":
        k = int(rng.integers(0, n - 32))
        y[:, k : k + 32] = y[:, k : k + 1]
    elif fault == "duplicate_segment":
        assert donor is not None, "duplicate_segment needs a donor segment"
        y = donor.astype(np.complex64).copy()
    elif fault == "gain_drift":
        g_db = np.linspace(0, 10 * rng.choice([-1, 1]), n)
        y *= (10 ** (g_db / 20)).astype(np.float32)
    else:
        raise ValueError(f"unknown fault {fault}")
    return y
