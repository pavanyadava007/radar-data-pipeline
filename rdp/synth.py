"""Synthetic fixture in the exact layout of the Zenodo cell matrix, so tests and CI need no download.

Signal model per segment (5 range cells x 256 slow-time samples at PRF 17 kHz):
two-way antenna beam envelope (Gaussian over azimuth) x target return in the centre cell
(with -10 dB leakage into neighbours) + weak zero-Doppler clutter + complex white noise.
Doppler content per label: drone = body line + rotor micro-Doppler (sinusoidal FM),
bird = line with slow wing-beat FM, human = line with limb sidebands, CR = 0 Hz.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rdp.config import N_RANGE, N_SLOW, PRF_HZ

LABELS = ("D1", "D2", "seagull", "human_walk", "CR")


def _target(label: str, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(N_SLOW) / PRF_HZ
    if label.startswith("D"):
        f0 = rng.uniform(-1500, 1500)
        fm = 2500 if label == "D1" else 4000
        ph = 2 * np.pi * f0 * t + (fm / 900) * np.sin(2 * np.pi * 900 * t + rng.uniform(0, 6.3))
        return np.exp(1j * ph) * (1 + 0.5 * rng.standard_normal())
    if label == "CR":
        return np.full(N_SLOW, 3.0 + 0j) * np.exp(1j * rng.uniform(0, 6.3))
    if label.startswith("human"):
        f0 = rng.uniform(-700, -300)
        return np.exp(1j * 2 * np.pi * f0 * t) * (1 + 0.6 * np.cos(2 * np.pi * 400 * t))
    f0 = rng.uniform(500, 2000)
    return 0.6 * np.exp(1j * (2 * np.pi * f0 * t + 3 * np.sin(2 * np.pi * 60 * t)))


def make_segment(label: str, rng: np.random.Generator, noise: float = 0.05, edge: bool = False) -> np.ndarray:
    n = np.arange(N_SLOW)
    env = np.exp(-0.5 * ((n - 128) / 35.0) ** 2)
    x = np.zeros((N_RANGE, N_SLOW), dtype=np.complex128)
    tgt = _target(label, rng) * env
    for c, g in zip(range(N_RANGE), (0.03, 0.3, 1.0, 0.3, 0.03)):
        x[c] = g * tgt * 10
        x[c] += 0.2 * np.exp(1j * rng.uniform(0, 6.3)) * env  # zero-Doppler clutter
    x += noise * (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape)) / np.sqrt(2)
    if edge:  # field-of-view edge: tail replaced by noise
        k = int(rng.integers(150, 200))
        x[:, k:] = (
            noise * (rng.standard_normal((N_RANGE, N_SLOW - k)) + 1j * rng.standard_normal((N_RANGE, N_SLOW - k))) / np.sqrt(2)
        )
    return x


def make_fixture(path: Path, n_meas: int = 3, n_seg: int = 40, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    rows = []
    for label in LABELS:
        for _ in range(n_meas):
            segs, edge = [], np.zeros(n_seg, dtype=np.uint8)
            for j in range(n_seg):
                e = label == "seagull" and j == 5
                edge[j] = e
                segs.append(make_segment(label, rng, edge=e).reshape(-1))
            m = np.stack(segs, axis=1)  # [1280, n]
            t = np.cumsum(np.full(n_seg, 0.1)) + rng.uniform(0, 0.01, n_seg)
            t[n_seg // 2 :] += 2.0  # one time gap per measurement
            rows.append(
                [
                    np.array([label]),
                    m,
                    np.full((n_seg, 1), 50.0),
                    t[:, None],
                    rng.integers(1, 4, (n_seg, 1)).astype(np.uint8),
                    edge[:, None],
                ]
            )
    rows[1][1][:, 3] = rows[1][1][:, 2]  # one exact duplicate segment inside a measurement
    cells = np.empty((len(rows), 6), dtype=object)
    for i, r in enumerate(rows):
        for k in range(6):
            cells[i, k] = r[k]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, cells, allow_pickle=True)
    return path
