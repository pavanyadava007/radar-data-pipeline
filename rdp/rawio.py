"""Reader for the raw Zenodo cell matrix (130 x 6 object array)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rdp.config import N_RANGE, N_SLOW


@dataclass
class Measurement:
    label: str
    iq: np.ndarray  # complex64 [n, 5, 256]
    range_m: np.ndarray
    time_s: np.ndarray
    paper_split: np.ndarray
    edge_flag: np.ndarray


def load_raw(path: Path) -> list[Measurement]:
    cells = np.load(path, allow_pickle=True)
    out = []
    for row in cells:
        label = str(np.asarray(row[0]).ravel()[0])
        m = np.asarray(row[1])  # [1280, n]; first 256 rows = first range cell
        n = m.shape[1]
        iq = m.T.reshape(n, N_RANGE, N_SLOW).astype(np.complex64)
        out.append(
            Measurement(
                label=label,
                iq=iq,
                range_m=np.asarray(row[2], dtype=np.float64).ravel(),
                time_s=np.asarray(row[3], dtype=np.float64).ravel(),
                paper_split=np.asarray(row[4]).ravel().astype(np.int8),
                edge_flag=np.asarray(row[5]).ravel().astype(np.int8),
            )
        )
    return out


def file_md5(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def segment_hash(x: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(x, dtype=np.complex64).tobytes()).hexdigest()
