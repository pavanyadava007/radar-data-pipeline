"""Central configuration: paths, radar constants and documented thresholds.

Paths can be redirected with environment variables so the test-suite and CI can
run the full pipeline on a small synthetic fixture:
  RDP_RAW      path to the raw .npy cell matrix
  RDP_DATA     directory for derived data (HDF5, SQLite, exports)
  RDP_RESULTS  directory for results/*.json
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _p(env: str, default: Path) -> Path:
    return Path(os.environ.get(env, str(default)))


@dataclass(frozen=True)
class Paths:
    raw: Path
    data: Path
    results: Path

    @property
    def h5(self) -> Path:
        return self.data / "derived" / "segments.h5"

    @property
    def catalog(self) -> Path:
        return self.data / "catalog.sqlite"

    @property
    def rd(self) -> Path:
        return self.data / "derived" / "rd_db.npy"

    @property
    def features(self) -> Path:
        return self.data / "derived" / "features.parquet"

    @property
    def exports(self) -> Path:
        return self.data / "exports"

    @property
    def models(self) -> Path:
        return self.data / "models"


def paths() -> Paths:
    return Paths(
        raw=_p("RDP_RAW", ROOT / "data" / "raw" / "data_SAAB_SIRS_77GHz_FMCW.npy"),
        data=_p("RDP_DATA", ROOT / "data"),
        results=_p("RDP_RESULTS", ROOT / "results"),
    )


# ---------------------------------------------------------------- radar constants
PRF_HZ = 17_000.0  # pulse (sweep) repetition frequency, from the dataset ReadMe
FC_HZ = 77e9  # carrier frequency
C_MPS = 299_792_458.0
WAVELENGTH_M = C_MPS / FC_HZ
N_RANGE = 5  # range cells per segment
N_SLOW = 256  # slow-time / azimuth samples per range cell
CENTRE_CELL = 2  # target is in the third range cell

# ---------------------------------------------------------------- DSP parameters
CFAR_GUARD = 4  # guard cells on each side of the cell under test
CFAR_TRAIN = 16  # training cells on each side
CFAR_PFA = 1e-4  # design probability of false alarm per Doppler bin
STFT_NPERSEG = 64
STFT_HOP = 16
MD_THRESHOLD_DB = 10.0  # spectrogram bins this far above the noise floor count as micro-Doppler
ZERO_DOPPLER_BINS = 2  # |bin| <= 2 around 0 Hz is treated as clutter / zero Doppler

# ---------------------------------------------------------------- quality rules
# Hard rules use fixed thresholds. Statistical rules use the RULE_PERCENTILE of the
# rule statistic on grouped-TRAIN, edge-free segments (computed in `rdp quality`,
# saved to results/rule_thresholds.json).
ZERO_EPS = 1e-9  # |x| below this counts as a dead sample
DEAD_RUN = 16  # >= this many consecutive dead samples (all range cells) -> dead block
FROZEN_RUN = 8  # >= this many consecutive identical samples -> frozen block
CLIP_TOL = 1e-6  # relative tolerance to call a sample "at the rail"
CLIP_MIN_SAMPLES = 6  # >= this many I or Q samples at the rail -> clipping
LOW_SNR_DB = 6.0  # quality flag only (not an integrity fault)
GAP_S = 0.5  # time gap between consecutive segments of one measurement worth flagging
RULE_PERCENTILE = 99.9
EDGE_WINDOW = 8  # sliding window (samples) used by the edge-fill cliff statistic

# ---------------------------------------------------------------- splits / training
GROUPED_FRACTIONS = (0.6, 0.2, 0.2)  # train / val / test by segment count, grouped by measurement
SEEDS = (0, 1, 2)

CLASS_GROUPS = ("drone", "bird", "human", "reflector")
DRONES = ("D1", "D2", "D3", "D4", "D5", "D6")


def class_group(label: str) -> str:
    if label.startswith("D") and label[1:].isdigit():
        return "drone"
    if label.startswith("human"):
        return "human"
    if label == "CR":
        return "reflector"
    return "bird"
