"""Rule-based data quality checks.

Each rule returns a per-segment statistic. Hard rules have fixed thresholds
(config.py); statistical rules are thresholded at the RULE_PERCENTILE of the statistic
on grouped-train, edge-free segments (see `calibrate`). `rule_score` combines all
integrity rules into one continuous score (> 1 means at least one rule fires), which
allows ROC analysis of the rule set next to the ML detectors.

Integrity rules (count as faults):
  nonfinite      any NaN / inf sample                                     fixed: > 0
  dead_block     longest run of samples with |x| < ZERO_EPS in all cells  fixed: >= DEAD_RUN
  frozen_block   longest run of identical consecutive samples            fixed: >= FROZEN_RUN
  clipping       number of I/Q samples sitting at the segment's rail      fixed: >= CLIP_MIN_SAMPLES
  duplicate      sha1 of the segment seen before in the catalogue          fixed: any
  dc_offset      |mean over cells of per-cell complex mean|^2 / noise      statistical
  spike          max over time of the median over cells of a robust z of
                 the second difference (impulsive interference)            statistical
  off_centre     max power of a non-centre cell / centre cell power        statistical
  edge_cliff     log10 of the largest power drop into a low tail
                 (noise-filled azimuth at the field-of-view edge)          statistical
Quality flags (stored, not faults):
  low_snr        snr_db < LOW_SNR_DB
  time_gap       gap to previous segment of the measurement > GAP_S
"""

from __future__ import annotations

import numpy as np

from rdp import config as C

HARD_RULES = ("nonfinite", "dead_block", "frozen_block", "clipping", "duplicate")
STAT_RULES = ("dc_offset", "spike", "off_centre", "edge_cliff")
INTEGRITY_RULES = HARD_RULES + STAT_RULES
HARD_THRESHOLDS = {
    "nonfinite": 1,
    "dead_block": C.DEAD_RUN,
    "frozen_block": C.FROZEN_RUN,
    "clipping": C.CLIP_MIN_SAMPLES,
    "duplicate": 1,
}


def _longest_run(mask: np.ndarray) -> np.ndarray:
    """Longest run of True along the last axis, mask [B, T] -> [B]."""
    b, t = mask.shape
    best = np.zeros(b, dtype=np.int32)
    cur = np.zeros(b, dtype=np.int32)
    for i in range(t):
        cur = np.where(mask[:, i], cur + 1, 0)
        best = np.maximum(best, cur)
    return best


def rule_stats(x: np.ndarray, known_hashes: set | None = None, hashes: list | None = None) -> dict[str, np.ndarray]:
    b = x.shape[0]
    s: dict[str, np.ndarray] = {}
    finite = np.isfinite(x.real) & np.isfinite(x.imag)
    s["nonfinite"] = (~finite).reshape(b, -1).sum(axis=1).astype(np.float32)
    x = np.where(finite, x, 0)
    a = np.abs(x)
    s["dead_block"] = _longest_run((a < C.ZERO_EPS).all(axis=1)).astype(np.float32)
    same = np.zeros((b, x.shape[-1]), dtype=bool)
    same[:, 1:] = (x[:, :, 1:] == x[:, :, :-1]).all(axis=1)
    s["frozen_block"] = (_longest_run(same) + 1).astype(np.float32)
    s["frozen_block"][~same.any(axis=1)] = 0
    re, im = np.abs(x.real), np.abs(x.imag)
    rail_re = re.reshape(b, -1).max(axis=1)
    rail_im = im.reshape(b, -1).max(axis=1)
    at_rail = (re >= rail_re[:, None, None] * (1 - C.CLIP_TOL)).reshape(b, -1).sum(axis=1) + (
        im >= rail_im[:, None, None] * (1 - C.CLIP_TOL)
    ).reshape(b, -1).sum(axis=1)
    s["clipping"] = at_rail.astype(np.float32)
    if hashes is not None:
        seen = set(known_hashes or ())
        dup = np.zeros(b, dtype=np.float32)
        for i, h in enumerate(hashes):
            if h in seen:
                dup[i] = 1
            seen.add(h)
        s["duplicate"] = dup
    else:
        s["duplicate"] = np.zeros(b, dtype=np.float32)
    # robust noise scale from the lowest-power quarter of samples
    pw = a**2
    noise = np.quantile(pw.reshape(b, -1), 0.25, axis=1) / 0.2877 + 1e-20  # exp. dist: q25 = 0.2877 mean
    dc = x.mean(axis=2).mean(axis=1)
    s["dc_offset"] = (np.abs(dc) ** 2 / noise).astype(np.float32)
    d2 = x[:, :, 1:-1] - 0.5 * (x[:, :, :-2] + x[:, :, 2:])
    ad = np.abs(d2)
    mad = np.median(ad, axis=2, keepdims=True) + 1e-20
    s["spike"] = np.median(ad / mad, axis=1).max(axis=1).astype(np.float32)
    pr = pw.sum(axis=2)
    others = np.delete(pr, C.CENTRE_CELL, axis=1).max(axis=1)
    s["off_centre"] = (others / (pr[:, C.CENTRE_CELL] + 1e-20)).astype(np.float32)
    s["edge_cliff"] = edge_cliff(x)
    return s


def edge_cliff(x: np.ndarray, w: int = C.EDGE_WINDOW) -> np.ndarray:
    """Largest drop (log10 power ratio) from a w-sample window into the following tail.

    For a cut at sample k on the right: mean power of [k-w, k) over the max of all
    sliding w-windows in [k, end). Mirror for the left. A noise-filled tail produces
    a step that a smooth two-way antenna pattern cannot produce.
    """
    p = (np.abs(x[:, 1:4]) ** 2).sum(axis=1)
    cs = np.concatenate([np.zeros((p.shape[0], 1)), np.cumsum(p, axis=1)], axis=1)
    sw = (cs[:, w:] - cs[:, :-w]) / w  # sw[:, i] = mean p[i:i+w]
    n = sw.shape[1]
    suf = np.maximum.accumulate(sw[:, ::-1], axis=1)[:, ::-1]
    pre = np.maximum.accumulate(sw, axis=1)
    right = (sw[:, : n - w] / (suf[:, w:] + 1e-20)).max(axis=1)
    left = (sw[:, w:] / (pre[:, : n - w] + 1e-20)).max(axis=1)
    return np.log10(np.maximum(np.maximum(right, left), 1e-20)).astype(np.float32)


def calibrate(stats: dict[str, np.ndarray], pct: float = C.RULE_PERCENTILE) -> dict[str, float]:
    thr = {k: float(v) for k, v in HARD_THRESHOLDS.items()}
    for k in STAT_RULES:
        thr[k] = float(np.percentile(stats[k], pct))
    return thr


def flags(stats: dict[str, np.ndarray], thr: dict[str, float]) -> dict[str, np.ndarray]:
    return {k: stats[k] >= thr[k] for k in INTEGRITY_RULES}


def rule_score(stats: dict[str, np.ndarray], thr: dict[str, float]) -> np.ndarray:
    """Continuous combined rule score: max over rules of stat / threshold (>= 1 means flagged)."""
    parts = []
    for k in INTEGRITY_RULES:
        v = stats[k].astype(np.float64)
        t = thr[k]
        if k in HARD_RULES:  # certain violations: huge score when fired, 0 otherwise
            parts.append(np.where(v >= t, 1e6, 0.0))
        elif k == "edge_cliff":  # log-domain statistic, compare in linear power
            parts.append(10 ** (v - t))
        else:
            parts.append(v / t)
    return np.max(np.stack(parts), axis=0)


def flag_bits(fl: dict[str, np.ndarray]) -> np.ndarray:
    bits = np.zeros(len(next(iter(fl.values()))), dtype=np.int64)
    for i, k in enumerate(INTEGRITY_RULES):
        bits |= fl[k].astype(np.int64) << i
    return bits
