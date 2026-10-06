"""Stage 3: rule checks on the real data + ML anomaly detection, evaluated on
(a) the REAL edge-truncated segments and (b) INJECTED (synthetic) faults on grouped-test segments.

Protocol
- "clean" = edge_flag == 0 (the dataset's own label). Rules are NOT used to define clean,
  so rule false alarms on clean data are measured, not assumed away.
- Statistical rule thresholds and all ML models are fitted on clean grouped-TRAIN only.
- ML deployed threshold = 99th percentile of the score on clean grouped-VAL (target 1 % FA).
- Faults are injected into randomly chosen clean grouped-TEST segments; negatives are all
  clean grouped-TEST segments.
"""

from __future__ import annotations

import json
import time

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from rdp import anomaly, catalog, dsp, hw, rules
from rdp import config as C
from rdp.faults import FAULTS, inject
from rdp.rawio import segment_hash

N_PER_FAULT = 500
METHODS = ("rules", "iforest", "pca", "cae", "rules_or_cae")


def _rule_stats_all(p, hashes: list[str]) -> dict[str, np.ndarray]:
    out: dict[str, list] = {}
    seen: set = set()
    with h5py.File(p.h5, "r") as f:
        ds = f["iq"]
        for s in range(0, ds.shape[0], 4096):
            x = ds[s : s + 4096]
            st = rules.rule_stats(x, known_hashes=seen, hashes=hashes[s : s + len(x)])
            seen.update(hashes[s : s + len(x)])
            for k, v in st.items():
                out.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in out.items()}


def _recall_at_fa(neg: np.ndarray, pos: np.ndarray, fa: float = 0.01) -> float:
    thr = np.quantile(neg, 1 - fa)
    return float((pos > thr).mean())


def _auc(neg: np.ndarray, pos: np.ndarray) -> float:
    y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    s = np.nan_to_num(np.r_[neg, pos].astype(np.float64), nan=1e300, posinf=1e300)
    return float(roc_auc_score(y, s))


def run(p=None, n_per_fault: int = N_PER_FAULT, cae_epochs: int = 15, seed: int = 0) -> dict:
    p = p or C.paths()
    con = catalog.connect(p.catalog)
    df = pd.read_parquet(p.features)
    hashes = pd.read_sql("SELECT sha1 FROM segments ORDER BY segment_id", con).sha1.tolist()
    rd = np.load(p.rd, mmap_mode="r")
    feat_cols = list(dsp.FEATURES)

    # ---------------------------------------------------------------- rules on real data
    t0 = time.perf_counter()
    st = _rule_stats_all(p, hashes)
    t_rules = time.perf_counter() - t0
    clean = (df.edge_flag == 0).to_numpy()
    tr = clean & (df.split_grouped == 1).to_numpy()
    va = clean & (df.split_grouped == 2).to_numpy()
    te = clean & (df.split_grouped == 3).to_numpy()
    thr = rules.calibrate({k: v[tr] for k, v in st.items()})
    fl = rules.flags(st, thr)
    rscore = rules.rule_score(st, thr)
    bits = rules.flag_bits(fl)
    low_snr = (df.snr_db < C.LOW_SNR_DB).to_numpy()
    gap = (df.gap_before_s.fillna(0) > C.GAP_S).to_numpy()

    # ---------------------------------------------------------------- ML detectors
    t1 = time.perf_counter()
    fd = anomaly.FeatureDetectors(seed=seed).fit(df.loc[tr, feat_cols].to_numpy())
    t_feat = time.perf_counter() - t1
    cae, cae_info = anomaly.train_cae(np.asarray(rd[tr]), epochs=cae_epochs, seed=seed)
    F_all = df[feat_cols].to_numpy()
    scores_real = {
        "rules": rscore,
        "iforest": fd.score_iforest(F_all),
        "pca": fd.score_pca(F_all),
        "cae": anomaly.score_cae(cae, np.asarray(rd)),
    }
    ml_thr = {m: float(np.quantile(scores_real[m][va], 0.99)) for m in ("iforest", "pca", "cae")}
    catalog.update_columns(
        con,
        df.segment_id,
        {
            "rule_flags": bits,
            "rule_score": rscore,
            "low_snr": low_snr.astype(int),
            "time_gap": gap.astype(int),
            "anomaly_score": scores_real["cae"],
            "anomaly_score_iforest": scores_real["iforest"],
        },
    )
    con.close()

    def flagged(m: str, s: dict, idx=slice(None)) -> np.ndarray:
        if m == "rules":
            return s["rules"][idx] >= 1.0
        if m == "rules_or_cae":
            return (s["rules"][idx] >= 1.0) | (s["cae"][idx] > ml_thr["cae"])
        return s[m][idx] > ml_thr[m]

    def combo_score(s: dict) -> np.ndarray:
        # rank-free combination for AUC: max of each score normalised by its deployed threshold
        return np.maximum(s["rules"], s["cae"] / ml_thr["cae"])

    for s in (scores_real,):
        s["rules_or_cae"] = combo_score(s)

    # ---------------------------------------------------------------- injected faults (synthetic)
    rng = np.random.default_rng(seed)
    te_idx = np.where(te)[0]
    known = set(hashes)
    with h5py.File(p.h5, "r") as f:
        ds = f["iq"]
        inj_scores: dict[str, dict] = {}
        per_rule: dict[str, dict] = {}
        for fault in FAULTS:
            pick = np.sort(rng.choice(te_idx, size=min(n_per_fault, len(te_idx)), replace=False))
            x = ds[pick]
            if fault == "duplicate_segment":
                donors_idx = np.clip(pick - 1, 0, None)
                donors = ds[np.unique(donors_idx)]
                lut = {d: i for i, d in enumerate(np.unique(donors_idx))}
                y = np.stack([inject(x[i], fault, rng, donor=donors[lut[d]]) for i, d in enumerate(donors_idx)])
            else:
                y = np.stack([inject(x[i], fault, rng) for i in range(len(x))])
            yh = [segment_hash(v) for v in y]
            s_rules = rules.rule_stats(y, known_hashes=known, hashes=yh)
            fe, r = dsp.features(y)
            Fy = np.stack([fe[k] for k in feat_cols], axis=1)
            sc = {
                "rules": rules.rule_score(s_rules, thr),
                "iforest": fd.score_iforest(Fy),
                "pca": fd.score_pca(Fy),
                "cae": anomaly.score_cae(cae, r),
            }
            sc["rules_or_cae"] = combo_score(sc)
            inj_scores[fault] = sc
            fly = rules.flags(s_rules, thr)
            per_rule[fault] = {k: round(float(v.mean()), 4) for k, v in fly.items() if v.any()}

    # ---------------------------------------------------------------- tables
    edge_idx = np.where((df.edge_flag == 1).to_numpy())[0]
    neg = {m: scores_real[m][te] for m in METHODS}
    table: dict[str, dict] = {}
    rows = [(f"edge_truncated (REAL, n={len(edge_idx)})", {m: scores_real[m][edge_idx] for m in METHODS}, "real")]
    rows += [(f"{fa} (injected, synthetic)", inj_scores[fa], "injected") for fa in FAULTS]
    for name, sc, kind in rows:
        table[name] = {"kind": kind, "n": int(len(sc["rules"]))}
        for m in METHODS:
            if m == "rules":
                det = sc[m] >= 1.0
            elif m == "rules_or_cae":
                det = (sc["rules"] >= 1.0) | (sc["cae"] > ml_thr["cae"])
            else:
                det = sc[m] > ml_thr[m]
            table[name][m] = {
                "recall_deployed": round(float(det.mean()), 4),
                "recall_at_1pct_fa": round(_recall_at_fa(neg[m], sc[m]), 4),
                "roc_auc": round(_auc(neg[m], sc[m]), 4),
            }
    fa_clean = {m: round(float(flagged(m, scores_real, te).mean()), 4) for m in METHODS}
    fa_clean_val = {m: round(float(flagged(m, scores_real, va).mean()), 4) for m in METHODS}

    def hist(v):
        v = np.log10(np.clip(np.nan_to_num(np.asarray(v, dtype=np.float64), nan=1e30, posinf=1e30), 1e-6, 1e30))
        return v

    bins = {}
    hists = {}
    for m in ("rules", "iforest", "cae"):
        allv = np.concatenate([hist(neg[m])] + [hist(inj_scores[f][m]) for f in FAULTS] + [hist(scores_real[m][edge_idx])])
        lo, hi = np.percentile(allv, 0.5), np.percentile(allv, 99.5)
        edges = np.linspace(lo, hi, 41)
        bins[m] = [round(float(e), 4) for e in edges]
        hists[m] = {
            "clean_test": np.histogram(np.clip(hist(neg[m]), lo, hi), edges)[0].tolist(),
            "edge_truncated_real": np.histogram(np.clip(hist(scores_real[m][edge_idx]), lo, hi), edges)[0].tolist(),
        }
        for f in FAULTS:
            hists[m][f] = np.histogram(np.clip(hist(inj_scores[f][m]), lo, hi), edges)[0].tolist()
        hists[m]["deployed_threshold_log10"] = round(float(np.log10(1.0 if m == "rules" else ml_thr[m])), 4)

    real_flags = {
        k: {g: int(fl[k][(df.class_group == g).to_numpy()].sum()) for g in C.CLASS_GROUPS} for k in rules.INTEGRITY_RULES
    }
    any_flag = bits > 0
    res_rules = {
        "thresholds": {k: round(v, 4) for k, v in thr.items()},
        "threshold_source": (
            f"hard rules fixed in config.py; statistical rules = {C.RULE_PERCENTILE}th percentile on clean grouped-train"
        ),
        "flags_on_real_data_by_class_group": real_flags,
        "any_integrity_flag_real": int(any_flag.sum()),
        "any_integrity_flag_rate_real": round(float(any_flag.mean()), 4),
        "edge_segments_flagged_by_edge_cliff": int(fl["edge_cliff"][edge_idx].sum()),
        "edge_segments_flagged_by_any_rule": int(any_flag[edge_idx].sum()),
        "low_snr_segments": int(low_snr.sum()),
        "low_snr_by_class_group": {g: int(low_snr[(df.class_group == g).to_numpy()].sum()) for g in C.CLASS_GROUPS},
        "time_gaps_over_0.5s": int(gap.sum()),
        "exact_duplicate_segments_real": int(fl["duplicate"].sum()),
        "rules_seconds": round(t_rules, 2),
    }
    res = {
        "protocol": __doc__.strip(),
        "n_clean_train": int(tr.sum()),
        "n_clean_val": int(va.sum()),
        "n_clean_test": int(te.sum()),
        "n_edge_real": int(len(edge_idx)),
        "n_per_injected_fault": n_per_fault,
        "ml_thresholds_deployed": {k: round(v, 6) for k, v in ml_thr.items()},
        "false_alarm_rate_clean_test": fa_clean,
        "false_alarm_rate_clean_val": fa_clean_val,
        "table": table,
        "rules_firing_per_injected_fault": per_rule,
        "score_histograms_log10": {"bins": bins, "counts": hists},
        "timing": {"feature_detectors_fit_seconds_cpu": round(t_feat, 2), "cae": cae_info, **hw.describe()},
    }
    (p.results / "quality_rules.json").write_text(json.dumps(res_rules, indent=2))
    (p.results / "anomaly.json").write_text(json.dumps(res, indent=2))
    return res
