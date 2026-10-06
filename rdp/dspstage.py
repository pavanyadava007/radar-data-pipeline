"""Stage 2: DSP over the whole HDF5 store -> features (SQLite + parquet) and log RD maps."""

from __future__ import annotations

import json
import time

import h5py
import numpy as np
import pandas as pd

from rdp import catalog, dsp, hw
from rdp import config as C

BATCH = 4096


def run(p=None) -> dict:
    p = p or C.paths()
    t0 = time.perf_counter()
    feats = {k: [] for k in dsp.FEATURES}
    with h5py.File(p.h5, "r") as f:
        ds = f["iq"]
        n = ds.shape[0]
        rd = np.lib.format.open_memmap(p.rd, mode="w+", dtype=np.float16, shape=(n, C.N_RANGE, C.N_SLOW))
        for s in range(0, n, BATCH):
            x = ds[s : s + BATCH]
            fe, r = dsp.features(x)
            rd[s : s + len(x)] = r
            for k in dsp.FEATURES:
                feats[k].append(fe[k])
        rd.flush()
        del rd
    elapsed = time.perf_counter() - t0
    df = pd.DataFrame({k: np.concatenate(v) for k, v in feats.items()})
    df.insert(0, "segment_id", np.arange(n))
    con = catalog.connect(p.catalog)
    meta = pd.read_sql(
        "SELECT s.segment_id, m.label, m.class_group, s.measurement_id, s.paper_split, s.split_grouped, s.edge_flag,"
        " s.gap_before_s FROM segments s JOIN measurements m USING(measurement_id) ORDER BY s.segment_id",
        con,
    )
    df = meta.merge(df, on="segment_id")
    df.to_parquet(p.features, index=False)
    catalog.update_columns(con, df.segment_id, {k: df[k].to_numpy() for k in dsp.FEATURES})
    con.close()

    def q(v):
        return {
            "median": round(float(np.median(v)), 2),
            "p10": round(float(np.percentile(v, 10)), 2),
            "p90": round(float(np.percentile(v, 90)), 2),
        }

    res = {
        "n_segments": int(n),
        "parameters": {
            "window": "periodic Hann, 256 slow-time samples, power normalised by sum(w)^2",
            "prf_hz": C.PRF_HZ,
            "fc_hz": C.FC_HZ,
            "doppler_resolution_hz": C.PRF_HZ / C.N_SLOW,
            "doppler_span_hz": [float(dsp.doppler_axis_hz()[0]), float(dsp.doppler_axis_hz()[-1])],
            "velocity_span_mps": [round(float(dsp.doppler_axis_mps()[0]), 2), round(float(dsp.doppler_axis_mps()[-1]), 2)],
            "cfar": {
                "type": "CA, circular, along Doppler, centre range cell",
                "guard_each_side": C.CFAR_GUARD,
                "train_each_side": C.CFAR_TRAIN,
                "pfa": C.CFAR_PFA,
                "alpha": round(dsp.cfar_alpha(2 * C.CFAR_TRAIN, C.CFAR_PFA), 3),
            },
            "stft": {"nperseg": C.STFT_NPERSEG, "hop": C.STFT_HOP, "md_threshold_db": C.MD_THRESHOLD_DB},
        },
        "snr_db_by_class_group": {g: q(df.snr_db[df.class_group == g]) for g in C.CLASS_GROUPS if (df.class_group == g).any()},
        "snr_db_by_label": {lab: q(df.snr_db[df.label == lab]) for lab in sorted(df.label.unique())},
        "feature_medians_by_class_group": {
            g: {k: round(float(df[k][df.class_group == g].median()), 3) for k in dsp.FEATURES}
            for g in C.CLASS_GROUPS
            if (df.class_group == g).any()
        },
        "n_low_snr": int((df.snr_db < C.LOW_SNR_DB).sum()),
        "timing": {
            "dsp_seconds": round(elapsed, 2),
            "segments_per_second": round(n / elapsed, 1),
            "device": "CPU (numpy, single process)",
            **hw.describe(),
        },
    }
    (p.results / "dsp.json").write_text(json.dumps(res, indent=2))
    return res
