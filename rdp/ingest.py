"""Stage 1: raw .npy -> chunked HDF5 tensor store + SQLite catalogue."""

from __future__ import annotations

import json
import time

import h5py
import numpy as np

from rdp import catalog
from rdp import config as C
from rdp.rawio import file_md5, load_raw, segment_hash
from rdp.splits import grouped_split


def run(p=None) -> dict:
    p = p or C.paths()
    t0 = time.perf_counter()
    meas = load_raw(p.raw)
    t_load = time.perf_counter() - t0
    # cross-measurement exact duplicates -> link measurements before splitting
    first_seen: dict[str, int] = {}
    links = set()
    for mid, m in enumerate(meas):
        for j in range(len(m.iq)):
            h = segment_hash(m.iq[j])
            if h in first_seen and first_seen[h] != mid:
                links.add((first_seen[h], mid))
            first_seen.setdefault(h, mid)
    split_g = grouped_split([m.label for m in meas], [len(m.iq) for m in meas], sorted(links))
    n = sum(len(m.iq) for m in meas)
    p.h5.parent.mkdir(parents=True, exist_ok=True)
    con = catalog.connect(p.catalog)
    catalog.create(con)
    t1 = time.perf_counter()
    rows = []
    hashes = {}
    dup = 0
    with h5py.File(p.h5, "w") as f:
        ds = f.create_dataset("iq", shape=(n, C.N_RANGE, C.N_SLOW), dtype=np.complex64, chunks=(256, C.N_RANGE, C.N_SLOW))
        f.attrs["prf_hz"] = C.PRF_HZ
        f.attrs["fc_hz"] = C.FC_HZ
        f.attrs["layout"] = "segment, range_cell, slow_time"
        sid = 0
        for mid, (m, sg) in enumerate(zip(meas, split_g)):
            k = len(m.iq)
            ds[sid : sid + k] = m.iq
            dur = float(m.time_s[-1] - m.time_s[0]) if k > 1 else 0.0
            con.execute("INSERT INTO measurements VALUES (?,?,?,?,?,?)", (mid, m.label, C.class_group(m.label), k, dur, sg))
            gaps = np.concatenate([[np.nan], np.diff(m.time_s)])
            for j in range(k):
                h = segment_hash(m.iq[j])
                if h in hashes:
                    dup += 1
                hashes.setdefault(h, sid + j)
                rows.append(
                    (
                        sid + j,
                        mid,
                        j,
                        float(m.range_m[j]),
                        float(m.time_s[j]),
                        None if np.isnan(gaps[j]) else float(gaps[j]),
                        int(m.paper_split[j]),
                        sg,
                        int(m.edge_flag[j]),
                        h,
                    )
                )
            sid += k
    con.executemany(
        "INSERT INTO segments (segment_id, measurement_id, idx_in_measurement, range_m, time_s, gap_before_s,"
        " paper_split, split_grouped, edge_flag, sha1) VALUES (?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    con.commit()
    con.close()
    t_write = time.perf_counter() - t1
    labels = sorted({m.label for m in meas})
    res = {
        "source_file": p.raw.name,
        "source_md5": file_md5(p.raw),
        "n_measurements": len(meas),
        "n_segments": n,
        "n_edge_flagged": int(sum(int(m.edge_flag.sum()) for m in meas)),
        "n_exact_duplicate_segments": dup,
        "segments_per_label": {lab: int(sum(len(m.iq) for m in meas if m.label == lab)) for lab in labels},
        "measurements_per_label": {lab: sum(1 for m in meas if m.label == lab) for lab in labels},
        "segments_per_class_group": {g: int(sum(len(m.iq) for m in meas if C.class_group(m.label) == g)) for g in C.CLASS_GROUPS},
        "paper_split_counts": {str(s): int(sum(int((m.paper_split == s).sum()) for m in meas)) for s in (1, 2, 3)},
        "grouped_split_counts": {str(s): int(sum(len(m.iq) for m, g in zip(meas, split_g) if g == s)) for s in (1, 2, 3)},
        "grouped_split_measurements": {str(s): split_g.count(s) for s in (1, 2, 3)},
        "measurement_links_from_duplicates": [list(x) for x in sorted(links)],
        "h5_bytes": p.h5.stat().st_size,
        "timing_s": {"load_npy": round(t_load, 2), "write_h5_and_sqlite": round(t_write, 2)},
    }
    p.results.mkdir(parents=True, exist_ok=True)
    (p.results / "ingest.json").write_text(json.dumps(res, indent=2))
    return res
