"""Stage 4: versioned AI-training dataset export (arrays + manifest with checksums)."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import yaml

from rdp import __version__, catalog
from rdp import config as C

ALLOWED_FILTER_COLUMNS = {"edge_flag", "rule_flags", "low_snr", "time_gap", "paper_split", "split_grouped"}


def sha256_array(a: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(a.dtype).encode() + str(a.shape).encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def load_config(path=None) -> list[dict]:
    path = path or (C.ROOT / "configs" / "datasets.yaml")
    with open(path) as f:
        return yaml.safe_load(f)["datasets"]


def export_one(spec: dict, p=None) -> dict:
    p = p or C.paths()
    con = catalog.connect(p.catalog)
    filt = spec.get("filters") or {}
    bad = set(filt) - ALLOWED_FILTER_COLUMNS
    if bad:
        raise ValueError(f"filter columns not allowed: {bad}")
    where = " AND ".join(f"s.{k} = ?" for k in filt) or "1=1"
    meta = pd.read_sql(
        "SELECT s.segment_id, s.measurement_id, m.label, m.class_group, s.paper_split, s.split_grouped"
        f" FROM segments s JOIN measurements m USING(measurement_id) WHERE {where} ORDER BY s.segment_id",
        con,
        params=list(filt.values()),
    )
    con.close()
    source_md5 = json.loads((p.results / "ingest.json").read_text())["source_md5"]
    rd = np.load(p.rd, mmap_mode="r")
    ids = meta.segment_id.to_numpy()
    labels = sorted(meta.label.unique())
    arrays = {
        "x_rd_db": np.asarray(rd[ids], dtype=np.float16),
        "segment_id": ids.astype(np.int64),
        "measurement_id": meta.measurement_id.to_numpy().astype(np.int32),
        "y_group": meta.class_group.map({g: i for i, g in enumerate(C.CLASS_GROUPS)}).to_numpy().astype(np.int8),
        "y_label": meta.label.map({lab: i for i, lab in enumerate(labels)}).to_numpy().astype(np.int8),
        "y_drone": meta.label.map({d: i for i, d in enumerate(C.DRONES)}).fillna(-1).to_numpy().astype(np.int8),
        "paper_split": meta.paper_split.to_numpy().astype(np.int8),
        "split_grouped": meta.split_grouped.to_numpy().astype(np.int8),
    }
    out = p.exports / f"{spec['name']}-v{spec['version']}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "arrays.npz", **arrays)

    def counts(col):
        return {str(s): {g: int(((meta[col] == s) & (meta.class_group == g)).sum()) for g in C.CLASS_GROUPS} for s in (1, 2, 3)}

    manifest = {
        "dataset": spec["name"],
        "version": spec["version"],
        "description": spec.get("description", ""),
        "rdp_version": __version__,
        "source": {"file": p.raw.name, "md5": source_md5, "doi": "10.5281/zenodo.5845259", "license": "CC BY 4.0"},
        "filters": filt,
        "n_segments": int(len(ids)),
        "input": {
            "array": "x_rd_db",
            "shape": list(arrays["x_rd_db"].shape),
            "dtype": "float16",
            "meaning": "10*log10 RD power re per-segment noise floor; axes segment, range cell, Doppler bin (0 Hz at 128)",
        },
        "class_maps": {"y_group": list(C.CLASS_GROUPS), "y_label": labels, "y_drone": list(C.DRONES)},
        "splits": {
            "paper_split": "per-sample split of Karlsson et al. 2022 (1 train, 2 val, 3 test)",
            "split_grouped": "grouped by measurement (rdp.splits.grouped_split, seed 0)",
        },
        "counts_paper_split": counts("paper_split"),
        "counts_grouped_split": counts("split_grouped"),
        "sha256": {k: sha256_array(v) for k, v in arrays.items()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (p.results / f"manifest_{spec['name']}.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def run(p=None) -> list[dict]:
    return [export_one(s, p) for s in load_config()]
