"""Catalogue schema, manifest determinism and a tiny end-to-end run on the synthetic fixture."""

import json
import sqlite3

import numpy as np
import pandas as pd

from rdp import dsp
from rdp.dataset import RadarRDDataset
from rdp.synth import LABELS


def test_ingest_counts(pipeline):
    r = pipeline["ingest"]
    assert r["n_measurements"] == len(LABELS) * 3
    assert r["n_segments"] == len(LABELS) * 3 * 40
    assert r["n_edge_flagged"] == 3
    assert r["n_exact_duplicate_segments"] == 1


def test_catalog_schema(pipeline):
    con = sqlite3.connect(pipeline["paths"].catalog)
    cols = {r[1] for r in con.execute("PRAGMA table_info(segments)")}
    need = {
        "segment_id",
        "measurement_id",
        "idx_in_measurement",
        "range_m",
        "time_s",
        "paper_split",
        "edge_flag",
        "snr_db",
        "peak_doppler_hz",
        "doppler_spread_hz",
        "rule_flags",
        "anomaly_score",
        "split_grouped",
    }
    assert need <= cols
    assert set(dsp.FEATURES) <= cols
    mcols = {r[1] for r in con.execute("PRAGMA table_info(measurements)")}
    assert {"measurement_id", "label", "class_group", "n_segments", "duration_s"} <= mcols
    assert con.execute("SELECT COUNT(*) FROM segments WHERE snr_db IS NULL OR rule_flags IS NULL").fetchone()[0] == 0
    groups = {r[0] for r in con.execute("SELECT DISTINCT class_group FROM measurements")}
    assert groups == {"drone", "bird", "human", "reflector"}


def test_no_measurement_in_two_grouped_splits(pipeline):
    con = sqlite3.connect(pipeline["paths"].catalog)
    n = con.execute(
        "SELECT COUNT(*) FROM (SELECT measurement_id FROM segments GROUP BY measurement_id"
        " HAVING COUNT(DISTINCT split_grouped) > 1)"
    ).fetchone()[0]
    assert n == 0
    leak = pipeline["sql"]["split_leakage"].set_index("split_kind")
    assert leak.loc["split_grouped", "measurements_in_multiple_splits"] == 0


def test_duplicate_flagged(pipeline):
    con = sqlite3.connect(pipeline["paths"].catalog)
    df = pd.read_sql("SELECT rule_flags FROM segments", con)
    from rdp.rules import INTEGRITY_RULES

    bit = 1 << INTEGRITY_RULES.index("duplicate")
    assert int(((df.rule_flags & bit) > 0).sum()) == 1


def test_anomaly_results_structure(pipeline):
    an = json.loads((pipeline["paths"].results / "anomaly.json").read_text())
    rows = list(an["table"])
    assert rows[0].startswith("edge_truncated (REAL")
    assert all("(injected, synthetic)" in r for r in rows[1:])
    for r in rows:
        for m in ("rules", "iforest", "pca", "cae", "rules_or_cae"):
            v = an["table"][r][m]
            assert 0 <= v["recall_deployed"] <= 1 and 0 <= v["roc_auc"] <= 1
    # hard-rule faults are caught by rules on the fixture too
    for f in ("dropped_block", "frozen_block", "duplicate_segment", "adc_clipping"):
        k = next(x for x in rows if x.startswith(f))
        assert an["table"][k]["rules"]["recall_deployed"] == 1.0


def test_manifest_and_determinism(pipeline):
    from rdp import export

    m1 = pipeline["export"]
    m2 = export.run()
    assert [m["sha256"] for m in m1] == [m["sha256"] for m in m2]
    clean = next(m for m in m1 if m["dataset"] == "rd_clean")
    assert clean["filters"] == {"edge_flag": 0, "rule_flags": 0}
    assert clean["source"]["md5"] == pipeline["ingest"]["source_md5"]
    total = sum(sum(v.values()) for v in clean["counts_grouped_split"].values())
    assert total == clean["n_segments"]


def test_dataset_and_training(pipeline):
    from rdp import train

    root = pipeline["paths"].exports / "rd_clean-v1.0.0"
    ds = RadarRDDataset(root, "grouped", 1, "group", verify=True)
    x, y = ds[0]
    assert tuple(x.shape) == (5, 256) and 0 <= y < 4
    ds6 = RadarRDDataset(root, "paper", None, "drone")
    assert set(np.unique(ds6.y)) <= set(range(6))
    r = train.train_one(root, "group", "grouped", 0, {"clean": root}, epochs=2, device="cpu")
    assert 0 <= r["eval"]["clean"]["accuracy"] <= 1
    assert np.array(r["eval"]["clean"]["confusion"]).shape == (4, 4)


def test_cli_query(pipeline, capsys):
    from pathlib import Path

    from rdp.cli import main

    main(["query", str(Path(__file__).resolve().parent.parent / "sql" / "class_balance_per_split.sql")])
    out = capsys.readouterr().out
    assert "drone" in out and "grouped_test" in out
