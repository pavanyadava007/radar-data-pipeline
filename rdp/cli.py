"""Command line interface: python -m rdp <command>."""

from __future__ import annotations

import argparse
import json
import sys
import time


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="rdp", description="Radar data pipeline (77 GHz FMCW, SAAB SIRS 1600 data)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="raw .npy -> HDF5 + SQLite catalogue")
    sub.add_parser("dsp", help="range-Doppler, CFAR, SNR, spectrogram features")
    q = sub.add_parser("quality", help="rule checks + ML anomaly detection + fault-injection evaluation")
    q.add_argument("--n-per-fault", type=int, default=500)
    q.add_argument("--cae-epochs", type=int, default=15)
    sub.add_parser("export", help="versioned training datasets (configs/datasets.yaml)")
    t = sub.add_parser("train", help="CNN classifier, paper vs grouped split, filtered vs unfiltered")
    t.add_argument("--epochs", type=int, default=12)
    t.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    qq = sub.add_parser("query", help="run one SQL file against the catalogue")
    qq.add_argument("sql_file")
    sub.add_parser("sql-all", help="run every sql/*.sql and save results/sql_*.csv")
    a = ap.parse_args(argv)

    t0 = time.perf_counter()
    if a.cmd == "ingest":
        from rdp import ingest

        r = ingest.run()
        print(json.dumps({k: r[k] for k in ("n_measurements", "n_segments", "n_edge_flagged", "source_md5")}, indent=2))
    elif a.cmd == "dsp":
        from rdp import dspstage

        r = dspstage.run()
        print(json.dumps(r["snr_db_by_class_group"], indent=2))
    elif a.cmd == "quality":
        from rdp import quality

        r = quality.run(n_per_fault=a.n_per_fault, cae_epochs=a.cae_epochs)
        print(json.dumps(r["false_alarm_rate_clean_test"], indent=2))
    elif a.cmd == "export":
        from rdp import export

        for m in export.run():
            print(m["dataset"], m["version"], m["n_segments"], m["sha256"]["x_rd_db"][:16])
    elif a.cmd == "train":
        from rdp import train

        train.run(seeds=tuple(a.seeds), epochs=a.epochs)
    elif a.cmd == "query":
        import pandas as pd

        from rdp.query import run_sql

        with pd.option_context("display.width", 200, "display.max_columns", 30, "display.max_rows", 200):
            print(run_sql(a.sql_file).to_string(index=False))
    elif a.cmd == "sql-all":
        from rdp.query import run_all

        for name, df in run_all().items():
            print(f"-- {name}: {len(df)} rows")
    print(f"[rdp {a.cmd}] done in {time.perf_counter() - t0:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
