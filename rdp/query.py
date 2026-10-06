"""Run SQL files against the catalogue and print / save the result."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rdp import catalog
from rdp import config as C


def run_sql(sql_file: Path, p=None) -> pd.DataFrame:
    p = p or C.paths()
    con = catalog.connect(p.catalog)
    try:
        return pd.read_sql(Path(sql_file).read_text(), con)
    finally:
        con.close()


def run_all(sql_dir: Path | None = None, p=None) -> dict[str, pd.DataFrame]:
    p = p or C.paths()
    sql_dir = sql_dir or (C.ROOT / "sql")
    out = {}
    for f in sorted(Path(sql_dir).glob("*.sql")):
        df = run_sql(f, p)
        df.to_csv(p.results / f"sql_{f.stem}.csv", index=False)
        out[f.stem] = df
    return out
