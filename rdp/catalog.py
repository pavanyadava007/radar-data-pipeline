"""SQLite catalogue (stdlib sqlite3): measurements + segments."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from rdp.dsp import FEATURES

SCHEMA = """
CREATE TABLE IF NOT EXISTS measurements (
    measurement_id INTEGER PRIMARY KEY,
    label TEXT NOT NULL,
    class_group TEXT NOT NULL CHECK (class_group IN ('drone','bird','human','reflector')),
    n_segments INTEGER NOT NULL,
    duration_s REAL NOT NULL,
    split_grouped INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS segments (
    segment_id INTEGER PRIMARY KEY,
    measurement_id INTEGER NOT NULL REFERENCES measurements(measurement_id),
    idx_in_measurement INTEGER NOT NULL,
    range_m REAL NOT NULL,
    time_s REAL NOT NULL,
    gap_before_s REAL,
    paper_split INTEGER NOT NULL,
    split_grouped INTEGER NOT NULL,
    edge_flag INTEGER NOT NULL,
    sha1 TEXT NOT NULL,
    {features},
    rule_flags INTEGER,
    rule_score REAL,
    low_snr INTEGER,
    time_gap INTEGER,
    anomaly_score REAL,
    anomaly_score_iforest REAL
);
CREATE INDEX IF NOT EXISTS ix_seg_meas ON segments(measurement_id);
CREATE INDEX IF NOT EXISTS ix_seg_split ON segments(paper_split, split_grouped);
"""


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path)


def create(con: sqlite3.Connection) -> None:
    con.executescript("DROP TABLE IF EXISTS segments; DROP TABLE IF EXISTS measurements;")
    con.executescript(SCHEMA.format(features=",\n    ".join(f"{f} REAL" for f in FEATURES)))
    con.commit()


def update_columns(con: sqlite3.Connection, segment_ids, columns: dict) -> None:
    names = list(columns)
    sql = f"UPDATE segments SET {', '.join(f'{n}=?' for n in names)} WHERE segment_id=?"
    rows = zip(*[[None if v != v else float(v) for v in columns[n]] for n in names], [int(s) for s in segment_ids])
    con.executemany(sql, rows)
    con.commit()
