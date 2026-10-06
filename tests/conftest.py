import os
from pathlib import Path

import pytest

from rdp.synth import make_fixture


@pytest.fixture(scope="session")
def pipeline(tmp_path_factory):
    """Run ingest -> dsp -> quality -> export -> sql on a synthetic fixture (no download)."""
    root = tmp_path_factory.mktemp("rdp")
    raw = make_fixture(root / "raw" / "fixture.npy")
    env = {"RDP_RAW": str(raw), "RDP_DATA": str(root / "data"), "RDP_RESULTS": str(root / "results")}
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    (root / "results").mkdir()
    from rdp import config as C
    from rdp import dspstage, export, ingest, quality, query

    out = {"root": root, "paths": C.paths()}
    out["ingest"] = ingest.run()
    out["dsp"] = dspstage.run()
    out["quality"] = quality.run(n_per_fault=20, cae_epochs=2)
    out["export"] = export.run()
    out["sql"] = query.run_all(Path(__file__).resolve().parent.parent / "sql")
    yield out
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
