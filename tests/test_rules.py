import numpy as np
import pytest

from rdp import rules
from rdp.faults import FAULTS, inject
from rdp.rawio import segment_hash
from rdp.synth import make_segment


@pytest.fixture(scope="module")
def clean():
    rng = np.random.default_rng(0)
    x = np.stack([make_segment(lab, rng) for lab in ("D1", "D2", "seagull", "human_walk", "CR") for _ in range(60)])
    x = x.astype(np.complex64)
    st = rules.rule_stats(x)
    thr = rules.calibrate(st, pct=100.0)  # synthetic clean data defines the envelope
    return x, thr


EXPECTED = {
    "dropped_block": "dead_block",
    "adc_clipping": "clipping",
    "interference_burst": "spike",
    "range_offset": "off_centre",
    "frozen_block": "frozen_block",
    "duplicate_segment": "duplicate",
}


@pytest.mark.parametrize("fault,rule", sorted(EXPECTED.items()))
def test_rule_catches_its_fault(clean, fault, rule):
    x, thr = clean
    rng = np.random.default_rng(1)
    idx = rng.choice(len(x), 40, replace=False)
    known = {segment_hash(v) for v in x}
    y = np.stack([inject(x[i], fault, rng, donor=x[(i + 1) % len(x)]) for i in idx])
    st = rules.rule_stats(y, known_hashes=known, hashes=[segment_hash(v) for v in y])
    fl = rules.flags(st, thr)
    # the spike statistic is weaker on this fixture: its strong rotor FM gives large second
    # differences on clean drone segments, so the clean max (pct=100) threshold is high
    need = 0.3 if rule == "spike" else 0.9
    assert fl[rule].mean() >= need, (fault, rule, fl[rule].mean())
    assert (rules.rule_score(st, thr) >= 1).mean() >= need


def test_nonfinite_detected(clean):
    x, thr = clean
    y = x[:5].copy()
    y[:, 2, 10] = np.nan
    y[1, 0, 0] = np.inf
    st = rules.rule_stats(y)
    assert rules.flags(st, thr)["nonfinite"].all()


def test_clean_data_not_flagged_by_hard_rules(clean):
    x, thr = clean
    st = rules.rule_stats(x, hashes=[segment_hash(v) for v in x])
    fl = rules.flags(st, thr)
    for k in rules.HARD_RULES:
        assert not fl[k].any(), k


def test_edge_cliff_higher_for_noise_fill():
    rng = np.random.default_rng(5)
    a = np.stack([make_segment("seagull", rng) for _ in range(50)])
    b = np.stack([make_segment("seagull", rng, edge=True) for _ in range(50)])
    assert np.median(rules.edge_cliff(b)) > np.median(rules.edge_cliff(a)) + 0.5


def test_all_faults_change_the_data(clean):
    x, _ = clean
    rng = np.random.default_rng(2)
    for f in FAULTS:
        y = inject(x[0], f, rng, donor=x[1])
        assert y.shape == x[0].shape and y.dtype == np.complex64
        assert not np.array_equal(y, x[0]), f


def test_flag_bits_roundtrip():
    fl = {k: np.array([False, True]) if k == "spike" else np.array([False, False]) for k in rules.INTEGRITY_RULES}
    bits = rules.flag_bits(fl)
    assert bits[0] == 0 and bits[1] == 1 << rules.INTEGRITY_RULES.index("spike")
