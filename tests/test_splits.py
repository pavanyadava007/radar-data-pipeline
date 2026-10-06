from rdp.splits import grouped_split


def test_grouped_split_basic():
    labels = ["a"] * 5 + ["b"] * 3 + ["c"] * 2 + ["d"]
    sizes = [100, 80, 60, 40, 20, 50, 50, 50, 30, 30, 10]
    s = grouped_split(labels, sizes)
    assert len(s) == len(labels) and set(s) <= {1, 2, 3}
    for lab in ("a", "b"):
        assert {s[i] for i, ell in enumerate(labels) if ell == lab} == {1, 2, 3}
    assert sorted(s[8:10]) == [1, 3]
    assert s[10] == 1


def test_linked_measurements_share_split():
    labels = ["a"] * 6
    sizes = [10, 10, 10, 10, 10, 10]
    s = grouped_split(labels, sizes, links=[(0, 5), (5, 3)])
    assert s[0] == s[5] == s[3]


def test_deterministic():
    labels = ["a"] * 7 + ["b"] * 4
    sizes = list(range(10, 120, 10))
    assert grouped_split(labels, sizes, seed=3) == grouped_split(labels, sizes, seed=3)
