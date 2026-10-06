"""Leakage-safe split grouped by measurement (no measurement appears in two splits).

Measurements that share an exactly identical segment (found by sha1 at ingest; the
raw data contains such cross-measurement copies) are merged into one group first, so
a copied segment cannot sit in train and test at the same time.
"""

from __future__ import annotations

import numpy as np

from rdp.config import GROUPED_FRACTIONS


def _components(n: int, links: list[tuple[int, int]]) -> list[int]:
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in links:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return [find(i) for i in range(n)]


def grouped_split(labels: list[str], sizes: list[int], links: list[tuple[int, int]] | None = None, seed: int = 0) -> list[int]:
    """Assign each measurement to 1=train, 2=val, 3=test.

    Groups (connected measurements) are labelled by their largest member. Per label,
    groups are visited largest first and each goes to the split with the largest
    segment-count deficit w.r.t. GROUPED_FRACTIONS. Labels with >= 3 groups get their two
    smallest groups in test and val first; labels with 2 groups go train + test;
    labels with 1 group go to train.
    """
    rng = np.random.default_rng(seed)
    comp = _components(len(labels), links or [])
    groups: dict[int, list[int]] = {}
    for i, c in enumerate(comp):
        groups.setdefault(c, []).append(i)
    g_label = {c: labels[max(m, key=lambda i: sizes[i])] for c, m in groups.items()}
    g_size = {c: sum(sizes[i] for i in m) for c, m in groups.items()}
    g_split: dict[int, int] = {}
    for lab in sorted(set(g_label.values())):
        idx = [c for c in groups if g_label[c] == lab]
        rng.shuffle(idx)
        idx.sort(key=lambda c: -g_size[c])
        total = sum(g_size[c] for c in idx)
        got = np.zeros(3)
        if len(idx) == 1:
            g_split[idx[0]] = 1
            continue
        if len(idx) == 2:
            g_split[idx[0]], g_split[idx[1]] = 1, 3
            continue
        g_split[idx[-1]], g_split[idx[-2]] = 3, 2
        got[2] += g_size[idx[-1]]
        got[1] += g_size[idx[-2]]
        for c in idx[:-2]:
            s = int(np.argmax(np.array(GROUPED_FRACTIONS) * total - got))
            g_split[c] = s + 1
            got[s] += g_size[c]
    return [g_split[comp[i]] for i in range(len(labels))]
