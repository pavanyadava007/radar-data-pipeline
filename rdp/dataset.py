"""PyTorch Dataset over an exported, versioned training set."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from rdp.models import normalise_rd


class RadarRDDataset(Dataset):
    """Items are (x[5, 256] float32, y int64).

    split_kind: "paper" or "grouped"; split: 1 train, 2 val, 3 test, None = all.
    task: "group" (drone/bird/human/reflector) or "drone" (D1-D6, non-drones dropped).
    """

    def __init__(self, root: Path, split_kind: str = "grouped", split: int | None = 1, task: str = "group", verify: bool = False):
        root = Path(root)
        self.manifest = json.loads((root / "manifest.json").read_text())
        z = np.load(root / "arrays.npz")
        if verify:
            from rdp.export import sha256_array

            for k, h in self.manifest["sha256"].items():
                if sha256_array(z[k]) != h:
                    raise ValueError(f"checksum mismatch for {k}")
        col = {"paper": "paper_split", "grouped": "split_grouped"}[split_kind]
        y = z["y_group"] if task == "group" else z["y_drone"]
        keep = y >= 0
        if split is not None:
            keep &= z[col] == split
        self.y = y[keep].astype(np.int64)
        self.x = z["x_rd_db"][keep]
        self.segment_id = z["segment_id"][keep]
        self.classes = self.manifest["class_maps"]["y_group" if task == "group" else "y_drone"]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(normalise_rd(self.x[i])), int(self.y[i])

    def tensors(self, device: str = "cpu"):
        """Whole split as tensors (fast path for small data)."""
        return torch.from_numpy(normalise_rd(self.x)).to(device), torch.from_numpy(self.y).to(device)
