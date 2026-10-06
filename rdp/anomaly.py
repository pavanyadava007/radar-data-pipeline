"""ML anomaly detectors trained only on clean (edge-free) grouped-train segments."""

from __future__ import annotations

import time

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from rdp.models import ConvAE, normalise_rd


def _signed_log(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


class FeatureDetectors:
    """IsolationForest and PCA reconstruction error on the standardised DSP feature vector."""

    def __init__(self, seed: int = 0, n_components: int = 4):
        self.scaler = StandardScaler()
        self.iforest = IsolationForest(n_estimators=300, random_state=seed, n_jobs=-1)
        self.pca = PCA(n_components=n_components, random_state=seed)

    def _prep(self, f: np.ndarray) -> np.ndarray:
        return self.scaler.transform(_signed_log(f))

    def fit(self, f: np.ndarray):
        z = self.scaler.fit_transform(_signed_log(f))
        self.iforest.fit(z)
        self.pca.fit(z)
        return self

    def score_iforest(self, f: np.ndarray) -> np.ndarray:
        return -self.iforest.score_samples(self._prep(f))

    def score_pca(self, f: np.ndarray) -> np.ndarray:
        z = self._prep(f)
        r = self.pca.inverse_transform(self.pca.transform(z))
        return ((z - r) ** 2).sum(axis=1)


def train_cae(rd_train: np.ndarray, epochs: int = 15, seed: int = 0, batch: int = 512, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ConvAE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.from_numpy(normalise_rd(rd_train)).to(device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    t0 = time.perf_counter()
    losses = []
    for _ in range(epochs):
        perm = torch.randperm(len(x), generator=g)
        tot = 0.0
        for i in range(0, len(x), batch):
            xb = x[perm[i : i + batch]]
            loss = ((model(xb) - xb) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        losses.append(tot / len(x))
    if device == "cuda":
        torch.cuda.synchronize()
    return model.eval(), {
        "train_seconds": round(time.perf_counter() - t0, 2),
        "epochs": epochs,
        "final_loss": round(losses[-1], 5),
        "device": device,
    }


@torch.no_grad()
def score_cae(model, rd: np.ndarray, batch: int = 4096) -> np.ndarray:
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(rd), batch):
        xb = torch.from_numpy(normalise_rd(rd[i : i + batch])).to(dev)
        out.append(((model(xb) - xb) ** 2).mean(dim=(1, 2)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)
