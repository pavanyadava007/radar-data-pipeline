"""Small PyTorch models: convolutional autoencoder (anomaly) and 1D CNN classifier.

Both take a log range-Doppler map [B, 5, 256] (dB re noise floor) and treat the 5 range
cells as channels of a 1D signal over Doppler.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def normalise_rd(rd: np.ndarray | torch.Tensor):
    """dB re noise floor -> roughly [-0.2, 2]. Shared by the AE, the classifier and the Dataset."""
    if isinstance(rd, np.ndarray):
        return np.clip(rd.astype(np.float32), -5.0, 60.0) / 30.0
    return torch.clamp(rd.float(), -5.0, 60.0) / 30.0


class ConvAE(nn.Module):
    def __init__(self, latent: int = 32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(5, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 32, latent),
        )
        self.dec_fc = nn.Linear(latent, 64 * 32)
        self.dec = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose1d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 5, 4, stride=2, padding=1),
        )

    def forward(self, x):
        z = self.enc(x)
        return self.dec(self.dec_fc(z).view(-1, 64, 32))


class RDClassifier(nn.Module):
    def __init__(self, n_classes: int, width: int = 32):
        super().__init__()
        w = width

        def block(i, o):
            return nn.Sequential(nn.Conv1d(i, o, 7, padding=3), nn.BatchNorm1d(o), nn.ReLU(), nn.MaxPool1d(2))

        self.features = nn.Sequential(block(5, w), block(w, 2 * w), block(2 * w, 4 * w), block(4 * w, 4 * w))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(8 * w, n_classes))

    def forward(self, x):
        h = self.features(x)
        h = torch.cat([h.mean(-1), h.amax(-1)], dim=1)
        return self.head(h)
