"""Padding and normalization for variable-length spectral sequences."""

from __future__ import annotations

import numpy as np


def pad_sequences(sequences: list[np.ndarray], t_max: int | None = None):
    """Zero-pad a list of ``(T_i, n_bands)`` arrays to a common length.

    Returns
    -------
    x : np.ndarray
        Padded tensor of shape ``(n_samples, t_max, n_bands)``.
    mask : np.ndarray
        Boolean array of shape ``(n_samples, t_max)``; True marks a frame
        that comes from the original sequence.
    """
    n_bands = sequences[0].shape[1]
    if any(s.shape[1] != n_bands for s in sequences):
        raise ValueError("All sequences must have the same number of bands.")

    t_max = t_max or max(s.shape[0] for s in sequences)
    x = np.zeros((len(sequences), t_max, n_bands), dtype=np.float32)
    mask = np.zeros((len(sequences), t_max), dtype=bool)

    for i, seq in enumerate(sequences):
        t = min(seq.shape[0], t_max)
        x[i, :t] = seq[:t]
        mask[i, :t] = True

    return x, mask


class ZScoreScaler:
    """Per-band standardization fitted on valid (non-padded) frames only.

    Padded positions are excluded from the statistics and are written back
    as zeros after the transform, so padding stays neutral.
    """

    def __init__(self, eps: float = 1e-5) -> None:
        self.eps = eps
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, x: np.ndarray, mask: np.ndarray) -> "ZScoreScaler":
        valid = x[mask]                       # (n_valid_frames, n_bands)
        if valid.size == 0:
            raise ValueError("The mask selects no frames; cannot fit.")
        self.mean_ = valid.mean(axis=0)
        self.std_ = np.sqrt(valid.var(axis=0) + self.eps)
        return self

    def transform(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Call fit() before transform().")
        out = (x - self.mean_) / self.std_
        return (out * mask[..., None]).astype(np.float32)

    def fit_transform(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self.fit(x, mask).transform(x, mask)
