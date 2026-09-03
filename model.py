"""Spectro-Kinetic Transformer: encoder for combustion emission sequences."""

from __future__ import annotations

import torch
import torch.nn as nn


class SpectroKineticTransformer(nn.Module):
    """Encoder-only Transformer for variable-length spectral sequences.

    A sequence of ``n_bands``-dimensional spectral frames is projected into a
    latent space of size ``d_model``, combined with a learned positional
    embedding, passed through ``num_layers`` pre-defined encoder blocks, and
    reduced to a single vector by average pooling over the temporal axis.

    Sequences are zero-padded to ``t_max`` before being passed in. Padded
    frames take part in attention and in the pooling; this reproduces the
    configuration used for the reported results.

    Parameters
    ----------
    n_bands : int
        Number of spectral bands per frame.
    t_max : int
        Padded sequence length. Fixes the size of the positional table.
    n_classes : int
        Number of output classes.
    """

    def __init__(
        self,
        n_bands: int,
        t_max: int,
        n_classes: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 512,
        encoder_dropout: float = 0.1,
        head_dropout: float = 0.3,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})."
            )

        self.t_max = t_max
        self.input_proj = nn.Linear(n_bands, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(t_max, d_model))
        nn.init.normal_(self.pos_embedding, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=encoder_dropout,
            activation=activation,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(head_dropout),
        )
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, t_max, n_bands)``.
        return_embedding : bool
            If True, also return the pooled representation.

        Returns
        -------
        torch.Tensor or tuple
            Logits of shape ``(batch, n_classes)``, and the pooled embedding
            of shape ``(batch, d_model)`` when requested.
        """
        if x.shape[1] != self.t_max:
            raise ValueError(
                f"Expected sequences of length {self.t_max}, got {x.shape[1]}. "
                f"Pad the batch to t_max before calling forward()."
            )

        h = self.input_proj(x) + self.pos_embedding.unsqueeze(0)
        h = self.encoder(h)
        pooled = h.mean(dim=1)
        embedding = self.head(pooled)
        logits = self.classifier(embedding)

        if return_embedding:
            return logits, embedding
        return logits
