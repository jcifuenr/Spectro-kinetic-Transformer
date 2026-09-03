"""Minimal training loop for the Spectro-Kinetic Transformer."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from model import SpectroKineticTransformer


def train(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    n_classes: int | None = None,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    activation: str = "gelu",
    seed: int = 42,
    device: str | None = None,
    **model_kwargs,
):
    """Train the model and return it together with the epoch history.

    Inputs are normalized, padded tensors of shape ``(n, t_max, n_bands)``.
    When a validation set is given, the weights of the epoch with the highest
    validation accuracy are restored before returning.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = n_classes or int(y_train.max()) + 1

    model = SpectroKineticTransformer(
        n_bands=x_train.shape[2],
        t_max=x_train.shape[1],
        n_classes=n_classes,
        activation=activation,
        **model_kwargs,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    xt = torch.as_tensor(x_train, dtype=torch.float32)
    yt = torch.as_tensor(y_train, dtype=torch.long)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xt, yt), batch_size=batch_size, shuffle=False
    )

    history: list[dict] = []
    best_acc, best_state = -1.0, None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        record = {"epoch": epoch + 1, "train_loss": total_loss / len(yt)}

        if x_val is not None:
            val_loss, val_acc = evaluate(model, x_val, y_val, device, batch_size)
            record.update(val_loss=val_loss, val_accuracy=val_acc)
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        history.append(record)

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


@torch.no_grad()
def evaluate(model, x, y, device: str, batch_size: int = 32):
    """Return (mean cross-entropy, accuracy) over the given set."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    xt = torch.as_tensor(x, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.long)

    total_loss, correct = 0.0, 0
    for i in range(0, len(yt), batch_size):
        xb = xt[i : i + batch_size].to(device)
        yb = yt[i : i + batch_size].to(device)
        logits = model(xb)
        total_loss += criterion(logits, yb).item()
        correct += (logits.argmax(dim=1) == yb).sum().item()

    return total_loss / len(yt), correct / len(yt)


@torch.no_grad()
def predict(model, x, device: str | None = None, batch_size: int = 32):
    """Return (predicted labels, pooled embeddings)."""
    device = device or next(model.parameters()).device
    model.eval()
    xt = torch.as_tensor(x, dtype=torch.float32)

    preds, embeddings = [], []
    for i in range(0, len(xt), batch_size):
        logits, emb = model(xt[i : i + batch_size].to(device), return_embedding=True)
        preds.append(logits.argmax(dim=1).cpu().numpy())
        embeddings.append(emb.cpu().numpy())

    return np.concatenate(preds), np.concatenate(embeddings)
