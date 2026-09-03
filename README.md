# Spectro-Kinetic Transformer

Transformer encoder for classifying spectro-kinetic combustion emission
sequences. Each sample is a variable-length sequence of spectral frames; the
model returns one class label per sequence.

This repository contains the model and the preprocessing it requires, nothing
else. Evaluation, cross-validation and figure generation are left to the
caller.

## Architecture

Spectral frames are projected linearly into a latent space of dimension
`d_model`, a learned positional embedding is added, and the sequence passes
through `num_layers` standard Transformer encoder blocks (multi-head
attention and a position-wise feed-forward sublayer, each with a residual
connection and layer normalization). The encoder output is averaged over the
temporal axis and passed to a linear classifier.

Sequences are zero-padded to a common length before being passed in. Padded
frames take part in attention and in the pooling, which is the configuration
used for the reported results. They are excluded only from the normalization
statistics.

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10 or later.

## Usage

```python
import numpy as np
from data import pad_sequences, ZScoreScaler
from train import train, evaluate, predict

# sequences: list of (T_i, n_bands) arrays; labels: integer array
x_train, mask_train = pad_sequences(sequences_train)
x_test, mask_test = pad_sequences(sequences_test, t_max=x_train.shape[1])

scaler = ZScoreScaler()
x_train = scaler.fit_transform(x_train, mask_train)
x_test = scaler.transform(x_test, mask_test)

model, history = train(x_train, y_train, x_test, y_test, n_classes=5)
loss, accuracy = evaluate(model, x_test, y_test, device="cpu")
labels, embeddings = predict(model, x_test)
```

The scaler is fitted on the training set only. The test set is padded to the
same `t_max` as the training set, since the positional table has a fixed
size.

## Configuration

Defaults reproduce the reported configuration.

| Parameter | Default | Description |
|---|---|---|
| `d_model` | 128 | Latent dimensionality |
| `num_heads` | 4 | Attention heads; head dimension is `d_model / num_heads` |
| `num_layers` | 2 | Encoder blocks |
| `d_ff` | 512 | Inner dimension of the feed-forward sublayer |
| `encoder_dropout` | 0.1 | Dropout inside the encoder blocks |
| `head_dropout` | 0.3 | Dropout in the classification head |
| `activation` | `gelu` | Feed-forward activation; `relu` also accepted |
| `lr` | 1e-3 | AdamW learning rate, constant |
| `weight_decay` | 0.01 | AdamW weight decay |
| `epochs` | 100 | Training epochs |
| `batch_size` | 32 | Batch size |

Training uses AdamW with a constant learning rate, no gradient clipping and
batches in a fixed order. When a validation set is supplied, the weights of
the epoch with the highest validation accuracy are restored.

## Files

| File | Contents |
|---|---|
| `model.py` | `SpectroKineticTransformer` |
| `data.py` | Padding, masking and per-band z-score normalization |
| `train.py` | Training loop, evaluation and inference |

## License

MIT. See `LICENSE`.
