"""Softmax + cross-entropy loss, fused into one block.

Why fuse them? The gradient of cross-entropy w.r.t. the logits collapses to the
famously clean form ``(softmax(logits) - onehot(y))``, which is both numerically
stable and cheap. Deriving that cancellation is one of the most important
exercises in this whole project (see notes/backprop_math.md).

------------------------------------------------------------------------------
YOU IMPLEMENT: forward() and backward().
------------------------------------------------------------------------------

forward(logits, y):
    logits : (N, C) raw scores
    y      : (N,)   integer class labels in [0, C)
    probs  = softmax(logits)                              # (N, C)
    loss   = -mean_over_batch( log( probs[i, y[i]] ) )    # scalar
    (cache probs and y for backward)

backward():
    returns dL/dlogits with shape (N, C):
        (probs - onehot(y)) / N
    The division by N matches the mean() taken in forward.
"""
from __future__ import annotations

import numpy as np

from .activations import softmax


class SoftmaxCrossEntropy:
    def __init__(self):
        self.probs = None  # cache: softmax output, (N, C)
        self.y = None      # cache: labels, (N,)

    def forward(self, logits: np.ndarray, y: np.ndarray) -> float:
        # TODO(week1): compute probs = softmax(logits); cache probs and y;
        # return the mean cross-entropy loss as a Python float.
        # Hint: add a tiny epsilon (e.g. 1e-12) inside log to avoid log(0).
        raise NotImplementedError("SoftmaxCrossEntropy.forward")

    def backward(self) -> np.ndarray:
        # TODO(week2): return (probs - onehot(y)) / N using the cached values.
        # Hint to build onehot: start from np.zeros_like(self.probs) and set
        # the column given by each label to 1.
        raise NotImplementedError("SoftmaxCrossEntropy.backward")
