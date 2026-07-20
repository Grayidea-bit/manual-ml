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
        self.y = y #self.y = correct class label for each sample; cache for backward
        probs = softmax(logits) #Get the confidences for every classes
        self.probs = probs # confidence for each of the 10 classes, per sample
        N = logits.shape[0] # number of samples in this batch (rows)
        correct_probs = probs[np.arange(N), y] # Take the confidence of the real class of the data
        return float(np.mean(-np.log(correct_probs + 1e-12))) # mean cross-entropy loss (lower = better)

    def backward(self) -> np.ndarray:
        onehot = np.zeros_like(self.probs)
        N = self.probs.shape[0]
        onehot[np.arange(N), self.y] = 1
        return onehot
