"""Linear (fully-connected) layer.

    y = x @ W + b

Shapes (N = batch size):
    x : (N, in_features)
    W : (in_features, out_features)
    b : (out_features,)
    y : (N, out_features)

------------------------------------------------------------------------------
YOU IMPLEMENT: forward() and backward().  __init__ (weight init) is provided.
------------------------------------------------------------------------------

Backward — given dL/dy (grad_output, shape (N, out_features)), compute:

    dL/dW = x.T @ grad_output          shape (in_features, out_features)
    dL/db = sum over batch of grad     shape (out_features,)   -> grad_output.sum(axis=0)
    dL/dx = grad_output @ W.T          shape (N, in_features)

Store dL/dW in self.dW and dL/db in self.db (the optimizer reads them there).
Return dL/dx so the previous layer can continue the chain.

Derivation lives in notes/backprop_math.md.
"""
from __future__ import annotations

import numpy as np


class Linear:
    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng()
        # He initialization: good default for ReLU networks. Not the learning
        # target here, so it's provided for you.
        std = np.sqrt(2.0 / in_features)
        self.W = (rng.standard_normal((in_features, out_features)) * std).astype(np.float32)
        self.b = np.zeros(out_features, dtype=np.float32)

        # Gradients, filled by backward().
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        # Cache for backward().
        self.x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # TODO(week1): cache x for backward, then return x @ W + b
        raise NotImplementedError("Linear.forward")

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # TODO(week2): set self.dW, self.db from the cached x and grad_output,
        # then return the gradient w.r.t. the input (dL/dx).
        raise NotImplementedError("Linear.backward")

    def params_and_grads(self):
        return [(self.W, self.dW), (self.b, self.db)]
