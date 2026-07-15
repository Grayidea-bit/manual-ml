"""Activation functions.

ReLU is the only activation we need for the hidden layers. Softmax is defined
here for inspection, but in training it is fused with cross-entropy (see
losses.py) for numerical stability and a much simpler gradient.

------------------------------------------------------------------------------
YOU IMPLEMENT: ReLU.forward / ReLU.backward, and softmax().
------------------------------------------------------------------------------

ReLU(x) = max(0, x), elementwise.
    forward:  y = max(0, x)          (cache the mask x > 0)
    backward: dL/dx = grad_output * (x > 0)   -- gradient flows only where x was positive
"""
from __future__ import annotations

import numpy as np


class ReLU:
    def __init__(self):
        self.mask = None  # cache: where the input was positive

    def forward(self, x: np.ndarray) -> np.ndarray:
        # TODO(week1): cache the positive mask, return max(0, x)
        raise NotImplementedError("ReLU.forward")

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # TODO(week2): pass the gradient through only where the input was positive
        raise NotImplementedError("ReLU.backward")

    def params_and_grads(self):
        return []  # ReLU has no parameters


def softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax. logits: (N, C) -> probs: (N, C).

    Numerically stable version: subtract the per-row max before exp so we never
    exponentiate a large positive number.

        p_i = exp(z_i - max(z)) / sum_j exp(z_j - max(z))
    """
    # TODO(week1): implement the stable row-wise softmax
    raise NotImplementedError("softmax")
