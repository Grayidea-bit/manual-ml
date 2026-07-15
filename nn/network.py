"""MLP container that chains layers together. (Provided scaffolding.)

You build the network by listing layer sizes, e.g. ``MLP([784, 128, 64, 10])``
gives Linear(784,128) -> ReLU -> Linear(128,64) -> ReLU -> Linear(64,10).

The forward pass walks the layers in order; the backward pass walks them in
reverse, threading the gradient through each ``backward()``. This is the whole
idea of backpropagation, made literal.
"""
from __future__ import annotations

import numpy as np

from .activations import ReLU
from .layers import Linear


class MLP:
    def __init__(self, sizes: list[int], rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng()
        self.layers = []
        for i in range(len(sizes) - 1):
            self.layers.append(Linear(sizes[i], sizes[i + 1], rng=rng))
            if i < len(sizes) - 2:  # ReLU after every hidden layer, not after the output
                self.layers.append(ReLU())

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x  # logits

    def backward(self, grad: np.ndarray) -> None:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def params_and_grads(self):
        pairs = []
        for layer in self.layers:
            pairs.extend(layer.params_and_grads())
        return pairs
