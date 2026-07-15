"""Optimizers: SGD (week 2) and Adam (week 3).

An optimizer holds a reference to the model and, on each ``step()``, reads the
current (parameter, gradient) pairs from ``model.params_and_grads()`` and nudges
the parameters to reduce the loss.

>>> CRITICAL: update parameters IN PLACE. <<<
    p -= lr * g          # correct: mutates the array the model holds
    p[...] = p - lr * g  # also correct
    p = p - lr * g       # WRONG: rebinds a local name, model is unchanged

Because ``p`` is a NumPy array that the model owns, in-place arithmetic changes
the real weights; plain assignment just points the local variable elsewhere.
"""
from __future__ import annotations

import numpy as np


class SGD:
    """Vanilla stochastic gradient descent:  p <- p - lr * grad."""

    def __init__(self, model, lr: float = 0.1):
        self.model = model
        self.lr = lr

    def step(self) -> None:
        # TODO(week2): for each (p, g) in self.model.params_and_grads(),
        # update p in place by -lr * g.
        raise NotImplementedError("SGD.step")


class Adam:
    """Adam (Kingma & Ba, 2014).

    Per parameter, maintain 1st- and 2nd-moment running averages m, v:

        t <- t + 1
        m <- beta1 * m + (1 - beta1) * g
        v <- beta2 * v + (1 - beta2) * g**2
        m_hat <- m / (1 - beta1**t)          # bias correction
        v_hat <- v / (1 - beta2**t)
        p <- p - lr * m_hat / (sqrt(v_hat) + eps)

    m and v start at zeros with the same shape as each parameter. Keep them in
    self.m / self.v as lists indexed the same way as params_and_grads().
    """

    def __init__(self, model, lr: float = 1e-3, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8):
        self.model = model
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        # Lazily initialized on first step, once we know the parameter shapes.
        self.m = None
        self.v = None

    def step(self) -> None:
        pairs = self.model.params_and_grads()
        if self.m is None:
            self.m = [np.zeros_like(p) for p, _ in pairs]
            self.v = [np.zeros_like(p) for p, _ in pairs]
        self.t += 1
        # TODO(week3): implement the Adam update for each i, (p, g) in
        # enumerate(pairs). Remember to update p IN PLACE.
        raise NotImplementedError("Adam.step")
