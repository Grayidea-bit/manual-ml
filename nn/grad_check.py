"""Numerical gradient checking. (Provided — your safety net for weeks 1-2.)

This is the single most useful tool for catching a wrong backward(). It compares
your analytic gradients against a numerical estimate computed purely from
forward passes (central difference):

    dL/dp  ~=  ( L(p + eps) - L(p - eps) ) / (2 * eps)

If your backward() is correct, the relative error between the two is tiny
(< 1e-6 in float64). A big error points straight at the layer you got wrong.

Run it once the forward + backward TODOs are done::

    python -m nn.grad_check
"""
from __future__ import annotations

import numpy as np

from .losses import SoftmaxCrossEntropy
from .network import MLP


def numerical_gradient(loss_fn, param: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Central-difference gradient of ``loss_fn()`` w.r.t. every entry of ``param``.

    ``loss_fn`` is a zero-argument callable that recomputes the scalar loss using
    the network's *current* parameters. ``param`` is mutated and restored in place.
    """
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=["multi_index"], op_flags=[["readwrite"]])
    while not it.finished:
        idx = it.multi_index
        original = param[idx]
        param[idx] = original + eps
        plus = loss_fn()
        param[idx] = original - eps
        minus = loss_fn()
        param[idx] = original
        grad[idx] = (plus - minus) / (2 * eps)
        it.iternext()
    return grad


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.maximum(1e-12, np.abs(a) + np.abs(b))
    return float(np.max(np.abs(a - b) / denom))


def check_model(seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    # Small net + small batch so numerical checking is fast. float64 for precision.
    model = MLP([6, 5, 4, 3], rng=rng)
    for layer in model.layers:
        if hasattr(layer, "W"):
            layer.W = layer.W.astype(np.float64)
            layer.b = layer.b.astype(np.float64)

    X = rng.standard_normal((4, 6))
    y = rng.integers(0, 3, size=4)
    loss_fn = SoftmaxCrossEntropy()

    def loss_only() -> float:
        return loss_fn.forward(model.forward(X), y)

    # Analytic gradients.
    loss_only()
    model.backward(loss_fn.backward())

    worst = 0.0
    for li, layer in enumerate(model.layers):
        if not hasattr(layer, "W"):
            continue
        for name, p, g in (("W", layer.W, layer.dW), ("b", layer.b, layer.db)):
            num = numerical_gradient(loss_only, p)
            err = relative_error(g, num)
            worst = max(worst, err)
            status = "ok" if err < 1e-6 else "FAIL"
            print(f"layer {li} {name}: rel_error = {err:.3e}  [{status}]")

    print("-" * 40)
    print(f"worst relative error = {worst:.3e}", "PASS" if worst < 1e-6 else "CHECK YOUR BACKWARD")


if __name__ == "__main__":
    check_model()
