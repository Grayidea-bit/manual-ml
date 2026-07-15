"""Weeks 3-4: compare SGD vs Adam vs a simple evolutionary algorithm.

The harness (data, fair identical initialization, training records, plotting) is
provided. What you implement in week 4 is the evolutionary optimizer in
``evolutionary_train`` — the gradient-free baseline that makes the comparison to
"Gradient descent / Evolutionary algorithms" (a THI keyword) concrete.

Run once SGD + Adam are done (EA can come later)::

    python experiments/optimizer_compare.py

Produces experiments/optimizer_compare.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: write a file, don't open a window
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.download_mnist import load_mnist
from nn.losses import SoftmaxCrossEntropy
from nn.network import MLP
from nn.optimizers import SGD, Adam

SEED = 0
SIZES = [784, 128, 64, 10]
EPOCHS = 10
BATCH = 128


def fresh_model():
    # Same seed every time -> every optimizer starts from identical weights.
    return MLP(SIZES, rng=np.random.default_rng(SEED))


def minibatches(X, y, rng):
    idx = rng.permutation(len(X))
    for s in range(0, len(X), BATCH):
        b = idx[s:s + BATCH]
        yield X[b], y[b]


def accuracy(model, X, y):
    correct = 0
    for s in range(0, len(X), 1000):
        logits = model.forward(X[s:s + 1000])
        correct += int((logits.argmax(1) == y[s:s + 1000]).sum())
    return correct / len(X)


def gradient_train(optim_name, X_train, y_train, X_test, y_test):
    model = fresh_model()
    loss_fn = SoftmaxCrossEntropy()
    optim = SGD(model, lr=0.1) if optim_name == "SGD" else Adam(model, lr=1e-3)
    rng = np.random.default_rng(SEED)
    losses = []
    for _ in range(EPOCHS):
        running, n = 0.0, 0
        for xb, yb in minibatches(X_train, y_train, rng):
            logits = model.forward(xb)
            running += loss_fn.forward(logits, yb)
            model.backward(loss_fn.backward())
            optim.step()
            n += 1
        losses.append(running / n)
    return losses, accuracy(model, X_test, y_test)


def evolutionary_train(X_train, y_train, X_test, y_test):
    """Gradient-free baseline. YOU IMPLEMENT (week 4).

    A minimal (mu, lambda) / Gaussian-perturbation scheme:
      1. Keep a current best set of parameters (the MLP's W's and b's).
      2. Each generation, create lambda children by adding Gaussian noise
         (std = sigma) to the parent's parameters.
      3. Evaluate each child's loss on a batch (forward pass only — no gradients).
      4. Keep the best child if it beats the parent (optionally decay sigma).
    Return (loss_per_generation, final_test_accuracy) so the plot lines up with
    the gradient methods. Use EPOCHS as the generation count for comparability.
    """
    raise NotImplementedError("evolutionary_train (week 4)")


def main():
    (X_train, y_train), (X_test, y_test) = load_mnist()

    results = {}
    results["SGD"] = gradient_train("SGD", X_train, y_train, X_test, y_test)
    results["Adam"] = gradient_train("Adam", X_train, y_train, X_test, y_test)
    try:
        results["Evolutionary"] = evolutionary_train(X_train, y_train, X_test, y_test)
    except NotImplementedError:
        print("(evolutionary_train not implemented yet — plotting SGD vs Adam only)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for name, (losses, _) in results.items():
        ax1.plot(range(1, len(losses) + 1), losses, marker="o", label=name)
    ax1.set(xlabel="epoch / generation", ylabel="training loss", title="Convergence speed")
    ax1.legend()
    ax1.grid(alpha=0.3)

    names = list(results)
    accs = [results[n][1] for n in names]
    ax2.bar(names, accs)
    ax2.set(ylabel="test accuracy", title="Final accuracy", ylim=(0, 1))
    for i, a in enumerate(accs):
        ax2.text(i, a + 0.01, f"{a:.3f}", ha="center")

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "optimizer_compare.png"
    fig.savefig(out, dpi=120)
    print("saved", out)


if __name__ == "__main__":
    main()
