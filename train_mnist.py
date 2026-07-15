"""Train the hand-written MLP on MNIST. (Provided scaffolding.)

Once the forward/backward/optimizer TODOs are implemented, run::

    python train_mnist.py

Expected: test accuracy >= 95% within a few epochs.
"""
from __future__ import annotations

import time

import numpy as np

from data.download_mnist import load_mnist
from nn.losses import SoftmaxCrossEntropy
from nn.network import MLP
from nn.optimizers import SGD


def iterate_minibatches(X, y, batch_size, rng):
    idx = rng.permutation(len(X))
    for start in range(0, len(X), batch_size):
        batch = idx[start:start + batch_size]
        yield X[batch], y[batch]


def accuracy(model, X, y, batch_size=1000):
    correct = 0
    for start in range(0, len(X), batch_size):
        logits = model.forward(X[start:start + batch_size])
        correct += int((logits.argmax(axis=1) == y[start:start + batch_size]).sum())
    return correct / len(X)


def main():
    rng = np.random.default_rng(0)
    (X_train, y_train), (X_test, y_test) = load_mnist()

    model = MLP([784, 128, 64, 10], rng=rng)
    loss_fn = SoftmaxCrossEntropy()
    optim = SGD(model, lr=0.1)

    epochs = 10
    batch_size = 128
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        running = 0.0
        n_batches = 0
        for xb, yb in iterate_minibatches(X_train, y_train, batch_size, rng):
            logits = model.forward(xb)
            loss = loss_fn.forward(logits, yb)
            model.backward(loss_fn.backward())
            optim.step()
            running += loss
            n_batches += 1
        train_loss = running / n_batches
        test_acc = accuracy(model, X_test, y_test)
        print(f"epoch {epoch:2d}  loss {train_loss:.4f}  test_acc {test_acc:.4f}  "
              f"({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
