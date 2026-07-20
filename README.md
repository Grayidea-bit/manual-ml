# manual-ml — A neural network and optimizers, hand-written from scratch

A from-scratch implementation of a multi-layer perceptron and its optimizers in
**pure Python + NumPy**, with **no autograd** (no PyTorch/TensorFlow). Every
forward pass, every gradient in backpropagation, and every optimizer update rule
is derived and coded by hand.

The goal is not to call an ML API — it is to *understand and demonstrate* the
mechanics underneath: how backpropagation applies the chain rule layer by layer,
and how gradient-based and gradient-free optimizers actually update parameters.

## What's implemented

- **A modular NN library** where every differentiable piece exposes the same
  `forward` / `backward` protocol, so the chain rule is visible in the code
  structure — Linear layers, ReLU, and a fused softmax + cross-entropy loss.
- **Backpropagation by hand**, verified against numerical gradients
  (central-difference gradient checking, relative error < 1e-6).
- **Optimizers from their update equations**: SGD, Adam (with bias correction),
  and a simple evolutionary (gradient-free) algorithm.
- **An optimizer comparison** on MNIST: convergence speed and final accuracy of
  SGD vs Adam vs evolutionary search, in one figure.

Trained on **MNIST**, whose raw IDX binary files are downloaded from a mirror and
decoded by hand (no dataset-loader library).

## Project layout

```
manual-ml/
├── data/download_mnist.py       # fetch + decode raw MNIST IDX files
├── nn/
│   ├── layers.py                # Linear layer  (forward/backward)
│   ├── activations.py           # ReLU, softmax
│   ├── losses.py                # fused softmax + cross-entropy
│   ├── network.py               # MLP: chains layers, runs forward/backward
│   ├── optimizers.py            # SGD, Adam
│   └── grad_check.py            # numerical gradient checking
├── train_mnist.py               # training loop -> test accuracy
├── experiments/optimizer_compare.py   # SGD vs Adam vs evolutionary + plot
└── notes/backprop_math.md       # derivations behind every backward()
```

## Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate      |  macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
python data/download_mnist.py           # download + cache MNIST
```

## Run

```bash
python -m nn.grad_check                  # verify backprop against numerical gradients
python train_mnist.py                    # train on MNIST  (target: >=95% test accuracy)
python experiments/optimizer_compare.py  # produce experiments/optimizer_compare.png
```

## Learning roadmap

| Date / Period | Topic | Status |
|------|-------|--------|
| 26-07-15 | Forward pass: Linear, ReLU, softmax + cross-entropy | DONE |
| 26-07-20 | Backpropagation + SGD; gradient checking; train to convergence | DONE |
|  | Adam from the update equations; SGD vs Adam | — |
|  | Evolutionary optimizer; three-way comparison + figure | — |

Part of a broader self-study track; a separate repository will cover the NLP
project (weeks 5–8).

## Background reading

Derivations for the backward passes are in
[`notes/backprop_math.md`](notes/backprop_math.md).
