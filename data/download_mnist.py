"""Download MNIST and decode the raw IDX binary format by hand.

We deliberately avoid framework dataset loaders (keras/sklearn/torchvision).
We fetch the original ``.gz`` files from a stable mirror and decode the IDX
format ourselves, because understanding how raw bytes become an ``(N, 784)``
float array is part of learning the fundamentals.

Run directly to download + cache::

    python data/download_mnist.py

Use elsewhere::

    from data.download_mnist import load_mnist
    (X_train, y_train), (X_test, y_test) = load_mnist()

IDX format reference (http://yann.lecun.com/exdb/mnist/):
    magic number (4 bytes) | dim sizes (4 bytes each) | data (row-major)
    magic = 0x00 0x00 <dtype> <ndim>   e.g. images -> 0x00000803, labels -> 0x00000801
    dtype 0x08 == unsigned byte (uint8), which is all MNIST uses.
"""
from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np

MIRROR = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"
CACHE = DATA_DIR / "mnist.npz"


def _download(filename: str) -> Path:
    RAW_DIR.mkdir(exist_ok=True)
    dest = RAW_DIR / filename
    if not dest.exists():
        url = MIRROR + filename
        print(f"downloading {url} ...")
        urllib.request.urlretrieve(url, dest)
    return dest


def _read_idx(path: Path) -> np.ndarray:
    """Decode a gzip-compressed IDX file into a numpy uint8 array."""
    with gzip.open(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        ndim = magic & 0xFF                       # low byte = number of dimensions
        shape = tuple(struct.unpack(">I", f.read(4))[0] for _ in range(ndim))
        buf = f.read()
    return np.frombuffer(buf, dtype=np.uint8).reshape(shape)


def load_mnist(normalize: bool = True, flatten: bool = True):
    """Return ``((X_train, y_train), (X_test, y_test))``.

    normalize: scale pixels from [0, 255] uint8 to [0, 1] float32.
    flatten:   reshape 28x28 images to length-784 vectors.
    """
    if CACHE.exists():
        d = np.load(CACHE)
        X_train, y_train = d["X_train"], d["y_train"]
        X_test, y_test = d["X_test"], d["y_test"]
    else:
        p = {k: _download(v) for k, v in FILES.items()}
        X_train = _read_idx(p["train_images"])
        y_train = _read_idx(p["train_labels"])
        X_test = _read_idx(p["test_images"])
        y_test = _read_idx(p["test_labels"])
        np.savez_compressed(
            CACHE, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        )

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    if normalize:
        X_train /= 255.0
        X_test /= 255.0
    if flatten:
        X_train = X_train.reshape(len(X_train), -1)
        X_test = X_test.reshape(len(X_test), -1)
    return (X_train, y_train.astype(np.int64)), (X_test, y_test.astype(np.int64))


if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = load_mnist()
    print("train images:", X_train.shape, X_train.dtype)
    print("train labels:", y_train.shape, y_train.dtype, "classes:", np.unique(y_train))
    print("test  images:", X_test.shape)
    print("pixel range :", float(X_train.min()), "->", float(X_train.max()))
