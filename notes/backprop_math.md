# Backpropagation — the derivations you'll implement

This is your reference while filling in the `backward()` TODOs. Work through it
with pen and paper at least once; the whole point of the project is that these
formulas stop being magic.

Notation: `N` = batch size, `C` = number of classes. `L` is the scalar loss.
For any quantity `q`, we write `dq` as shorthand for `∂L/∂q`. Gradients always
have the **same shape** as the thing they're the gradient of — use that as a
sanity check on every line.

---

## 1. Linear layer  `y = xW + b`

Shapes: `x:(N, in)`, `W:(in, out)`, `b:(out,)`, `y:(N, out)`.

Given `dy` (shape `(N, out)`) coming from the layer above:

```
dW = xᵀ · dy          # (in, N)·(N, out) = (in, out)   ✓ same shape as W
db = sum over batch of dy = dy.sum(axis=0)   # (out,)   ✓ same shape as b
dx = dy · Wᵀ          # (N, out)·(out, in) = (N, in)    ✓ same shape as x
```

Why `db` sums over the batch: `b` is broadcast (added to every row), so its
gradient is the sum of the contributions from all rows. Why the transposes:
they're forced — they're the only arrangement that makes the matrix shapes line
up. That's a reliable trick when you forget the exact form.

---

## 2. ReLU  `y = max(0, x)`

Elementwise, so the local derivative is a 0/1 mask:

```
dy/dx = 1 where x > 0, else 0
dx = dy * (x > 0)
```

The gradient only flows back through the units that were active on the forward
pass. Cache the mask `(x > 0)` during forward so backward is a cheap multiply.

---

## 3. Softmax + cross-entropy (fused)  — the elegant one

Softmax: `p_i = exp(z_i) / Σ_j exp(z_j)`  (per row; subtract `max(z)` first for
numerical stability — it cancels out mathematically).

Cross-entropy for one example with true class `t`:  `L = -log(p_t)`.

**Claim:** `∂L/∂z_i = p_i - 1[i == t]`.

Sketch (single example): split into the `i = t` term and `i ≠ t` terms.
- `∂L/∂z_i = -∂ log p_t / ∂z_i = -(1/p_t) · ∂p_t/∂z_i`.
- Softmax Jacobian: `∂p_t/∂z_i = p_t(1[i==t] - p_i)`.
- Substitute: `∂L/∂z_i = -(1/p_t) · p_t(1[i==t] - p_i) = p_i - 1[i==t]`.

All the `p_t` factors cancel — that cancellation is exactly why we fuse softmax
and cross-entropy instead of backpropagating through them separately.

Averaged over a batch of `N` (because forward takes `mean`):

```
dlogits = (probs - onehot(y)) / N        # (N, C)
```

`onehot(y)` is the `(N, C)` matrix with a 1 in column `y[i]` of row `i`.

---

## 4. How the layers chain (what `MLP.backward` does)

Forward:  `x → Linear1 → ReLU → Linear2 → ReLU → Linear3 → logits → loss`.

Backward walks it in reverse, each layer multiplying by its local derivative:

```
dlogits            = loss.backward()
d(Linear3 input)   = Linear3.backward(dlogits)
d(ReLU input)      = ReLU.backward(...)
...                back to the first layer
```

Each `backward()` receives `∂L/∂(its output)` and returns `∂L/∂(its input)` —
which is precisely `∂L/∂(previous layer's output)`. That hand-off *is* the chain
rule. Nothing else is going on.

---

## 5. Optimizer updates (weeks 2–3)

**SGD:**  `p ← p − lr · dp`.

**Adam:** keep running averages of the gradient (`m`) and its square (`v`):

```
t ← t + 1
m ← β1·m + (1−β1)·g
v ← β2·v + (1−β2)·g²
m̂ ← m / (1 − β1ᵗ)          # bias correction (m, v start at 0, so early steps are biased low)
v̂ ← v / (1 − β2ᵗ)
p ← p − lr · m̂ / (√v̂ + ε)
```

Intuition: `m` is a smoothed gradient (momentum); dividing by `√v̂` scales each
parameter's step by how noisy/large its gradient has been — big, erratic
gradients get smaller steps. That per-parameter adaptivity is why Adam usually
converges faster than plain SGD, which you'll confirm in week 3.
```
