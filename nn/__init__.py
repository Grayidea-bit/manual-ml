"""A tiny neural-network library, hand-written with NumPy only (no autograd).

Every differentiable piece implements the same two-method protocol so that the
chain rule shows up directly in the code structure:

    forward(x)            -> output, and caches whatever backward() needs
    backward(grad_output) -> grad_input (gradient w.r.t. this piece's input)

Layers that own parameters (Linear) additionally expose ``params_and_grads()``.
"""
