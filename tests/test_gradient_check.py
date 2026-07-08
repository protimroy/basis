# -*- coding: utf-8 -*-
"""
Gradient check for the from-scratch neural network library.

The library implements backpropagation manually (it only uses PyTorch for the
tensor operations themselves, not for autograd). This test verifies that the
hand-derived gradients match the gradients that PyTorch's autograd computes for
the exact same loss, which is the ground truth.

Run from the repository root with either:

    python -m pytest tests/test_gradient_check.py
    python tests/test_gradient_check.py
"""

import os
import sys

import torch

# Allow running directly (python tests/test_gradient_check.py) from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.layers.input import InputLayer
from src.layers.dense import DenseLayer
from src.models.neural_network import NeuralNetwork
from src.losses.cross_entropy import Loss
from src.optimizers.gradient_descent import GD


def _build_network(hyperparams, device="cpu"):
    input_layer = InputLayer(hyperparams, name="input", device=device)
    hidden_layer = DenseLayer(hyperparams, name="hidden", transfer="sigmoid", device=device)
    output_layer = DenseLayer(hyperparams, name="output", transfer="sigmoid", device=device)
    network = NeuralNetwork(input_layer, hidden_layer, output_layer, device=device)
    network.compile(loss=Loss(name="binary_cross_entropy"), optimizer_class=GD, learning_rate=0.5)
    return network


def _run_check(hyperparams, num_samples, device="cpu", tol=1e-5):
    torch.manual_seed(0)
    network = _build_network(hyperparams, device=device)

    # Column-major layout: units along dim 0, samples along dim 1.
    X = torch.rand(hyperparams["input_units"], num_samples, dtype=torch.float32, device=device)
    Y = (torch.rand(hyperparams["output_units"], num_samples, device=device) > 0.5).float()

    params = list(network.trainable_parameters.values())

    # --- Autograd reference gradients ---
    network.clear_all_gradients()
    output = network.forward(X)["output"]
    loss = network.loss_fn(output, Y)
    autograd_grads = torch.autograd.grad(loss, params, retain_graph=False)

    # --- Manual gradients from the library's own backward pass ---
    network.clear_all_gradients()
    output = network.forward(X)["output"]
    _ = network.loss_fn(output, Y)
    grad_loss_wrt_output = network.loss_fn.backward()
    network.backward(grad_loss_wrt_output)

    max_error = 0.0
    for param, ref_grad in zip(params, autograd_grads):
        manual_grad = param.grad
        assert manual_grad is not None, "Manual backward did not populate a gradient."
        assert manual_grad.shape == ref_grad.shape, (
            f"Shape mismatch: manual {tuple(manual_grad.shape)} vs autograd {tuple(ref_grad.shape)}"
        )
        rel_error = (manual_grad - ref_grad).abs().max().item()
        max_error = max(max_error, rel_error)
        assert torch.allclose(manual_grad, ref_grad, atol=tol, rtol=tol), (
            f"Gradient mismatch (max abs diff {rel_error:.3e}) exceeds tolerance {tol:.1e}."
        )
    return max_error


def test_gradient_check_single_sample():
    hyperparams = {"input_units": 2, "hidden_units": 2, "output_units": 1}
    _run_check(hyperparams, num_samples=1)


def test_gradient_check_batch():
    hyperparams = {"input_units": 2, "hidden_units": 2, "output_units": 1}
    _run_check(hyperparams, num_samples=4)


def test_gradient_check_wider_network():
    hyperparams = {"input_units": 5, "hidden_units": 8, "output_units": 3}
    _run_check(hyperparams, num_samples=10)


if __name__ == "__main__":
    for name, fn in [
        ("single sample", test_gradient_check_single_sample),
        ("batch of 4", test_gradient_check_batch),
        ("wider network", test_gradient_check_wider_network),
    ]:
        fn()
        print(f"[PASS] gradient check ({name})")
    print("All gradient checks passed.")
