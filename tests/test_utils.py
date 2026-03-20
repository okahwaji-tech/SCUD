"""Tests for mathematical utility functions (scud.utils)."""
from __future__ import annotations

import pytest
import torch

from scud.utils import convert_to_probs, get_inf_gen, kls


class TestGetInfGen:
    """Tests for infinitesimal generator matrix construction."""

    def test_uniform_generator_rows_sum_to_zero(self) -> None:
        L = get_inf_gen({"type": "uniform"}, num_classes=4)
        row_sums = L.sum(dim=-1)
        assert torch.allclose(row_sums, torch.zeros(4), atol=1e-6)

    def test_gaussian_generator_rows_sum_to_zero(self) -> None:
        L = get_inf_gen({"type": "gaussian", "bandwidth": 0.05}, num_classes=4)
        row_sums = L.sum(dim=-1)
        assert torch.allclose(row_sums, torch.zeros(4), atol=1e-6)

    def test_uniform_generator_diagonal_is_minus_one(self) -> None:
        L = get_inf_gen({"type": "uniform"}, num_classes=4)
        diag = L.diagonal()
        assert torch.allclose(diag, -torch.ones(4), atol=1e-6)

    def test_uniform_generator_shape(self) -> None:
        L = get_inf_gen({"type": "uniform"}, num_classes=8)
        assert L.shape == (8, 8)

    def test_gaussian_generator_shape(self) -> None:
        L = get_inf_gen({"type": "gaussian", "bandwidth": 0.05}, num_classes=8)
        assert L.shape == (8, 8)


class TestKLS:
    """Tests for KL divergence computation."""

    def test_kls_zero_for_same_distribution(self) -> None:
        x = torch.randn(5, 4)
        kl = kls(x, x)
        assert torch.allclose(kl, torch.zeros(5), atol=1e-5)

    def test_kls_nonnegative(self) -> None:
        dist1 = torch.randn(10, 6)
        dist2 = torch.randn(10, 6)
        kl = kls(dist1, dist2)
        assert torch.all(kl >= -1e-6)


class TestConvertToProbs:
    """Tests for convert_to_probs utility."""

    def test_convert_to_probs_integer_input(self) -> None:
        x = torch.tensor([0, 1, 2, 3])
        probs = convert_to_probs(x, num_classes=4)
        expected = torch.eye(4)
        assert torch.allclose(probs.float(), expected)

    def test_convert_to_probs_float_input(self) -> None:
        x = torch.randn(3, 4)
        probs = convert_to_probs(x, num_classes=4)
        # Softmax outputs should sum to 1 along last dim
        row_sums = probs.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(3), atol=1e-5)
        # All probabilities should be non-negative
        assert torch.all(probs >= 0)
