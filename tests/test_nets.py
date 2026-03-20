"""Tests for neural network model registry and configuration (scud.nets)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("omegaconf")

sys.path.insert(0, str(Path(__file__).parent))
from conftest import make_config
from scud.nets import get_model_setup


class TestScheduleConditioning:
    """Verify schedule_conditioning flag is set correctly for each model type."""

    def test_scud_enables_schedule_conditioning(self) -> None:
        cfg = make_config(model="SCUD")
        _, nn_params = get_model_setup(cfg)
        assert nn_params["schedule_conditioning"] is True

    def test_masking_enables_schedule_conditioning(self) -> None:
        cfg = make_config(model="MaskingDiffusion")
        _, nn_params = get_model_setup(cfg)
        assert nn_params["schedule_conditioning"] is True

    def test_classical_disables_schedule_conditioning(self) -> None:
        cfg = make_config(model="Classical")
        _, nn_params = get_model_setup(cfg)
        assert nn_params["schedule_conditioning"] is False


class TestImageModelParams:
    """Verify nn_params construction for image models."""

    def test_image_model_has_correct_params(self) -> None:
        cfg = make_config(model="SCUD", data="CIFAR10", N=4)
        nn_class, nn_params = get_model_setup(cfg)

        expected_keys = {
            "n_channel",
            "N",
            "n_T",
            "schedule_conditioning",
            "s_dim",
            "s_lengthscale",
            "time_lengthscale",
            "n_layers",
            "time_embed_dim",
            "not_logistic_pars",
            "semb_style",
            "s_embed_dim",
            "film",
            "input_logits",
            "first_mult",
        }
        assert expected_keys.issubset(set(nn_params.keys()))
        assert nn_params["n_channel"] == 3  # CIFAR10 has 3 channels
        assert nn_params["N"] == 4  # num_classes for non-masking model

    def test_mnist_gets_one_channel(self) -> None:
        cfg = make_config(model="SCUD", data="MNIST", N=4)
        _, nn_params = get_model_setup(cfg)
        assert nn_params["n_channel"] == 1

    def test_masking_model_adds_one_to_N(self) -> None:
        cfg = make_config(model="MaskingDiffusion", N=4)
        _, nn_params = get_model_setup(cfg)
        assert nn_params["N"] == 5  # N + 1 for mask token
