"""Smoke tests for the SCUD model with a tiny configuration."""

from __future__ import annotations

import pytest

pl = pytest.importorskip("lightning.pytorch")

import torch  # noqa: E402

from scud.scud import SCUD  # noqa: E402
from scud.unet import KingmaUNet  # noqa: E402


@pytest.fixture
def tiny_scud_model() -> SCUD:
    """Create a minimal SCUD model for smoke testing."""
    num_classes = 4
    nn_params = {
        "n_channel": 3,
        "N": num_classes,
        "n_T": 10,
        "schedule_conditioning": True,
        "s_dim": 8,
        "width": 8,
        "ch": 8,
        "s_lengthscale": 50,
        "time_lengthscale": 1,
        "n_layers": 1,
        "time_embed_dim": 0,
        "not_logistic_pars": True,
        "semb_style": "u_inject",
        "s_embed_dim": 16,
        "film": False,
        "input_logits": False,
        "first_mult": False,
    }
    model = SCUD(
        x0_model_class=KingmaUNet,
        nn_params=nn_params,
        num_classes=num_classes,
        forward_kwargs={"type": "uniform"},
        schedule_type="cos",
        gamma=0,
        logistic_pars=False,
    )
    # Set up p0 (uniform data distribution) and schedule functions
    model.p0 = torch.ones(num_classes) / num_classes
    model.log_alpha, model.beta = model.get_beta_func(
        model.K.cpu(), model.p0.cpu(), type_="schedule_condition", scale=model.rate.cpu()
    )
    model.eval()
    return model


class TestSCUDForward:
    """Smoke tests for SCUD model forward pass."""

    def test_forward_returns_loss_and_info(self, tiny_scud_model: SCUD) -> None:
        model = tiny_scud_model
        # Random integer data: batch=2, channels=3, height=8, width=8
        x = torch.randint(0, 4, (2, 3, 8, 8))
        with torch.no_grad():
            loss, info = model(x)
        assert isinstance(loss, torch.Tensor)
        assert isinstance(info, dict)

    def test_loss_is_scalar_and_not_nan(self, tiny_scud_model: SCUD) -> None:
        model = tiny_scud_model
        x = torch.randint(0, 4, (2, 3, 8, 8))
        with torch.no_grad():
            loss, _ = model(x)
        assert loss.dim() == 0, "Loss should be a scalar"
        assert not torch.isnan(loss), "Loss should not be NaN"

    def test_info_dict_has_expected_keys(self, tiny_scud_model: SCUD) -> None:
        model = tiny_scud_model
        x = torch.randint(0, 4, (2, 3, 8, 8))
        with torch.no_grad():
            _, info = model(x)
        assert "vb_loss" in info
        assert "ce_loss" in info
