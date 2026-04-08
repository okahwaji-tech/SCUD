"""Tests for TCR utilities and SCUD_TCR model."""

from __future__ import annotations

import pytest

pl = pytest.importorskip("lightning.pytorch")

import torch  # noqa: E402

from scud.scud_tcr import SCUD_TCR  # noqa: E402
from scud.tcr import detached_predict, sample_k, scud_backward_transition  # noqa: E402
from scud.unet import KingmaUNet  # noqa: E402


@pytest.fixture
def tiny_scud_tcr_model() -> SCUD_TCR:
    """Create a minimal SCUD_TCR model for smoke testing."""
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
    model = SCUD_TCR(
        x0_model_class=KingmaUNet,
        nn_params=nn_params,
        num_classes=num_classes,
        forward_kwargs={"type": "uniform"},
        schedule_type="cos",
        gamma=0,
        logistic_pars=False,
    )
    model.p0 = torch.ones(num_classes) / num_classes
    model.log_alpha, model.beta = model.get_beta_func(
        model.K.cpu(), model.p0.cpu(), type_="schedule_condition", scale=model.rate.cpu()
    )
    model.eval()
    return model


class TestTCRUtilities:
    """Tests for TCR utility functions in scud/tcr.py."""

    def test_sample_k_range(self) -> None:
        S = torch.tensor([[3, 0, 5, 1]])
        s_low, k = sample_k(S)
        # k is 0 where S is 0
        assert (k[S == 0] == 0).all()
        # k >= 1 where S > 0
        assert (k[S > 0] >= 1).all()
        # k <= S everywhere
        assert (k <= S).all()
        # s_low = S - k
        assert (s_low == S - k).all()

    def test_detached_predict_shape_and_detach(self, tiny_scud_tcr_model: SCUD_TCR) -> None:
        model = tiny_scud_tcr_model
        x = torch.randint(0, 4, (2, 3, 8, 8))
        t, S, x_t = model.sample_point(x)
        x0_hat, x0_logits = detached_predict(model, x_t, t, None, S)
        assert x0_hat.shape == x_t.shape
        assert x0_logits.shape == (*x_t.shape, model.num_classes)
        assert not x0_logits.requires_grad

    def test_backward_transition_shape(self, tiny_scud_tcr_model: SCUD_TCR) -> None:
        model = tiny_scud_tcr_model
        x = torch.randint(0, 4, (2, 3, 8, 8))
        t, S, x_t = model.sample_point(x)
        x0_hat, _ = detached_predict(model, x_t, t, None, S)
        s_low, k = sample_k(S)
        x_prev = scud_backward_transition(
            x_t,
            x0_hat,
            model.K_powers,
            S,
            k,
            model.num_classes,
            model.eigenvectors,
            model.eigenvalues,
            model.eigenvectors_inv,
        )
        assert x_prev.shape == x_t.shape


class TestTauEmbedding:
    """Tests for tau embedding in KingmaUNet."""

    def test_unet_tau_none_matches_tau_zero_at_init(self) -> None:
        unet = KingmaUNet(
            n_channel=3,
            N=4,
            n_T=10,
            schedule_conditioning=True,
            s_dim=8,
            width=8,
            ch=8,
            s_lengthscale=50,
            time_lengthscale=1,
            n_layers=1,
            time_embed_dim=0,
            not_logistic_pars=True,
            semb_style="u_inject",
            s_embed_dim=16,
            film=False,
            input_logits=False,
            first_mult=False,
        )
        unet.eval()
        x = torch.randint(0, 4, (1, 3, 8, 8))
        t = torch.tensor([0.5])
        S = torch.ones(1, 3, 8, 8, dtype=torch.long)
        with torch.no_grad():
            out_none = unet(x, t, S=S, tau=None)
            out_zero = unet(x, t, S=S, tau=torch.zeros_like(S, dtype=torch.float32))
        assert torch.allclose(out_none, out_zero, atol=1e-5)


class TestSCUDTCRForward:
    """Smoke tests for SCUD_TCR model forward pass."""

    def test_forward_returns_loss_and_info(self, tiny_scud_tcr_model: SCUD_TCR) -> None:
        model = tiny_scud_tcr_model
        x = torch.randint(0, 4, (2, 3, 8, 8))
        with torch.no_grad():
            loss, info = model(x)
        assert isinstance(loss, torch.Tensor)
        assert isinstance(info, dict)

    def test_loss_is_scalar_and_not_nan(self, tiny_scud_tcr_model: SCUD_TCR) -> None:
        model = tiny_scud_tcr_model
        x = torch.randint(0, 4, (2, 3, 8, 8))
        with torch.no_grad():
            loss, _ = model(x)
        assert loss.dim() == 0, "Loss should be a scalar"
        assert not torch.isnan(loss), "Loss should not be NaN"

    def test_info_dict_has_expected_keys(self, tiny_scud_tcr_model: SCUD_TCR) -> None:
        model = tiny_scud_tcr_model
        x = torch.randint(0, 4, (2, 3, 8, 8))
        with torch.no_grad():
            _, info = model(x)
        assert "vb_loss" in info
        assert "ce_loss" in info
        assert "ce_loss_tcr" in info
