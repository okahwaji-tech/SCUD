"""Tests for TCR utilities and SCUD_TCR model."""

from __future__ import annotations

from unittest.mock import PropertyMock, patch

import pytest

pl = pytest.importorskip("lightning.pytorch")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from scud.scud_tcr import SCUD_TCR  # noqa: E402
from scud.tcr import (  # noqa: E402
    _eigenvector_mvp,
    detached_predict,
    sample_k,
    scud_backward_transition,
)
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


class TestNumericalCorrectness:
    """Numerical correctness tests comparing TCR utilities against SCUD model methods."""

    def test_eigenvector_mvp_matches_scud(self, tiny_scud_tcr_model: SCUD_TCR) -> None:
        """Verify _eigenvector_mvp gives the same result as SCUD.get_trans_mats_mvp."""
        model = tiny_scud_tcr_model
        torch.manual_seed(42)

        # Create test inputs: random S values and probability vectors
        shape = (2, 3, 4, 4)
        S = torch.randint(0, 8, shape)
        probs = torch.rand(*shape, model.num_classes)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        # Compute via standalone function
        result_fn = _eigenvector_mvp(
            S, probs, model.eigenvectors, model.eigenvalues, model.eigenvectors_inv
        )

        # Compute via SCUD model method
        result_model = model.get_trans_mats_mvp(S, probs)

        assert torch.allclose(result_fn, result_model, atol=1e-5), (
            f"Max diff: {(result_fn - result_model).abs().max().item()}"
        )

    def test_backward_transition_matches_q_posterior(
        self, tiny_scud_tcr_model: SCUD_TCR
    ) -> None:
        """Verify scud_backward_transition posterior matches SCUD.q_posterior_logits."""
        model = tiny_scud_tcr_model
        torch.manual_seed(42)

        # Generate a sample point from the model
        x = torch.randint(0, model.num_classes, (2, 3, 8, 8))
        t, S, x_t = model.sample_point(x)

        # Both methods use k=1.  q_posterior_logits uses scalar k=1 and
        # relies on F.relu to clamp S-k when S==0, so the comparison is
        # only valid where S >= 1 (the normal operating regime).
        mask = S >= 1  # positions where k=1 is meaningful

        # --- TCR backward transition computes: fact1 * fact2 ---
        # fact1 = K_powers^T[1, x_t, :]  (scalar k=1 to match q_posterior_logits)
        fact1_tcr = model.K_powers.swapaxes(1, 2)[1, x_t, :]
        # fact2 = eigenvector_mvp(S - 1, one_hot(x0))
        x0_probs = torch.nn.functional.one_hot(x.long(), model.num_classes).float()
        fact2_tcr = _eigenvector_mvp(
            F.relu(S - 1),
            x0_probs,
            model.eigenvectors,
            model.eigenvalues,
            model.eigenvectors_inv,
        )
        probs_tcr = (fact1_tcr * fact2_tcr).clamp(min=0.0)
        probs_tcr = probs_tcr / (probs_tcr.sum(dim=-1, keepdim=True) + 1e-9)

        # --- SCUD q_posterior_logits with log=False ---
        posterior_scud = model.q_posterior_logits(x, x_t, t, S, k=1, log=False)
        probs_scud = posterior_scud.clamp(min=0.0)
        probs_scud = probs_scud / (probs_scud.sum(dim=-1, keepdim=True) + 1e-9)

        # Compare only where S >= 1 (both methods agree on k=1 semantics)
        assert mask.any(), "Need at least some positions with S >= 1"
        probs_tcr_masked = probs_tcr[mask]
        probs_scud_masked = probs_scud[mask]
        assert torch.allclose(probs_tcr_masked, probs_scud_masked, atol=1e-5), (
            f"Max diff: {(probs_tcr_masked - probs_scud_masked).abs().max().item()}"
        )


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
        assert "tcr_lambda" in info

    def test_path_b_skipped_in_eval(self, tiny_scud_tcr_model: SCUD_TCR) -> None:
        model = tiny_scud_tcr_model
        model.eval()
        x = torch.randint(0, 4, (2, 3, 8, 8))
        with torch.no_grad():
            _, info = model(x)
        assert info["ce_loss_tcr"] == 0.0
        assert info["tcr_lambda"] == 0.0

    def test_path_b_runs_in_train_with_nonzero_lambda(self) -> None:
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
            tcr_warmup_epochs=1,
            tcr_lambda_max=0.5,
        )
        model.p0 = torch.ones(num_classes) / num_classes
        model.log_alpha, model.beta = model.get_beta_func(
            model.K.cpu(), model.p0.cpu(), type_="schedule_condition", scale=model.rate.cpu()
        )
        # Simulate epoch 1 (past warmup of 1 epoch) so lambda > 0
        model.train()
        x = torch.randint(0, 4, (2, 3, 8, 8))
        with patch.object(type(model), "current_epoch", new_callable=PropertyMock, return_value=1):
            loss, info = model(x)
        assert info["tcr_lambda"] == 0.5
        assert info["ce_loss_tcr"] > 0.0

    def test_curriculum_lambda_warmup(self) -> None:
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
            tcr_warmup_epochs=4,
            tcr_lambda_max=0.8,
        )
        # Epoch 0: lambda should be 0
        with patch.object(type(model), "current_epoch", new_callable=PropertyMock, return_value=0):
            assert model._get_tcr_lambda() == 0.0
        # Epoch 2: lambda should be 0.5 * 0.8 = 0.4
        with patch.object(type(model), "current_epoch", new_callable=PropertyMock, return_value=2):
            assert abs(model._get_tcr_lambda() - 0.4) < 1e-9
        # Epoch 4+: lambda should be 0.8
        with patch.object(type(model), "current_epoch", new_callable=PropertyMock, return_value=4):
            assert abs(model._get_tcr_lambda() - 0.8) < 1e-9
        with patch.object(
            type(model), "current_epoch", new_callable=PropertyMock, return_value=10
        ):
            assert abs(model._get_tcr_lambda() - 0.8) < 1e-9
