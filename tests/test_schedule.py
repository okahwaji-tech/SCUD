"""Tests for schedule sampling utilities (scud.schedule_sample)."""
from __future__ import annotations

import torch

from scud.schedule_sample import sample_n_transitions_cont


def _dummy_log_alpha(t: torch.Tensor) -> torch.Tensor:
    """A simple log_alpha function: log(cos(pi*t/2)) scaled."""
    return torch.log(torch.cos(t * torch.pi / 2).clamp(min=1e-9))


class TestSampleNTransitionsCont:
    """Tests for continuous-time Poisson transition sampling."""

    def test_poisson_sampling_nonnegative(self) -> None:
        times = torch.tensor([0.1, 0.5, 0.9])
        result = sample_n_transitions_cont(_dummy_log_alpha, batch_size=16, times=times)
        assert torch.all(result >= 0)

    def test_poisson_sampling_shape(self) -> None:
        times = torch.tensor([0.1, 0.3, 0.5, 0.7])
        batch_size = 8
        result = sample_n_transitions_cont(_dummy_log_alpha, batch_size=batch_size, times=times)
        assert result.shape == (batch_size, len(times))

    def test_poisson_sampling_integer_valued(self) -> None:
        times = torch.tensor([0.2, 0.6])
        result = sample_n_transitions_cont(_dummy_log_alpha, batch_size=4, times=times)
        assert torch.allclose(result, result.round())

    def test_higher_time_gives_more_transitions_on_average(self) -> None:
        """At higher t, -log_alpha(t) is larger, so Poisson mean is larger."""
        times_low = torch.tensor([0.1])
        times_high = torch.tensor([0.9])
        n_samples = 1000
        low = sample_n_transitions_cont(_dummy_log_alpha, batch_size=n_samples, times=times_low)
        high = sample_n_transitions_cont(_dummy_log_alpha, batch_size=n_samples, times=times_high)
        assert high.float().mean() > low.float().mean()
