"""Tests for the data preparation pipeline."""

from __future__ import annotations

import os

from omegaconf import OmegaConf

from scud.data.prepare import prepare_blosum, prepare_data


class TestPrepareBlosum:
    """Test BLOSUM62 matrix download."""

    def test_downloads_blosum_to_target_dir(self, tmp_path: object) -> None:
        """BLOSUM62 file should be downloaded when missing."""
        data_dir = str(tmp_path)
        prepare_blosum(data_dir)
        blosum_path = os.path.join(data_dir, "blosum62-special-MSA.mat")
        assert os.path.exists(blosum_path)
        with open(blosum_path) as f:
            content = f.read()
        assert "A" in content
        assert len(content) > 100

    def test_skips_if_already_exists(self, tmp_path: object) -> None:
        """Should not re-download if file already exists."""
        data_dir = str(tmp_path)
        blosum_path = os.path.join(data_dir, "blosum62-special-MSA.mat")
        os.makedirs(data_dir, exist_ok=True)
        with open(blosum_path, "w") as f:
            f.write("existing content")
        prepare_blosum(data_dir)
        with open(blosum_path) as f:
            assert f.read() == "existing content"


class TestPrepareData:
    """Test the main prepare_data dispatcher."""

    def test_cifar10_is_noop(self) -> None:
        """CIFAR-10 should not trigger any download (HF handles it)."""
        cfg = OmegaConf.create(
            {
                "data": {"data": "CIFAR10", "N": 128},
                "model": {"forward_kwargs": {"type": "gaussian"}},
            }
        )
        prepare_data(cfg)

    def test_mnist_is_noop(self) -> None:
        """MNIST should not trigger any download."""
        cfg = OmegaConf.create(
            {
                "data": {"data": "MNIST", "N": 10},
                "model": {"forward_kwargs": {"type": "uniform"}},
            }
        )
        prepare_data(cfg)

    def test_uniref50_with_blosum_downloads_matrix(self, tmp_path: object) -> None:
        """UniRef50 + BLOSUM should download the BLOSUM matrix."""
        cfg = OmegaConf.create(
            {
                "data": {"data": "uniref50", "data_dir": str(tmp_path), "N": 31},
                "model": {"forward_kwargs": {"type": "blosum"}},
            }
        )
        prepare_data(cfg)
        assert os.path.exists(os.path.join(str(tmp_path), "blosum62-special-MSA.mat"))

    def test_uniref50_without_blosum_is_noop(self) -> None:
        """UniRef50 with uniform forward process should not download BLOSUM."""
        cfg = OmegaConf.create(
            {
                "data": {"data": "uniref50", "N": 31},
                "model": {"forward_kwargs": {"type": "uniform"}},
            }
        )
        prepare_data(cfg)
