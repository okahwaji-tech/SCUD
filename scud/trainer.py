"""PyTorch Lightning training infrastructure for discrete diffusion models.

Provides DiffusionTrainer, the Lightning base class that handles training
and validation loops, logging, sample generation (images/text/GIFs), and
optimizer configuration. Subclassed by ContinuousTimeDiffusion.

Reference: "Why Masking Diffusion Works" (NeurIPS 2025).
"""

from __future__ import annotations

import tempfile

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from lightning.pytorch.utilities import rank_zero_only
from PIL import Image
from torchvision.utils import make_grid
from tqdm import tqdm


def get_gif(
    sample_x: torch.Tensor,
    sample_a: torch.Tensor | None,
    model: DiffusionTrainer,
    gen_trans_step: int,
    batch_size: int,
) -> tuple[str | None, str | None]:
    # save images
    p = model.get_stationary()
    samples = torch.multinomial(
        p, num_samples=batch_size * sample_x.shape[1:].numel(), replacement=True
    )
    init_noise = samples.reshape((batch_size,) + sample_x.shape[1:]).to(sample_x.device)
    if sample_a is not None:
        attn_mask = sample_a.repeat(batch_size, *[1] * (sample_a.dim() - 1))
    else:
        attn_mask = None
    use_tau = getattr(model, "use_tau_at_inference", False)
    images = model.sample_sequence(
        init_noise,
        attn_mask,
        stride=3,
        n_T=gen_trans_step,
        use_tau=use_tau,
    )
    if images is not None:
        # image sequences to gif
        gif = []
        num_classes: int = model.num_classes  # type: ignore[assignment]
        for image in images:
            x_as_image = make_grid(image.float() / (num_classes - 1), nrow=2)
            img = x_as_image.permute(1, 2, 0).cpu().numpy()
            img = (img * 255).astype(np.uint8)
            gif.append(Image.fromarray(img))

        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as temp_file:
            gif[0].save(
                temp_file.name,
                format="GIF",
                save_all=True,
                append_images=gif[1:],
                duration=100,
                loop=0,
            )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file_img:
            last_img = gif[-1]
            last_img.save(temp_file_img)
        return temp_file.name, temp_file_img.name
    else:
        return None, None


def get_text(
    sample_x: torch.Tensor,
    sample_a: torch.Tensor | None,
    model: DiffusionTrainer,
    gen_trans_step: int,
    batch_size: int,
    tokenizer: object,
) -> tuple[list[str], list[list[str]]] | None:
    # save images
    p = model.get_stationary()
    samples = torch.multinomial(
        p, num_samples=batch_size * sample_x.shape[1:].numel(), replacement=True
    )
    init_noise = samples.reshape((batch_size,) + sample_x.shape[1:]).to(sample_x.device)
    if sample_a is not None:
        attn_mask = sample_a.repeat(batch_size, *[1] * (sample_a.dim() - 1))
    else:
        attn_mask = None
    use_tau = getattr(model, "use_tau_at_inference", False)
    tokens = model.sample_sequence(
        init_noise,
        attn_mask,
        stride=3,
        n_T=gen_trans_step,
        use_tau=use_tau,
    )
    if tokens is not None:
        last_token = tokens[-1]
        stride_tokens = tokens[:: (gen_trans_step // 3) // 10 + 1]
        if sample_a is not None:
            assert attn_mask is not None
            if hasattr(tokenizer, "pad_id"):
                pad_id = tokenizer.pad_id  # type: ignore[attr-defined]
            elif hasattr(tokenizer, "pad_token_id"):
                pad_id = tokenizer.pad_token_id  # type: ignore[attr-defined]
            last_token[attn_mask == 0.0] = pad_id
            for t in stride_tokens:
                t[attn_mask == 0.0] = pad_id
        if hasattr(tokenizer, "decode"):
            dt = lambda tok: [tokenizer.decode(t) for t in tok]  # type: ignore[union-attr]
        elif hasattr(tokenizer, "untokenize"):
            assert attn_mask is not None
            dt = lambda tok: [
                tokenizer.untokenize(t)[: int(a.sum())] for t, a in zip(tok, attn_mask, strict=True)  # type: ignore[union-attr]
            ]
        return dt(last_token), [dt(t) for t in stride_tokens]
    else:
        return None


class DiffusionTrainer(pl.LightningModule):
    """Base Lightning module for training discrete diffusion models.

    Handles the training/validation loop, optimizer configuration, gradient
    clipping, data distribution estimation, and sample generation for
    visualization (images as GIFs, text sequences).

    Args:
        lr: Learning rate.
        gen_trans_step: Number of denoising steps for sample generation.
        n_gen_images: Number of images/samples to generate for validation logging.
        grad_clip_val: Maximum gradient norm for clipping.
        weight_decay: AdamW weight decay.
        seed: Random seed.
        n_stat_samples: Number of samples used to estimate data distribution p0.
        tokenizer: Optional tokenizer for text/protein decoding during logging.
    """

    def __init__(
        self,
        lr: float = 1e-3,
        gen_trans_step: int = 1000,
        n_gen_images: int = 4,
        grad_clip_val: float = 1,
        weight_decay: float = 0,
        seed: int = 0,
        n_stat_samples: float = 2e6,
        tokenizer: object | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["tokenizer"])
        self.lr = lr
        self.grad_clip_val = grad_clip_val
        self.weight_decay = weight_decay
        # logging
        self.sample_x = None
        self.validation_step_outputs: list[dict[str, float]] = []
        self.gen_trans_step = gen_trans_step
        self.n_gen_images = n_gen_images
        self.n_stat_samples = n_stat_samples
        self.tokenizer = tokenizer

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the training loss. Must be overridden by subclasses."""
        raise NotImplementedError

    def get_stationary(self) -> torch.Tensor:
        """Return the stationary distribution. Must be overridden."""
        raise NotImplementedError

    def sample_sequence(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        n_T: int = 200,
        stride: int = 10,
        **kwargs: object,
    ) -> list[torch.Tensor]:
        """Generate samples. Must be overridden."""
        raise NotImplementedError

    def get_kl_t1(self, x: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence at t=1 (terminal time). Must be overridden."""
        raise NotImplementedError

    def pre_configure_model(self, dataloader: object) -> None:
        """Hook for model setup that requires data (e.g., schedule calibration)."""
        pass

    def calc_p0(self, dataloader: object) -> None:
        # get stationary dist
        num_classes: int = self.num_classes  # type: ignore[assignment]
        p0 = torch.ones(num_classes)
        pbar = tqdm(total=self.n_stat_samples)
        for _i, batch in tqdm(enumerate(dataloader)):  # type: ignore[arg-type]
            if p0.sum() > self.n_stat_samples:
                break
            if isinstance(batch, tuple):  # image datasets
                x, _ = batch
            elif isinstance(batch, dict):  # text datasets
                x = batch["input_ids"]
            new = (
                F.one_hot(x.long(), num_classes=num_classes)
                .to(torch.float32)
                .view((-1, num_classes))
                .sum(0)
            )
            p0 = p0 + new
            pbar.update(new.sum().item())
        pbar.close()
        p0 = p0 / p0.sum()
        self.p0 = p0

    def training_step(self, batch: object, batch_idx: int) -> torch.Tensor:
        if isinstance(batch, tuple):  # protein datasets
            x, attn_mask = batch
        elif isinstance(batch, dict):  # text datasets
            x, attn_mask = batch["input_ids"], batch["attention_mask"]
        else:  # image datasets
            x = batch
            attn_mask = None
        loss, info = self(x, attn_mask)
        if self.sample_x is None:
            self.sample_x = x[:1]
            self.sample_a = None if attn_mask is None else attn_mask[:1]

        self.log("train_loss", info["vb_loss"], sync_dist=True)
        self.log("train_ce_loss", info["ce_loss"], sync_dist=True)
        if "ce_loss_tcr" in info:
            self.log("train_ce_loss_tcr", info["ce_loss_tcr"], sync_dist=True)
        result: torch.Tensor = loss
        return result

    def validation_step(self, batch: object, batch_idx: int) -> dict[str, float]:
        if isinstance(batch, tuple):  # protein datasets
            x, attn_mask = batch
        elif isinstance(batch, dict):  # text datasets
            x, attn_mask = batch["input_ids"], batch["attention_mask"]
        else:  # image datasets
            x = batch
            attn_mask = None

        loss, info = self(x, attn_mask)
        self.log("val_l01", info["vb_loss"], on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "val_l1",
            self.get_kl_t1(x).detach().item(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log("val_ce_loss", info["ce_loss"], on_step=False, on_epoch=True, sync_dist=True)
        loss_dict = {
            "val_ce_loss": info["ce_loss"],
            "val_l01": info["vb_loss"],
            "val_l1": self.get_kl_t1(x).detach().item(),
        }
        return loss_dict

    @rank_zero_only
    def on_validation_epoch_end(
        self,
    ):
        # generate image
        if self.sample_x is not None:
            with torch.inference_mode():
                if self.tokenizer is None:
                    gif_fname, img_fname = get_gif(
                        self.sample_x, self.sample_a, self, self.gen_trans_step, self.n_gen_images
                    )
                    if gif_fname is not None and isinstance(self.logger, pl.loggers.WandbLogger):
                        self.logger.experiment.log({"sample_gif": wandb.Image(gif_fname)})
                        self.logger.experiment.log({"sample_gif_last": wandb.Image(img_fname)})
                else:
                    last_text, gen_text = get_text(
                        self.sample_x,
                        self.sample_a,
                        self,
                        self.gen_trans_step,
                        self.n_gen_images,
                        self.tokenizer,
                    )
                    if last_text is not None and isinstance(self.logger, pl.loggers.WandbLogger):
                        joined_text = "\n\n".join(last_text)
                        self.logger.experiment.log(
                            {"sample_text": wandb.Table(columns=["text"], data=[[joined_text]])}
                        )
                        joined_text_gen = ["\n\n".join(t) for t in gen_text]
                        self.logger.experiment.log(
                            {
                                "sample_text_process": wandb.Table(
                                    columns=["text"], data=[[jt] for jt in joined_text_gen]
                                )
                            }
                        )

    def configure_optimizers(self) -> dict[str, object]:  # type: ignore[override]
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return {
            "optimizer": optimizer,
        }
