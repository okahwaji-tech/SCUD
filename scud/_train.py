"""Internal training logic extracted from train.py."""

import glob
import logging
import os

import certifi
import lightning.pytorch as pl
import torch
import wandb
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf

from scud.classical_diffusion import ClassicalDiffusion
from scud.data import get_dataloaders
from scud.ema import EMA
from scud.masking_diffusion import MaskingDiffusion
from scud.nets import get_model_setup
from scud.scud import SCUD
from scud.scud_tcr import SCUD_TCR

os.environ["SSL_CERT_FILE"] = certifi.where()

logger = logging.getLogger(__name__)


def run_training(cfg: DictConfig) -> None:
    """Run the full training pipeline for SCUD/Masking/Classical diffusion models."""
    # Enable CPU fallback for MPS-unsupported ops (torch.poisson, etc.)
    if not torch.cuda.is_available() and torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    @rank_zero_only
    def init_wandb() -> None:
        wandb.login()

    init_wandb()
    ##### Prepare data (auto-download if missing)
    from scud.data.prepare import prepare_data

    prepare_data(cfg)

    ##### Load data
    pl.seed_everything(cfg.model.seed, workers=True)
    logger.info("Getting dataloaders.")
    train_dataloader, test_dataloader = get_dataloaders(cfg)
    tokenizer = getattr(train_dataloader, "tokenizer", None)

    ##### Setup x0_model
    logger.info("Setting up model.")
    x0_model_class, nn_params = get_model_setup(cfg, tokenizer)

    logger.info("Config: %s", cfg)

    ##### Pick model
    model_name_dict = {
        "SCUD": SCUD,
        "SCUD_TCR": SCUD_TCR,
        "Masking": MaskingDiffusion,
        "Classical": ClassicalDiffusion,
    }
    if not cfg.model.restart:
        model = model_name_dict[cfg.model.model](
            x0_model_class,
            nn_params,
            num_classes=len(tokenizer) if tokenizer else cfg.data.N,
            gamma=cfg.model.gamma,
            forward_kwargs=OmegaConf.to_container(cfg.model.forward_kwargs, resolve=True),
            schedule_type=cfg.model.schedule_type,
            logistic_pars=cfg.model.logistic_pars,
            gen_trans_step=cfg.sampling.gen_trans_step,
            t_max=cfg.model.t_max,
            seed=cfg.model.seed,
            tokenizer=tokenizer if cfg.data.data != "uniref50" else __import__("evodiff.utils", fromlist=["Tokenizer"]).Tokenizer(),
            **OmegaConf.to_container(cfg.train, resolve=True),  # type: ignore[arg-type]
        )
        ckpt_path = None
    else:
        ckpt_path = f"checkpoints/{cfg.model.restart}"
        ckpt_path = max(glob.glob(os.path.join(ckpt_path, "*.ckpt")), key=os.path.getmtime)
        model = model_name_dict[cfg.model.model].load_from_checkpoint(ckpt_path)  # type: ignore[attr-defined]

    ##### Load data
    model.pre_configure_model(train_dataloader)

    ##### Train
    wandb_logger = WandbLogger(project=OmegaConf.select(cfg, "wandb.project", default="scud"))
    lightning_model = model
    torch.set_float32_matmul_precision("high")

    @rank_zero_only
    def update_wandb_config():
        if wandb.run is not None:
            wandb.config.update(lightning_model.hparams)

    update_wandb_config()

    if cfg.data.data == "uniref50":
        val_check_interval = 2 * (210000 // cfg.train.batch_size)
    else:
        val_check_interval = 1.0
    trainer = Trainer(
        max_epochs=cfg.train.n_epoch,
        accelerator=cfg.train.get("accelerator", "auto"),
        devices=cfg.train.get("devices", "auto"),
        logger=wandb_logger,
        strategy=DDPStrategy(broadcast_buffers=True) if torch.cuda.is_available() else "auto",
        callbacks=(
            [EMA(0.9999)] * cfg.train.ema
            + [
                ModelCheckpoint(
                    dirpath=f"checkpoints/{wandb_logger.experiment.name}",
                    save_on_train_epoch_end=False,
                )
            ]
        ),
        val_check_interval=val_check_interval,
        accumulate_grad_batches=cfg.train.accumulate,
        precision=cfg.train.get("precision", "32-true"),
        gradient_clip_val=cfg.train.grad_clip_val,
        gradient_clip_algorithm="norm",
    )
    trainer.fit(lightning_model, train_dataloader, test_dataloader, ckpt_path=ckpt_path)
    wandb.finish()
