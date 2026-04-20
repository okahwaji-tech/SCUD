import sys
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only
import pytorch_lightning as pl
from scud.scud_sm_pretrain import SM_SCUD_PT

from evodiff.utils import Tokenizer

from scud.scud import SCUD
from scud.sm_scud_tcr import SM_SCUD_TCR
from scud.scud_sm_pretrain import SM_SCUD_PT
from scud.masking_diffusion import MaskingDiffusion
from scud.classical_diffusion import ClassicalDiffusion

from nets import get_model_setup
from data import get_dataloaders
from ema import EMA

import getpass

import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()

@hydra.main(version_base=None, config_path="configs", config_name="basic")
def train(cfg: DictConfig) -> None:
    @rank_zero_only
    def init_wandb():
        wandb.login()
    init_wandb()
    ##### Load data
    pl.seed_everything(cfg.model.seed, workers=True)
    print("Getting dataloaders.")
    train_dataloader, test_dataloader = get_dataloaders(cfg)
    tokenizer = train_dataloader.tokenizer if hasattr(train_dataloader, "tokenizer") else None

    ##### Setup x0_model
    print("Setting up model.")
    x0_model_class, nn_params = get_model_setup(cfg, tokenizer) 
    
    print(cfg)
    
    ##### Pick model
    model_name_dict = {"SCUD":SCUD,
                       "SM_SCUD_TCR": SM_SCUD_TCR,
                       "Masking":MaskingDiffusion,
                       "Classical": ClassicalDiffusion,
                       "SM_SCUD_PT": SM_SCUD_PT}
    model_kwargs = {}
    if hasattr(cfg.model, 'correction_weighting'):
        model_kwargs['correction_weighting'] = cfg.model.correction_weighting
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
        tokenizer=tokenizer if cfg.data.data != 'uniref50' else Tokenizer(),
        **OmegaConf.to_container(cfg.train, resolve=True),
        **model_kwargs,
    )
    if cfg.model.restart:
        # Load weights from a previous run's checkpoint (e.g., pretrain → fine-tune).
        # We construct the model fresh with the current config and only transfer weights,
        # so config changes (different loss class, unfrozen modules, etc.) take effect.
        # Kernel buffers (K, K_powers, eigenvalues, etc.) are persistent=False so they
        # are recomputed fresh and not overwritten by load_state_dict.
        ckpt_dir = f'checkpoints/{cfg.model.restart}'
        ckpt_path = max(glob.glob(os.path.join(ckpt_dir, '*.ckpt')), key=os.path.getmtime)
        print(f"Loading weights from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
        if missing:
            print(f"Missing keys (random init): {missing}")
        if unexpected:
            print(f"Unexpected keys (ignored): {unexpected}")
        ckpt_path = None  # don't pass to trainer.fit — we just want weights, not optimizer state
    else:
        ckpt_path = None

    ##### Load data
    model.pre_configure_model(train_dataloader)

    ##### Train
    # wandb.init()
    wandb_logger = WandbLogger(project=cfg.wandb.project, name=cfg.wandb.run_name)
    lightning_model = model
    torch.set_float32_matmul_precision('high')
    @rank_zero_only
    def update_wandb_config():
        wandb_logger.experiment.config.update(lightning_model.hparams)
    update_wandb_config()

    if cfg.data.data == 'uniref50':
        val_check_interval = min(2 * (210000//cfg.train.batch_size), len(train_dataloader))
    else:
        val_check_interval = 1.0
    trainer = Trainer(
        max_epochs=cfg.train.n_epoch, 
        accelerator='auto', 
        devices=torch.cuda.device_count(), 
        logger=wandb_logger, 
        strategy=DDPStrategy(broadcast_buffers=True, find_unused_parameters=True),
        callbacks=([EMA(0.9999)] * cfg.train.ema
                   +[ModelCheckpoint(dirpath=f'checkpoints/{wandb_logger.experiment.name}',
                                   save_on_train_epoch_end=False)]),
        val_check_interval=val_check_interval,
        accumulate_grad_batches=cfg.train.accumulate,
    )
    trainer.fit(lightning_model, train_dataloader, test_dataloader, ckpt_path=ckpt_path)
    wandb.finish()

if __name__ == "__main__":
    train()
