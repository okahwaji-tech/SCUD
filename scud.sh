#!/bin/bash
#SBATCH --job-name=scud-tcr
#SBATCH --account=jbgrp
#SBATCH --partition=gpu-a100-h
#SBATCH --gres=gpu:a100:1
#SBATCH --time=100:00:00
#SBATCH --mem=160G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

export WANDB_API_KEY="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"

CONFIG=${1:-text8_scud_tcr}

python train.py --config-name=$CONFIG
