# SCUD: Schedule-Conditioned Discrete Diffusion

[Alan N Amin](https://alannawzadamin.github.io), [Nate Gruver](https://ngruver.github.io), [Andrew Gordon Wilson](https://cims.nyu.edu/~andrewgw/).

[Paper](https://arxiv.org/abs/2506.08316)

<p align="center">
  <img width="321" alt="concept" src="https://github.com/user-attachments/assets/40daf76e-8381-4986-b8f9-fc9d93571bf9" />
</p>

## Description

Masking discrete diffusion makes use of a fundamental difference between continuous and discrete Markov processes: discrete Markov processes evolve by discontinuous jumps at a fixed rate and, unlike other discrete diffusion models, masking diffusion *builds in the known distribution of jump times* and only learns where to jump to. We show that we can similarly bake in the known distribution of jump times into *any* discrete diffusion model. The resulting models -- schedule-conditioned diffusion (SCUD) -- generalize classical discrete diffusion and masking diffusion. By applying SCUD to models with noising processes that incorporate inductive biases on images, text, and protein data, we build diffusion models that outperform masking.

This codebase implements schedule-conditioned diffusion (**SCUD**). We provide instructions to train models on image and protein data. We also include code to train masking diffusion or classical diffusion models.

----

## Installation

```bash
uv venv && uv pip install -e ".[dev]"
```

Or with pip:

```bash
pip install -e ".[dev]"
```

## Usage

### Training

Train a small U-Net on CIFAR10 with 128 states:

```bash
scud-train
# or equivalently:
python train.py
```

Override config values from the command line:

```bash
scud-train model.model=Masking model.schedule_type=cos data.N=64
scud-train --config-name=basic_protein
```

### Sampling

Generate samples from a trained checkpoint:

```bash
scud-sample model.restart=<checkpoint_folder> sampling.num_samples=16
```

### Evaluation

Evaluate a trained checkpoint on test data (computes NLL and BPD):

```bash
scud-evaluate model.restart=<checkpoint_folder>
```

## Reproducing Paper Results

We provide config files matching the paper's experimental settings:

| Config | Description |
|--------|-------------|
| `configs/paper_cifar10.yaml` | CIFAR-10 with 128 states, Gaussian forward process |
| `configs/paper_cifar10_large.yaml` | CIFAR-10 with 256 states, full resolution |
| `configs/paper_protein.yaml` | UniRef50 protein sequences with BLOSUM forward process |

Example:

```bash
scud-train --config-name=paper_cifar10
```

### Training protein models

To train protein models, download Uniref50 data from [here](https://zenodo.org/records/6564798) and place it in `data/uniref_2020/uniref50/`.
Also download the BLOSUM62 matrix from [here](https://github.com/microsoft/evodiff/blob/main/data/blosum62-special-MSA.mat) and place it in `data/blosum62-special-MSA.mat`.

## Configuration

`model.model` can be set to `SCUD`, `Masking`, or `Classical`.
`model.gamma` controls the conditioning parameter. `model.schedule_type` controls the noise rate function and can be set to `linear`, `cos`, or `mutual_information`.
`model.forward_kwargs` controls the forward process; see `get_inf_gen` in `scud/utils.py` for choices.
`model.logistic_pars` toggles the logistic parameterization for image data.
Set `model.restart` to the folder of a checkpoint to restart training.

### Note on gamma convention

The gamma convention in code is **inverted** from the paper: code `gamma=0` corresponds to paper `gamma=1` (full schedule conditioning). See the detailed comment in [`scud/scud.py`](scud/scud.py).

### Note on noise rate function

We choose our function beta(t) to linearly decrease the mutual information in time.
This involves finding a zero using Newton's method, which can slow down training when there are too many states.
When there are more than 200 states, we precompute the values at a resolution of 1e-6 and save them in `data/save_alphas` before beginning training. This can take up to an hour the first time.

## Citation

```bibtex
@article{amin2025masking,
  title={Why Masking Diffusion Works: Condition on the Jump Schedule for Improved Discrete Diffusion},
  author={Amin, Alan N and Gruver, Nate and Wilson, Andrew Gordon},
  journal={arXiv preprint arXiv:2506.08316},
  year={2025}
}
```
