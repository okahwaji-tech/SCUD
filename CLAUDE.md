# SCUD Project Instructions

## Architecture
- `scud/scud.py` - Core SCUD model (Propositions 4.1-4.4)
- `scud/masking_diffusion.py` - Masking diffusion (Proposition 5.1)
- `scud/classical_diffusion.py` - Classical diffusion baseline
- `scud/continuous_time_diffusion.py` - Base class
- `scud/trainer.py` - PyTorch Lightning training module
- `scud/unet.py` - KingmaUNet for images
- `scud/protein_convnet.py` - ByteNet for proteins
- `scud/utils.py` - Infinitesimal generators, KL utilities
- `scud/mutual_info_schedule.py` - MI-based rate schedules
- `scud/data/` - Data loading (image, protein, text)

## Commands
- Train: `scud-train --config-name=basic`
- Test: `pytest tests/ -v`
- Format: `black --line-length 100 scud/`
- Lint: `ruff check scud/`

## Key Conventions
- Gamma convention is INVERTED from paper (see comment in scud/scud.py)
- Code gamma=0 = full schedule conditioning (paper gamma=1)
- All math in scud/scud.py is verified correct against the paper
- Do NOT modify mathematical logic without verifying against paper equations
