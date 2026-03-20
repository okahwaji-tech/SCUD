"""SCUD: Schedule-Conditioned Discrete Diffusion.

A framework for discrete diffusion models that conditions on the jump schedule,
generalizing both masking diffusion and classical discrete diffusion.

Paper: "Why Masking Diffusion Works" (NeurIPS 2025)
"""

from scud.classical_diffusion import ClassicalDiffusion
from scud.masking_diffusion import MaskingDiffusion
from scud.scud import SCUD

__version__ = "0.1.0"
__all__ = ["SCUD", "MaskingDiffusion", "ClassicalDiffusion"]
