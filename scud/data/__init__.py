"""Data loading for SCUD: image, protein, and text datasets."""

from scud.data.image import get_img_dataloaders
from scud.data.loader import get_dataloaders
from scud.data.protein import get_protein_dataloaders
from scud.data.text import get_text_dataloaders

__all__ = [
    "get_dataloaders",
    "get_img_dataloaders",
    "get_protein_dataloaders",
    "get_text_dataloaders",
]
