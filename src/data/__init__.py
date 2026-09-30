"""Data pipeline for NCKH Image Captioning.

Re-exports the main classes and factory functions so that users can write::

    from src.data import COCODataset, Vocabulary, get_train_transforms
"""

from src.data.coco_dataset import COCODataset, get_dataloader
from src.data.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_train_transforms,
    get_val_transforms,
)
from src.data.vocabulary import (
    END_IDX,
    END_TOKEN,
    PAD_IDX,
    PAD_TOKEN,
    START_IDX,
    START_TOKEN,
    UNK_IDX,
    UNK_TOKEN,
    Vocabulary,
)

__all__ = [
    # Dataset
    "COCODataset",
    "get_dataloader",
    # Vocabulary
    "Vocabulary",
    "PAD_TOKEN",
    "START_TOKEN",
    "END_TOKEN",
    "UNK_TOKEN",
    "PAD_IDX",
    "START_IDX",
    "END_IDX",
    "UNK_IDX",
    # Transforms
    "get_train_transforms",
    "get_val_transforms",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
]
