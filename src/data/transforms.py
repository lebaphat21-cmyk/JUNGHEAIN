"""Image transformation pipelines for COCO Image Captioning.

Provides separate transform compositions for training (with data
augmentation) and validation/testing.  All transforms normalise
with the standard ImageNet channel-wise mean and standard deviation.

Usage::

    train_tfm = get_train_transforms(image_size=256, crop_size=224)
    val_tfm   = get_val_transforms(image_size=256, crop_size=224)
"""

from __future__ import annotations

from torchvision import transforms

# --------------------------------------------------------------------------- #
# ImageNet normalisation constants
# --------------------------------------------------------------------------- #
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def get_train_transforms(
    image_size: int = 256,
    crop_size: int = 224,
) -> transforms.Compose:
    """Return the augmentation pipeline used during training.

    Pipeline:
        1. Resize the shorter side to *image_size* (preserving aspect ratio).
        2. Random crop to (*crop_size* × *crop_size*).
        3. Random horizontal flip (p = 0.5).
        4. Colour jitter (brightness, contrast, saturation).
        5. Convert to tensor [0, 1].
        6. Normalise with ImageNet mean/std.

    Args:
        image_size: Target size for the initial resize.
        crop_size: Spatial size of the random crop.

    Returns:
        A :class:`torchvision.transforms.Compose` object.
    """
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_val_transforms(
    image_size: int = 256,
    crop_size: int = 224,
) -> transforms.Compose:
    """Return the deterministic pipeline used during validation / testing.

    Pipeline:
        1. Resize the shorter side to *image_size*.
        2. Centre crop to (*crop_size* × *crop_size*).
        3. Convert to tensor [0, 1].
        4. Normalise with ImageNet mean/std.

    Args:
        image_size: Target size for the initial resize.
        crop_size: Spatial size of the centre crop.

    Returns:
        A :class:`torchvision.transforms.Compose` object.
    """
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
