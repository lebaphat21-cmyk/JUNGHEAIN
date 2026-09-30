"""Visual Encoder module for Image Captioning.

Extracts and encodes visual features from images using a pre-trained
ResNet-101 backbone followed by a Transformer Encoder with 2D sinusoidal
positional encoding.

Typical usage:
    encoder = VisualEncoder(d_model=512)
    visual_features, visual_mask = encoder(images)  # images: (B, 3, 224, 224)
    # visual_features: (B, 49, 512), visual_mask: (B, 49)
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as models


class SpatialPositionalEncoding2D(nn.Module):
    """2-D sinusoidal positional encoding for a spatial feature grid.

    For a grid of size (H, W), each spatial position (r, c) gets a
    ``d_model``-dimensional encoding constructed by interleaving
    sin/cos signals for the row index and the column index:

        PE[r, c, 2i]     = sin(r / 10000^(4i / d_model))
        PE[r, c, 2i + 1] = cos(r / 10000^(4i / d_model))   (row part)
        PE[r, c, 2j]     = sin(c / 10000^(4j / d_model))
        PE[r, c, 2j + 1] = cos(c / 10000^(4j / d_model))   (col part)

    The first ``d_model // 2`` channels encode the row position and the
    remaining ``d_model // 2`` channels encode the column position.

    Args:
        d_model: Dimensionality of the positional encoding vector.
        max_h: Maximum grid height (default 7 for ResNet layer4).
        max_w: Maximum grid width  (default 7 for ResNet layer4).
    """

    def __init__(self, d_model: int = 512, max_h: int = 7, max_w: int = 7) -> None:
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError(
                f"d_model must be divisible by 4 for 2-D sinusoidal PE, got {d_model}"
            )

        pe = torch.zeros(d_model, max_h, max_w)  # (D, H, W)
        half_d = d_model // 2  # channels allocated to each axis

        # Divisor term: shared log-space frequencies
        div_term = torch.exp(
            torch.arange(0, half_d, 2, dtype=torch.float32)
            * -(math.log(10_000.0) / half_d)
        )  # (half_d // 2,)

        # Row positions → first half of channels
        pos_h = torch.arange(0, max_h, dtype=torch.float32).unsqueeze(1)  # (H, 1)
        # Column positions → second half of channels
        pos_w = torch.arange(0, max_w, dtype=torch.float32).unsqueeze(1)  # (W, 1)

        # sin/cos for rows  → channels [0, half_d)
        pe_row = torch.zeros(half_d, max_h, 1)
        pe_row[0::2, :, 0] = torch.sin(pos_h * div_term).T  # (half_d//2, H)
        pe_row[1::2, :, 0] = torch.cos(pos_h * div_term).T

        # sin/cos for cols  → channels [half_d, d_model)
        pe_col = torch.zeros(half_d, 1, max_w)
        pe_col[0::2, 0, :] = torch.sin(pos_w * div_term).T  # (half_d//2, W)
        pe_col[1::2, 0, :] = torch.cos(pos_w * div_term).T

        # Broadcast and combine: (D, H, W)
        pe[:half_d, :, :] = pe_row.expand(-1, -1, max_w)
        pe[half_d:, :, :] = pe_col.expand(-1, max_h, -1)

        # Register as buffer so it moves with the model but is not a parameter
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, D, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add 2-D positional encoding to spatial feature maps.

        Args:
            x: Feature maps of shape ``(batch, d_model, H, W)``.

        Returns:
            Feature maps with positional encoding added, same shape.
        """
        return x + self.pe[:, :, : x.size(2), : x.size(3)]


class VisualEncoder(nn.Module):
    """Extracts and encodes visual features using a configurable ResNet backbone.

    Pipeline:
        1. ResNet-101 backbone (pre-trained, up to layer4) → (B, 2048, 7, 7)
        2. Linear projection 2048 → d_model                → (B, d_model, 7, 7)
        3. 2-D sinusoidal positional encoding               → (B, d_model, 7, 7)
        4. Reshape to sequence                              → (B, 49, d_model)
        5. Transformer Encoder (n_layers)                   → (B, 49, d_model)

    Args:
        d_model: Hidden dimension throughout the encoder (default 512).
        n_heads: Number of attention heads in Transformer (default 8).
        n_layers: Number of Transformer encoder layers (default 3).
        d_ff: Feed-forward inner dimension (default 2048).
        dropout: Dropout rate (default 0.1).
        freeze_cnn: Whether to freeze ResNet backbone weights at init
            (default ``True``).

    Example:
        >>> encoder = VisualEncoder(d_model=512, freeze_cnn=True)
        >>> images = torch.randn(4, 3, 224, 224)
        >>> features, mask = encoder(images)
        >>> features.shape
        torch.Size([4, 49, 512])
        >>> mask.shape
        torch.Size([4, 49])
    """

    # ResNet-101 layer4 output channels
    _BACKBONE_OUT_DIM: int = 2048
    # Spatial grid size from layer4 with 224×224 input
    _GRID_H: int = 7
    _GRID_W: int = 7

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 2048,
        dropout: float = 0.1,
        freeze_cnn: bool = True,
        backbone: str = "resnet101",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.backbone_name = backbone.lower().strip()

        # ---- 1. Configurable ResNet Backbone ----
        if self.backbone_name == "resnet50":
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        elif self.backbone_name == "resnet101":
            resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
        else:
            raise ValueError(
                f"Unsupported visual backbone '{backbone}'. "
                "Choose 'resnet50' or 'resnet101'."
            )
        # Keep everything up to (and including) layer4; drop avgpool + fc
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        if freeze_cnn:
            self.freeze_backbone()

        # ---- 2. Linear Projection: 2048 → d_model ----
        # Implemented as 1×1 convolution for spatial compatibility
        self.projection = nn.Sequential(
            nn.Conv2d(self._BACKBONE_OUT_DIM, d_model, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

        # ---- 3. 2-D Sinusoidal Positional Encoding ----
        self.pos_encoding = SpatialPositionalEncoding2D(
            d_model=d_model,
            max_h=self._GRID_H,
            max_w=self._GRID_W,
        )

        # ---- 4. LayerNorm before Transformer (pre-norm stabilisation) ----
        self.pre_norm = nn.LayerNorm(d_model)

        # ---- 5. Transformer Encoder ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,  # post-norm (standard)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),  # final LayerNorm
        )

    # ------------------------------------------------------------------
    # Backbone freezing / unfreezing
    # ------------------------------------------------------------------

    def freeze_backbone(self) -> None:
        """Freeze all parameters of the ResNet-101 backbone.

        Useful during initial training to preserve pre-trained features.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Also switch to eval mode so BatchNorm uses running stats
        self.backbone.eval()

    def unfreeze_backbone(self) -> None:
        """Unfreeze all parameters of the ResNet-101 backbone.

        Call this for fine-tuning the CNN together with the captioning model
        (typically during SCST / second training stage).
        """
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.backbone.train()

    # ------------------------------------------------------------------
    # Override train() to keep backbone in eval when frozen
    # ------------------------------------------------------------------

    def train(self, mode: bool = True) -> "VisualEncoder":
        """Override ``train()`` to keep frozen backbone in eval mode."""
        super().train(mode)
        # If backbone is frozen, keep it in eval regardless
        if not any(p.requires_grad for p in self.backbone.parameters()):
            self.backbone.eval()
        return self

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract and encode visual features from input images.

        Args:
            images: Batch of images, shape ``(batch, 3, 224, 224)``.
                Expected to be normalised with ImageNet statistics.

        Returns:
            Tuple of:
                - **visual_features**: Encoded features, ``(batch, 49, d_model)``.
                - **visual_mask**: Padding mask, ``(batch, 49)`` — all ``False``
                  because visual features never have padding.
        """
        batch_size = images.size(0)

        # 1. Backbone: (B, 3, 224, 224) → (B, 2048, 7, 7)
        with torch.no_grad() if not any(
            p.requires_grad for p in self.backbone.parameters()
        ) else _nullcontext():
            cnn_features = self.backbone(images)

        # 2. Projection: (B, 2048, 7, 7) → (B, d_model, 7, 7)
        projected = self.projection(cnn_features)

        # 3. Positional encoding: add 2-D sin/cos PE
        projected = self.pos_encoding(projected)

        # 4. Reshape to sequence: (B, d_model, 7, 7) → (B, 49, d_model)
        # Flatten spatial dims and transpose
        features_seq = projected.flatten(2).permute(0, 2, 1)  # (B, 49, d_model)

        # 5. Pre-normalise
        features_seq = self.pre_norm(features_seq)

        # 6. Transformer Encoder: (B, 49, d_model) → (B, 49, d_model)
        visual_features = self.transformer_encoder(features_seq)

        # 7. No padding for visual features → mask is all False
        visual_mask = torch.zeros(
            batch_size, features_seq.size(1), dtype=torch.bool, device=images.device
        )

        return visual_features, visual_mask

    def forward_intermediate(self, images: torch.Tensor) -> Tuple[list[torch.Tensor], torch.Tensor]:
        """Extract and encode visual features, returning features after each Transformer layer.

        Args:
            images: Batch of images, shape ``(batch, 3, 224, 224)``.

        Returns:
            Tuple of:
                - **layer_features**: List of ``(batch, 49, d_model)`` tensors,
                  one per Transformer Encoder layer.
                - **visual_mask**: Padding mask, ``(batch, 49)`` (all False).
        """
        batch_size = images.size(0)

        with torch.no_grad() if not any(
            p.requires_grad for p in self.backbone.parameters()
        ) else _nullcontext():
            cnn_features = self.backbone(images)

        projected = self.projection(cnn_features)
        projected = self.pos_encoding(projected)
        features_seq = projected.flatten(2).permute(0, 2, 1)
        features_seq = self.pre_norm(features_seq)

        layer_features = []
        x = features_seq
        for layer in self.transformer_encoder.layers:
            x = layer(x)
            layer_features.append(x)

        if self.transformer_encoder.norm is not None:
            layer_features = [self.transformer_encoder.norm(feat) for feat in layer_features]

        visual_mask = torch.zeros(
            batch_size, features_seq.size(1), dtype=torch.bool, device=images.device
        )

        return layer_features, visual_mask


# ---------------------------------------------------------------------------
# Utility: null context manager for torch.no_grad() toggle
# ---------------------------------------------------------------------------


class _nullcontext:
    """Minimal no-op context manager (backport for <3.7 compat, though we
    target 3.9+, this keeps the code self-contained)."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        pass
