"""Concatenation fusion – the baseline fusion strategy.

This is the simplest fusion approach: visual and semantic features are
concatenated along the sequence dimension, then normalised and projected
through a linear layer to produce a unified representation.

Output shape: (batch, N_v + N_s, d_model) = (batch, 69, 512).
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.models.fusion.base_fusion import BaseFusion


class ConcatenationFusion(BaseFusion):
    """Concatenation-based feature fusion (baseline).

    Pipeline::

        fused = LayerNorm( [visual ; semantic] )   # (B, 69, 512)
        fused = Linear(fused)                       # (B, 69, 512)

    This strategy makes **no** modality interaction – visual and semantic
    tokens are simply placed side-by-side and left for the downstream
    decoder cross-attention to learn how to attend to each modality.

    Args:
        d_model: Hidden dimension (must match encoder outputs).
        dropout: Dropout probability applied after the linear projection.
    """

    def __init__(self, d_model: int = 512, dropout: float = 0.1) -> None:
        super().__init__(d_model)

        self.layer_norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        visual_features: torch.Tensor,
        semantic_features: torch.Tensor,
        visual_mask: Optional[torch.Tensor] = None,
        semantic_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Concatenate visual and semantic features.

        Args:
            visual_features: ``(batch, N_v, d_model)`` – visual encoder
                output (N_v = 49).
            semantic_features: ``(batch, N_s, d_model)`` – semantic encoder
                output (N_s = 20).
            visual_mask: ``(batch, N_v)`` bool mask, ``True`` = padded.
            semantic_mask: ``(batch, N_s)`` bool mask, ``True`` = padded.

        Returns:
            Tuple of:
                - **fused_features**: ``(batch, N_v + N_s, d_model)``
                - **fused_mask**: ``(batch, N_v + N_s)`` or ``None``
        """
        batch_size = visual_features.size(0)
        n_v = visual_features.size(1)
        n_s = semantic_features.size(1)

        # --- Concatenate along sequence dimension ---
        # (batch, N_v + N_s, d_model)
        fused = torch.cat([visual_features, semantic_features], dim=1)

        # --- Normalise then project ---
        fused = self.layer_norm(fused)
        fused = self.dropout(self.projection(fused))

        # --- Combine masks ---
        fused_mask = self._combine_masks(
            visual_mask, semantic_mask, batch_size, n_v, n_s, fused.device
        )

        return fused, fused_mask
