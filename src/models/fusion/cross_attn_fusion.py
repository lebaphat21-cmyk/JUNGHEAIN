"""Cross-attention fusion module.

Semantic features attend to visual features via multi-head cross-attention,
producing *visually-grounded* semantic representations.  The attended
semantic features are further refined with a self-attention layer before
being concatenated with the original visual features.

Output shape: (batch, N_v + N_s, d_model) = (batch, 69, 512).
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.models.fusion.base_fusion import BaseFusion


class CrossAttentionFusion(BaseFusion):
    """Cross-attention based feature fusion.

    Pipeline::

        # 1. Semantic → Visual cross-attention
        attended_sem = MHA(Q=semantic, K=visual, V=visual)
        attended_sem = LayerNorm(semantic + attended_sem)     # residual

        # 2. Self-attention refinement on attended semantics
        refined_sem  = MHA_self(attended_sem)
        refined_sem  = LayerNorm(attended_sem + refined_sem)  # residual

        # 3. Concatenate and project
        fused = Linear(LayerNorm([visual ; refined_sem]))

    Args:
        d_model: Hidden dimension size.
        n_heads: Number of attention heads.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(d_model)

        # --- Cross-attention: semantic attends to visual ---
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn_dropout = nn.Dropout(dropout)

        # --- Self-attention refinement on attended semantics ---
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_attn_dropout = nn.Dropout(dropout)

        # --- Output projection ---
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self.output_dropout = nn.Dropout(dropout)

    def forward(
        self,
        visual_features: torch.Tensor,
        semantic_features: torch.Tensor,
        visual_mask: Optional[torch.Tensor] = None,
        semantic_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Fuse via cross-attention from semantic to visual.

        Args:
            visual_features: ``(batch, N_v, d_model)`` with N_v = 49.
            semantic_features: ``(batch, N_s, d_model)`` with N_s = 20.
            visual_mask: ``(batch, N_v)`` bool, ``True`` = padded.
            semantic_mask: ``(batch, N_s)`` bool, ``True`` = padded.

        Returns:
            Tuple of:
                - **fused_features**: ``(batch, N_v + N_s, d_model)``
                - **fused_mask**: ``(batch, N_v + N_s)`` or ``None``
        """
        batch_size = visual_features.size(0)
        n_v = visual_features.size(1)
        n_s = semantic_features.size(1)

        # ----------------------------------------------------------------
        # 1. Cross-attention: semantic queries attend to visual keys/values
        # ----------------------------------------------------------------
        # key_padding_mask marks **visual** positions that are padded so
        # the attention mechanism ignores them.
        attended_sem, _ = self.cross_attn(
            query=semantic_features,
            key=visual_features,
            value=visual_features,
            key_padding_mask=visual_mask,  # (batch, N_v) or None
        )
        # Residual + LayerNorm (Post-LN style)
        attended_sem = self.cross_attn_norm(
            semantic_features + self.cross_attn_dropout(attended_sem)
        )

        # ----------------------------------------------------------------
        # 2. Self-attention refinement on the attended semantics
        # ----------------------------------------------------------------
        refined_sem, _ = self.self_attn(
            query=attended_sem,
            key=attended_sem,
            value=attended_sem,
            key_padding_mask=semantic_mask,  # (batch, N_s) or None
        )
        refined_sem = self.self_attn_norm(
            attended_sem + self.self_attn_dropout(refined_sem)
        )

        # ----------------------------------------------------------------
        # 3. Concatenate original visual + refined semantic, then project
        # ----------------------------------------------------------------
        fused = torch.cat([visual_features, refined_sem], dim=1)  # (B, 69, d)
        fused = self.output_norm(fused)
        fused = self.output_dropout(self.output_proj(fused))

        # --- Combine masks ---
        fused_mask = self._combine_masks(
            visual_mask, semantic_mask, batch_size, n_v, n_s, fused.device
        )

        return fused, fused_mask
