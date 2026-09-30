"""Gated fusion module.

A learnable gate decides, *per spatial position*, how much to trust the
visual signal versus the semantically-enriched signal.  First, semantic
features are projected into the visual feature space via cross-attention
(so both have shape ``(batch, N_v, d_model)``).  Then a sigmoid gate
controls the interpolation:

    gate  = σ(W_v · visual + W_s · attended_semantic + b)
    fused = gate ⊙ visual + (1 − gate) ⊙ attended_semantic

A position-wise feed-forward network is applied after gating.

Output shape: (batch, N_v, d_model) = (batch, 49, 512).
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.models.fusion.base_fusion import BaseFusion


class GatedFusion(BaseFusion):
    """Gated feature fusion with learned modality weighting.

    The gate mechanism allows the model to adaptively decide, at each
    spatial position, whether the visual or semantic modality is more
    informative.  This is especially useful when some image regions have
    no corresponding scene-graph triple.

    Args:
        d_model: Hidden dimension size.
        n_heads: Number of attention heads in the cross-attention.
        d_ff: Dimension of the internal feed-forward layer.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(d_model)

        # --- Cross-attention to align semantic → visual sequence length ---
        # Q = visual (N_v positions), K/V = semantic
        # Result has shape (batch, N_v, d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn_dropout = nn.Dropout(dropout)

        # --- Gating mechanism ---
        # Projects both modalities to a gate value in [0, 1]
        self.gate_visual = nn.Linear(d_model, d_model)
        self.gate_semantic = nn.Linear(d_model, d_model)
        self.gate_bias = nn.Parameter(torch.zeros(d_model))

        # --- Position-wise Feed-Forward Network ---
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        visual_features: torch.Tensor,
        semantic_features: torch.Tensor,
        visual_mask: Optional[torch.Tensor] = None,
        semantic_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Fuse via gated interpolation of visual and semantic features.

        Args:
            visual_features: ``(batch, N_v, d_model)`` with N_v = 49.
            semantic_features: ``(batch, N_s, d_model)`` with N_s = 20.
            visual_mask: ``(batch, N_v)`` bool, ``True`` = padded.
            semantic_mask: ``(batch, N_s)`` bool, ``True`` = padded.

        Returns:
            Tuple of:
                - **fused_features**: ``(batch, N_v, d_model)`` = (B, 49, 512)
                - **fused_mask**: ``(batch, N_v)`` or ``None``

        Note:
            The output sequence length equals N_v (49) because the gate
            operates in the visual feature space.
        """
        # ----------------------------------------------------------------
        # 1. Cross-attention: visual queries attend to semantic keys/values
        #    This produces visual-aligned semantic features.
        # ----------------------------------------------------------------
        attended_sem, _ = self.cross_attn(
            query=visual_features,       # (B, N_v, d)
            key=semantic_features,       # (B, N_s, d)
            value=semantic_features,     # (B, N_s, d)
            key_padding_mask=semantic_mask,  # ignore padded triples
        )
        # Residual around the cross-attention uses *visual* as the identity
        # because attended_sem lives in the visual position space.
        # We treat this as a "semantic enrichment" of visual positions.
        attended_sem = self.cross_attn_norm(
            visual_features + self.cross_attn_dropout(attended_sem)
        )
        # attended_sem: (batch, N_v, d_model)

        # ----------------------------------------------------------------
        # 2. Compute the gating signal
        # ----------------------------------------------------------------
        # gate ∈ (0, 1) per position per feature dimension
        gate = torch.sigmoid(
            self.gate_visual(visual_features)
            + self.gate_semantic(attended_sem)
            + self.gate_bias
        )  # (batch, N_v, d_model)

        # Gated combination
        fused = gate * visual_features + (1.0 - gate) * attended_sem
        # fused: (batch, N_v, d_model)

        # ----------------------------------------------------------------
        # 3. Feed-forward refinement with residual + LayerNorm
        # ----------------------------------------------------------------
        fused = self.ffn_norm(fused + self.ffn(fused))
        # fused: (batch, N_v, d_model)

        # The output mask matches the visual positions
        fused_mask = visual_mask  # (batch, N_v) or None

        return fused, fused_mask
