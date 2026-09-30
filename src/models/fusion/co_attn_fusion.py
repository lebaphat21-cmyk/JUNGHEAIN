"""Co-attention fusion module (bidirectional cross-attention).

This is the most sophisticated fusion strategy.  Both modalities attend to
each other simultaneously:

    visual  → attends to semantic (what scene-graph info is relevant here?)
    semantic → attends to visual  (which spatial regions support this triple?)

Each attended representation is refined with a position-wise FFN, then the
two are concatenated and projected.

Output shape: (batch, N_v + N_s, d_model) = (batch, 69, 512).
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.models.fusion.base_fusion import BaseFusion


class CoAttentionFusion(BaseFusion):
    """Bidirectional co-attention feature fusion.

    Pipeline::

        # Visual branch: attend to semantic context
        v_attended = MHA(Q=visual, K=semantic, V=semantic) + residual + LN
        v_refined  = FFN(v_attended) + residual + LN

        # Semantic branch: attend to visual context
        s_attended = MHA(Q=semantic, K=visual, V=visual) + residual + LN
        s_refined  = FFN(s_attended) + residual + LN

        # Merge
        fused = Linear( LayerNorm( [v_refined ; s_refined] ) )

    This approach is expected to yield the **best** captioning quality
    because both modalities are enriched by the other before being passed
    to the decoder.

    Args:
        d_model: Hidden dimension size.
        n_heads: Number of attention heads.
        d_ff: Dimension of the position-wise feed-forward layer.
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

        # =============================================================
        # Visual branch: visual queries attend to semantic keys/values
        # =============================================================
        self.v2s_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.v2s_norm = nn.LayerNorm(d_model)
        self.v2s_dropout = nn.Dropout(dropout)

        self.v_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.v_ffn_norm = nn.LayerNorm(d_model)

        # =============================================================
        # Semantic branch: semantic queries attend to visual keys/values
        # =============================================================
        self.s2v_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.s2v_norm = nn.LayerNorm(d_model)
        self.s2v_dropout = nn.Dropout(dropout)

        self.s_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.s_ffn_norm = nn.LayerNorm(d_model)

        # =============================================================
        # Final output projection after concatenation
        # =============================================================
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
        """Fuse features via bidirectional co-attention.

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

        # =============================================================
        # Visual branch: Q=visual, K/V=semantic
        # "What semantic information is relevant for each spatial region?"
        # =============================================================
        v_attended, _ = self.v2s_attn(
            query=visual_features,
            key=semantic_features,
            value=semantic_features,
            key_padding_mask=semantic_mask,
        )
        v_attended = self.v2s_norm(
            visual_features + self.v2s_dropout(v_attended)
        )  # (B, N_v, d)

        # FFN refinement with residual + LayerNorm
        v_refined = self.v_ffn_norm(
            v_attended + self.v_ffn(v_attended)
        )  # (B, N_v, d)

        # =============================================================
        # Semantic branch: Q=semantic, K/V=visual
        # "Which spatial regions support this scene-graph triple?"
        # =============================================================
        s_attended, _ = self.s2v_attn(
            query=semantic_features,
            key=visual_features,
            value=visual_features,
            key_padding_mask=visual_mask,
        )
        s_attended = self.s2v_norm(
            semantic_features + self.s2v_dropout(s_attended)
        )  # (B, N_s, d)

        # FFN refinement with residual + LayerNorm
        s_refined = self.s_ffn_norm(
            s_attended + self.s_ffn(s_attended)
        )  # (B, N_s, d)

        # =============================================================
        # Concatenate refined branches and project
        # =============================================================
        fused = torch.cat([v_refined, s_refined], dim=1)  # (B, N_v+N_s, d)
        fused = self.output_norm(fused)
        fused = self.output_dropout(self.output_proj(fused))

        # --- Combine masks ---
        fused_mask = self._combine_masks(
            visual_mask, semantic_mask, batch_size, n_v, n_s, fused.device
        )

        return fused, fused_mask
