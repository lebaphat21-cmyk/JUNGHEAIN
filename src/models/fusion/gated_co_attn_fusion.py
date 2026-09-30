"""Gated Co-attention fusion module (bidirectional cross-attention with sigmoid gating).

This module combines bidirectional cross-attention with a sigmoid gating mechanism
to filter out noise (especially from potentially incorrect scene graph triples).
It also supports low-rank linear projection to reduce attention complexity.

Output shape: (batch, N_v + N_s, d_model) = (batch, 69, 512).
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.models.fusion.base_fusion import BaseFusion


class LowRankProjection(nn.Module):
    """Projects features along the sequence dimension to reduce attention complexity.

    Given input of shape (batch, seq_len, d_model), it returns (batch, proj_len, d_model).
    """

    def __init__(self, seq_len: int, proj_len: int, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(seq_len, proj_len)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, seq_len, d)
        # mask: (B, seq_len) with True for padded positions
        if mask is not None:
            # Mask out padding positions before projection so they don't affect output
            x = x.masked_fill(mask.unsqueeze(-1), 0.0)

        # (B, seq_len, d) -> transpose -> (B, d, seq_len) -> linear -> (B, d, proj_len) -> transpose -> (B, proj_len, d)
        x_proj = self.proj(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x_proj)


class GatedCoAttentionFusion(BaseFusion):
    """Bidirectional co-attention feature fusion with gating and linear attention.

    Args:
        d_model: Hidden dimension size.
        n_heads: Number of attention heads.
        d_ff: Dimension of the position-wise feed-forward layer.
        dropout: Dropout probability.
        linear_attention: If True, uses low-rank projection to reduce attention cost.
        proj_len: Sequence length projection size for low-rank attention.
        max_v_len: Maximum visual sequence length (default 49).
        max_s_len: Maximum semantic sequence length (default 20).
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        linear_attention: bool = False,
        proj_len: int = 16,
        max_v_len: int = 49,
        max_s_len: int = 20,
    ) -> None:
        super().__init__(d_model)
        self.linear_attention = linear_attention

        # =============================================================
        # Low-rank projections for linear attention
        # =============================================================
        if self.linear_attention:
            # For visual branch (attending to semantic K/V)
            self.v2s_k_proj = LowRankProjection(max_s_len, proj_len, d_model)
            self.v2s_v_proj = LowRankProjection(max_s_len, proj_len, d_model)
            # For semantic branch (attending to visual K/V)
            self.s2v_k_proj = LowRankProjection(max_v_len, proj_len, d_model)
            self.s2v_v_proj = LowRankProjection(max_v_len, proj_len, d_model)

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

        # Gating for visual branch
        self.gate_v_orig = nn.Linear(d_model, d_model)
        self.gate_v_att = nn.Linear(d_model, d_model)
        self.gate_v_bias = nn.Parameter(torch.zeros(d_model))

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

        # Gating for semantic branch
        self.gate_s_orig = nn.Linear(d_model, d_model)
        self.gate_s_att = nn.Linear(d_model, d_model)
        self.gate_s_bias = nn.Parameter(torch.zeros(d_model))

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
        """Fuse features via gated bidirectional co-attention.

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
        # Visual branch: Q=visual, K/V=semantic (optionally low-rank)
        # =============================================================
        if self.linear_attention:
            k_sem = self.v2s_k_proj(semantic_features, semantic_mask)
            v_sem = self.v2s_v_proj(semantic_features, semantic_mask)
            v_attended, _ = self.v2s_attn(
                query=visual_features,
                key=k_sem,
                value=v_sem,
                key_padding_mask=None,
            )
        else:
            v_attended, _ = self.v2s_attn(
                query=visual_features,
                key=semantic_features,
                value=semantic_features,
                key_padding_mask=semantic_mask,
            )

        # Gated fusion on visual branch (orig visual and attended semantic)
        gate_v = torch.sigmoid(
            self.gate_v_orig(visual_features)
            + self.gate_v_att(v_attended)
            + self.gate_v_bias
        )
        v_fused = gate_v * visual_features + (1.0 - gate_v) * self.v2s_dropout(v_attended)
        v_fused = self.v2s_norm(v_fused)

        # FFN refinement with residual + LayerNorm
        v_refined = self.v_ffn_norm(
            v_fused + self.v_ffn(v_fused)
        )  # (B, N_v, d)

        # =============================================================
        # Semantic branch: Q=semantic, K/V=visual (optionally low-rank)
        # =============================================================
        if self.linear_attention:
            k_vis = self.s2v_k_proj(visual_features, visual_mask)
            v_vis = self.s2v_v_proj(visual_features, visual_mask)
            s_attended, _ = self.s2v_attn(
                query=semantic_features,
                key=k_vis,
                value=v_vis,
                key_padding_mask=None,
            )
        else:
            s_attended, _ = self.s2v_attn(
                query=semantic_features,
                key=visual_features,
                value=visual_features,
                key_padding_mask=visual_mask,
            )

        # Gated fusion on semantic branch (orig semantic and attended visual)
        gate_s = torch.sigmoid(
            self.gate_s_orig(semantic_features)
            + self.gate_s_att(s_attended)
            + self.gate_s_bias
        )
        s_fused = gate_s * semantic_features + (1.0 - gate_s) * self.s2v_dropout(s_attended)
        s_fused = self.s2v_norm(s_fused)

        # FFN refinement with residual + LayerNorm
        s_refined = self.s_ffn_norm(
            s_fused + self.s_ffn(s_fused)
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
