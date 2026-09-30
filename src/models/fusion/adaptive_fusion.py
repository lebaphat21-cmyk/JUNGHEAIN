"""Adaptive Mixture-of-Fusion module.

This module dynamically computes combination weights (via a routing network)
for multiple fusion strategies on a per-sample basis, combining their outputs
via a soft-attention weighted sum.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.fusion.base_fusion import BaseFusion
from src.models.fusion.concat_fusion import ConcatenationFusion
from src.models.fusion.cross_attn_fusion import CrossAttentionFusion
from src.models.fusion.gated_fusion import GatedFusion
from src.models.fusion.co_attn_fusion import CoAttentionFusion
from src.models.fusion.gated_co_attn_fusion import GatedCoAttentionFusion

_STRATEGY_MAP = {
    "concat": ConcatenationFusion,
    "concatenation": ConcatenationFusion,
    "cross_attention": CrossAttentionFusion,
    "cross_attn": CrossAttentionFusion,
    "gated": GatedFusion,
    "gated_fusion": GatedFusion,
    "co_attention": CoAttentionFusion,
    "co_attn": CoAttentionFusion,
    "gated_co_attention": GatedCoAttentionFusion,
    "gated_co_attn": GatedCoAttentionFusion,
}


class AdaptiveFusion(BaseFusion):
    """Adaptive Mixture-of-Fusion module.

    Runs multiple fusion strategies in parallel and combines them dynamically using
    routing weights computed from the global average-pooled visual and semantic features.
    """

    def __init__(
        self,
        d_model: int = 512,
        types: Optional[list[str]] = None,
        **kwargs
    ) -> None:
        super().__init__(d_model)

        if types is None:
            types = ["concat", "cross_attn", "gated", "gated_co_attn"]

        self.fusion_types = [t.lower().strip() for t in types]
        self.fusers = nn.ModuleList()

        import inspect
        for t in self.fusion_types:
            if t not in _STRATEGY_MAP:
                raise ValueError(
                    f"Unknown fusion type '{t}' for AdaptiveFusion. "
                    f"Available types: {list(_STRATEGY_MAP.keys())}"
                )
            fuser_cls = _STRATEGY_MAP[t]

            # Only pass kwargs that the target constructor actually accepts,
            # so that simple fusers (e.g. ConcatenationFusion) do not error
            # when receiving attention-specific keys like n_heads or d_ff.
            valid_params = set(inspect.signature(fuser_cls.__init__).parameters.keys())
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
            self.fusers.append(fuser_cls(d_model=d_model, **filtered_kwargs))

        # Routing network: pools visual + semantic, maps to strategy weights
        self.router = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(kwargs.get("dropout", 0.1)),
            nn.Linear(d_model, len(self.fusion_types)),
        )

    def forward(
        self,
        visual_features: torch.Tensor,
        semantic_features: torch.Tensor,
        visual_mask: Optional[torch.Tensor] = None,
        semantic_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Fuse features using a dynamic mixture of multiple fusion strategies.

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
        device = visual_features.device

        # 1. Compute routing logits from pooled features
        v_pool = visual_features.mean(dim=1)  # (B, d_model)
        s_pool = semantic_features.mean(dim=1)  # (B, d_model)
        combined_pool = torch.cat([v_pool, s_pool], dim=-1)  # (B, 2 * d_model)

        logits = self.router(combined_pool)  # (B, num_fusers)
        weights = F.softmax(logits, dim=-1)  # (B, num_fusers)

        # 2. Run all fusers and align shapes to (B, N_v + N_s, d_model)
        fused_outputs = []
        for fuser in self.fusers:
            out, _ = fuser(visual_features, semantic_features, visual_mask, semantic_mask)

            # If strategy outputs length N_v (like GatedFusion), zero-pad it to N_v + N_s
            if out.size(1) == n_v:
                out = F.pad(out, (0, 0, 0, n_s), value=0.0)

            fused_outputs.append(out)

        # (B, num_fusers, N_v + N_s, d_model)
        stacked = torch.stack(fused_outputs, dim=1)

        # Weighted sum: (B, N_v + N_s, d_model)
        fused = (stacked * weights.view(batch_size, -1, 1, 1)).sum(dim=1)

        # 3. Combined mask is standard (B, N_v + N_s)
        fused_mask = self._combine_masks(
            visual_mask, semantic_mask, batch_size, n_v, n_s, device
        )

        return fused, fused_mask
