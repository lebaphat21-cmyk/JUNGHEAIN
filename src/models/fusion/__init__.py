"""Fusion modules for combining visual and semantic features.

This package provides four fusion strategies, ordered from simplest to
most sophisticated:

1. **ConcatenationFusion** – Baseline.  Concatenate + project.
2. **CrossAttentionFusion** – Semantic attends to visual.
3. **GatedFusion** – Learnable gate per position.
4. **CoAttentionFusion** – Bidirectional cross-attention (best).

Use :func:`build_fusion` to instantiate a fusion module by name.

Example::

    >>> from src.models.fusion import build_fusion
    >>> fuser = build_fusion("co_attention", d_model=512)
    >>> fused, mask = fuser(visual_feats, semantic_feats, v_mask, s_mask)
"""

from src.models.fusion.base_fusion import BaseFusion
from src.models.fusion.co_attn_fusion import CoAttentionFusion
from src.models.fusion.concat_fusion import ConcatenationFusion
from src.models.fusion.cross_attn_fusion import CrossAttentionFusion
from src.models.fusion.gated_fusion import GatedFusion
from src.models.fusion.gated_co_attn_fusion import GatedCoAttentionFusion
from src.models.fusion.adaptive_fusion import AdaptiveFusion

# Registry mapping short names → fusion classes
_FUSION_REGISTRY: dict[str, type[BaseFusion]] = {
    "concat": ConcatenationFusion,
    "concatenation": ConcatenationFusion,
    "cross_attention": CrossAttentionFusion,
    "cross_attn": CrossAttentionFusion,
    "gated": GatedFusion,
    "gated_fusion": GatedFusion,
    "co_attention": CoAttentionFusion,
    "co_attn": CoAttentionFusion,
    "coattention": CoAttentionFusion,
    "gated_co_attention": GatedCoAttentionFusion,
    "gated_co_attn": GatedCoAttentionFusion,
    "adaptive": AdaptiveFusion,
    "adaptive_fusion": AdaptiveFusion,
}


def build_fusion(fusion_type: str, d_model: int = 512, **kwargs) -> BaseFusion:
    """Factory function to create a fusion module by name.

    Args:
        fusion_type: Name of the fusion strategy.  Accepted values:
            ``'concat'``, ``'cross_attention'``, ``'gated'``,
            ``'co_attention'`` (and common aliases).
        d_model: Hidden dimension (default 512).  Must match the output
            dimension of both the visual and semantic encoders.
        **kwargs: Additional keyword arguments forwarded to the fusion
            class constructor (e.g. ``n_heads``, ``dropout``, ``d_ff``).

    Returns:
        An instance of the requested :class:`BaseFusion` subclass.

    Raises:
        ValueError: If ``fusion_type`` is not recognised.

    Example::

        >>> fuser = build_fusion("gated", d_model=512, dropout=0.1)
        >>> print(fuser)
    """
    key = fusion_type.lower().strip()
    if key not in _FUSION_REGISTRY:
        available = sorted(set(_FUSION_REGISTRY.keys()))
        raise ValueError(
            f"Unknown fusion type '{fusion_type}'. "
            f"Available types: {available}"
        )

    import inspect
    fusion_cls = _FUSION_REGISTRY[key]
    # Only pass kwargs the target class actually accepts so that simple
    # classes (e.g. ConcatenationFusion) don't error on n_heads / d_ff etc.
    valid = set(inspect.signature(fusion_cls.__init__).parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in valid}
    return fusion_cls(d_model=d_model, **filtered)


__all__ = [
    "BaseFusion",
    "ConcatenationFusion",
    "CrossAttentionFusion",
    "GatedFusion",
    "CoAttentionFusion",
    "GatedCoAttentionFusion",
    "AdaptiveFusion",
    "build_fusion",
]
