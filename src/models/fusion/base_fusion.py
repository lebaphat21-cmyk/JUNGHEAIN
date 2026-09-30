"""Base class for all fusion modules.

This module defines the abstract interface that all fusion strategies must
implement. Each fusion module takes visual and semantic features (with
optional padding masks) and returns a fused representation.

Architecture Context:
    - Visual features come from ResNet-101 encoder: (batch, 49, 512)
    - Semantic features come from Scene Graph encoder: (batch, 20, 512)
    - Masks use the convention: True = padded (ignored) position
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn


class BaseFusion(nn.Module, ABC):
    """Abstract base class for feature fusion modules.

    All fusion strategies inherit from this class and implement the
    ``forward`` method to combine visual and semantic features into a
    single fused representation that is fed to the Transformer decoder.

    Attributes:
        d_model: Hidden dimension size (default 512).
    """

    def __init__(self, d_model: int = 512) -> None:
        """Initialise the base fusion module.

        Args:
            d_model: Hidden dimension for all features. Must match the
                output dimensions of both the visual and semantic encoders.
        """
        super().__init__()
        self.d_model = d_model

    @abstractmethod
    def forward(
        self,
        visual_features: torch.Tensor,
        semantic_features: torch.Tensor,
        visual_mask: Optional[torch.Tensor] = None,
        semantic_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Fuse visual and semantic features.

        Args:
            visual_features: Visual feature tensor of shape
                ``(batch, N_v, d_model)`` where ``N_v = 49`` (7×7 spatial
                grid from ResNet-101).
            semantic_features: Semantic feature tensor of shape
                ``(batch, N_s, d_model)`` where ``N_s = 20`` (max scene-
                graph triples).
            visual_mask: Optional boolean mask of shape ``(batch, N_v)``.
                ``True`` indicates a **padded** position that should be
                ignored during attention.
            semantic_mask: Optional boolean mask of shape ``(batch, N_s)``.
                ``True`` indicates a **padded** position.

        Returns:
            A tuple of:
                - **fused_features** – ``(batch, N_fused, d_model)`` tensor
                  containing the merged representation.
                - **fused_mask** – ``(batch, N_fused)`` boolean mask (or
                  ``None`` if no masks were provided).
        """
        raise NotImplementedError

    def _combine_masks(
        self,
        visual_mask: Optional[torch.Tensor],
        semantic_mask: Optional[torch.Tensor],
        batch_size: int,
        n_v: int,
        n_s: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Concatenate visual and semantic masks along the sequence dim.

        If both masks are ``None`` the method returns ``None``. If only one
        mask is provided, the other is assumed to be all-``False`` (i.e. no
        padding).

        Args:
            visual_mask: ``(batch, N_v)`` or ``None``.
            semantic_mask: ``(batch, N_s)`` or ``None``.
            batch_size: Batch size ``B``.
            n_v: Number of visual positions.
            n_s: Number of semantic positions.
            device: Device to create default masks on.

        Returns:
            Concatenated mask ``(batch, N_v + N_s)`` or ``None``.
        """
        if visual_mask is None and semantic_mask is None:
            return None

        if visual_mask is None:
            visual_mask = torch.zeros(batch_size, n_v, dtype=torch.bool, device=device)
        if semantic_mask is None:
            semantic_mask = torch.zeros(batch_size, n_s, dtype=torch.bool, device=device)

        # (batch, N_v + N_s)
        return torch.cat([visual_mask, semantic_mask], dim=1)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}"
