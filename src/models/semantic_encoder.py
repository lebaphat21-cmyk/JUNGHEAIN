"""Semantic Encoder module for Image Captioning.

Encodes semantic information extracted from scene graph triples
(subject, predicate, object) using shared word embeddings, triple-level
fusion, and a Transformer Encoder.

Typical usage:
    encoder = SemanticEncoder(vocab_size=10000, d_model=512)
    # triples: (B, max_triples, 3)  — vocab indices for (subj, pred, obj)
    # triple_mask: (B, max_triples)  — True for padded positions
    semantic_features, semantic_mask = encoder(triples, triple_mask)
    # semantic_features: (B, max_triples, 512)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class TripleFusion(nn.Module):
    """Fuses a scene-graph triple (subject, predicate, object) into a single vector.

    Each of the three elements is independently embedded and projected to
    ``d_model`` dimensions, then the three are concatenated and mapped back
    to ``d_model`` via a linear layer:

        triple_emb = Linear( [subj_emb || pred_emb || obj_emb] )

    Args:
        d_model: Output (and per-element) dimension.
        dropout: Dropout applied after fusion.
    """

    def __init__(self, d_model: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        subj_emb: torch.Tensor,
        pred_emb: torch.Tensor,
        obj_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse three per-element embeddings into one triple embedding.

        Args:
            subj_emb: Subject embeddings ``(batch, max_triples, d_model)``.
            pred_emb: Predicate embeddings ``(batch, max_triples, d_model)``.
            obj_emb: Object embeddings ``(batch, max_triples, d_model)``.

        Returns:
            Fused triple embeddings ``(batch, max_triples, d_model)``.
        """
        # Concatenate along feature dimension
        concat = torch.cat([subj_emb, pred_emb, obj_emb], dim=-1)  # (B, N_s, 3*d_model)
        return self.fusion(concat)  # (B, N_s, d_model)


class SemanticEncoder(nn.Module):
    """Encodes semantic information from scene graph triples using embeddings + Transformer Encoder.

    Pipeline:
        1. Shared embedding layer (vocab → embed_dim=300)
        2. Linear projection: embed_dim → d_model
        3. Triple fusion: concat(subj, pred, obj) → Linear → d_model
        4. Learnable positional encoding for the triple sequence
        5. Transformer Encoder with src_key_padding_mask

    Args:
        vocab_size: Size of the shared vocabulary (entities + predicates).
        d_model: Hidden dimension (default 512).
        embed_dim: Embedding dimension, matching GloVe (default 300).
        n_heads: Number of attention heads (default 8).
        n_layers: Number of Transformer encoder layers (default 2).
        d_ff: Feed-forward inner dimension (default 2048).
        max_triples: Maximum number of triples per image (default 20).
        dropout: Dropout rate (default 0.1).
        padding_idx: Index for the ``<pad>`` token in the shared vocab
            (default 0).

    Example:
        >>> encoder = SemanticEncoder(vocab_size=5000)
        >>> triples = torch.randint(1, 5000, (4, 20, 3))
        >>> mask = torch.zeros(4, 20, dtype=torch.bool)
        >>> mask[:, 15:] = True  # last 5 positions are padding
        >>> feats, out_mask = encoder(triples, mask)
        >>> feats.shape
        torch.Size([4, 20, 512])
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        embed_dim: int = 300,
        n_heads: int = 8,
        n_layers: int = 2,
        d_ff: int = 2048,
        max_triples: int = 20,
        dropout: float = 0.1,
        padding_idx: int = 0,
        use_confidence: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.embed_dim = embed_dim
        self.max_triples = max_triples
        self.use_confidence = use_confidence

        # ---- 1. Shared Word Embedding ----
        self.word_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=padding_idx,
        )

        # ---- 2. Projection: embed_dim (300) → d_model (512) ----
        self.embed_projection = nn.Sequential(
            nn.Linear(embed_dim, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # ---- 3. Triple Fusion ----
        self.triple_fusion = TripleFusion(d_model=d_model, dropout=dropout)

        # ---- 3.5. Confidence Estimator (Confidence-aware SG) ----
        if self.use_confidence:
            self.confidence_estimator = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
                nn.Sigmoid(),
            )

        # ---- 4. Learnable Positional Encoding ----
        self.positional_encoding = nn.Embedding(max_triples, d_model)

        # ---- 5. Pre-norm for Transformer input ----
        self.pre_norm = nn.LayerNorm(d_model)

        # ---- 6. Transformer Encoder ----
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

        # Initialize parameters
        self._init_parameters()

    # ------------------------------------------------------------------
    # Parameter initialisation
    # ------------------------------------------------------------------

    def _init_parameters(self) -> None:
        """Xavier-uniform initialisation for linear layers; normal for embeddings."""
        # Positional encoding
        nn.init.normal_(self.positional_encoding.weight, mean=0.0, std=0.02)

        # Projection and fusion linear layers
        for module in [self.embed_projection, self.triple_fusion]:
            for submodule in module.modules():
                if isinstance(submodule, nn.Linear):
                    nn.init.xavier_uniform_(submodule.weight)
                    if submodule.bias is not None:
                        nn.init.zeros_(submodule.bias)

        if self.use_confidence:
            for submodule in self.confidence_estimator.modules():
                if isinstance(submodule, nn.Linear):
                    nn.init.xavier_uniform_(submodule.weight)
                    if submodule.bias is not None:
                        nn.init.zeros_(submodule.bias)

    # ------------------------------------------------------------------
    # GloVe embedding loading
    # ------------------------------------------------------------------

    def load_glove_embeddings(
        self,
        glove_matrix: torch.Tensor,
        freeze: bool = False,
    ) -> None:
        """Load pre-trained GloVe vectors into the word embedding layer.

        Args:
            glove_matrix: Pre-trained embedding matrix of shape
                ``(vocab_size, embed_dim)``.  Rows should be ordered to
                match the vocabulary indices.  The row at ``padding_idx``
                will automatically be zeroed out by PyTorch.
            freeze: If ``True``, the embedding weights will not be updated
                during training (default ``False``).

        Raises:
            ValueError: If the matrix shape does not match
                ``(vocab_size, embed_dim)``.
        """
        expected_shape = (
            self.word_embedding.num_embeddings,
            self.word_embedding.embedding_dim,
        )
        if glove_matrix.shape != expected_shape:
            raise ValueError(
                f"GloVe matrix shape {glove_matrix.shape} does not match "
                f"expected shape {expected_shape} (vocab_size, embed_dim)."
            )

        self.word_embedding.weight = nn.Parameter(
            glove_matrix.clone(), requires_grad=not freeze
        )

        # Re-zero the padding vector (safety measure)
        if self.word_embedding.padding_idx is not None:
            with torch.no_grad():
                self.word_embedding.weight[self.word_embedding.padding_idx].zero_()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        triples: torch.Tensor,
        triple_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode scene-graph triples into semantic feature vectors.

        Args:
            triples: Integer tensor of vocab indices, shape
                ``(batch, max_triples, 3)``.
            triple_mask: Boolean padding mask, shape
                ``(batch, max_triples)``.

        Returns:
            Tuple of:
                - **semantic_features**: Encoded features, ``(batch, max_triples, d_model)``.
                - **semantic_mask**: The input ``triple_mask``.
        """
        batch_size, num_triples, _ = triples.shape

        # --- 1. Embed each element of the triple ---
        subj_idx = triples[:, :, 0]
        pred_idx = triples[:, :, 1]
        obj_idx  = triples[:, :, 2]

        subj_emb = self.word_embedding(subj_idx)
        pred_emb = self.word_embedding(pred_idx)
        obj_emb  = self.word_embedding(obj_idx)

        # --- 2. Project each embedding: embed_dim → d_model ---
        subj_emb = self.embed_projection(subj_emb)
        pred_emb = self.embed_projection(pred_emb)
        obj_emb  = self.embed_projection(obj_emb)

        # --- 3. Triple Fusion: concat + linear → d_model ---
        fused = self.triple_fusion(subj_emb, pred_emb, obj_emb)

        # --- 3.5. Confidence Gating ---
        if self.use_confidence:
            confidence = self.confidence_estimator(fused)  # (B, N_s, 1)
            fused = fused * confidence

        # --- 4. Add learnable positional encoding ---
        positions = torch.arange(num_triples, device=triples.device)
        pos_emb = self.positional_encoding(positions)
        fused = fused + pos_emb.unsqueeze(0)

        # --- 5. Pre-normalise ---
        fused = self.pre_norm(fused)

        # --- 6. Build mask (default: no padding) ---
        if triple_mask is None:
            triple_mask = torch.zeros(
                batch_size, num_triples, dtype=torch.bool, device=triples.device
            )
        else:
            all_padded = triple_mask.all(dim=-1)
            if all_padded.any():
                triple_mask = triple_mask.clone()
                triple_mask[all_padded, 0] = False

        # --- 7. Transformer Encoder ---
        semantic_features = self.transformer_encoder(
            fused, src_key_padding_mask=triple_mask
        )

        return semantic_features, triple_mask

    def forward_intermediate(
        self,
        triples: torch.Tensor,
        triple_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[list[torch.Tensor], torch.Tensor]:
        """Encode scene-graph triples and return features after each Transformer Encoder layer.

        Args:
            triples: Integer tensor of vocab indices, shape ``(batch, max_triples, 3)``.
            triple_mask: Boolean padding mask, shape ``(batch, max_triples)``.

        Returns:
            Tuple of:
                - **layer_features**: List of ``(batch, max_triples, d_model)`` tensors,
                  one per Transformer Encoder layer.
                - **semantic_mask**: The processed ``triple_mask``.
        """
        batch_size, num_triples, _ = triples.shape

        subj_idx = triples[:, :, 0]
        pred_idx = triples[:, :, 1]
        obj_idx  = triples[:, :, 2]

        subj_emb = self.word_embedding(subj_idx)
        pred_emb = self.word_embedding(pred_idx)
        obj_emb  = self.word_embedding(obj_idx)

        subj_emb = self.embed_projection(subj_emb)
        pred_emb = self.embed_projection(pred_emb)
        obj_emb  = self.embed_projection(obj_emb)

        fused = self.triple_fusion(subj_emb, pred_emb, obj_emb)

        if self.use_confidence:
            confidence = self.confidence_estimator(fused)
            fused = fused * confidence

        positions = torch.arange(num_triples, device=triples.device)
        pos_emb = self.positional_encoding(positions)
        fused = fused + pos_emb.unsqueeze(0)

        fused = self.pre_norm(fused)

        if triple_mask is None:
            triple_mask = torch.zeros(
                batch_size, num_triples, dtype=torch.bool, device=triples.device
            )
        else:
            all_padded = triple_mask.all(dim=-1)
            if all_padded.any():
                triple_mask = triple_mask.clone()
                triple_mask[all_padded, 0] = False

        # Run Transformer Encoder layer by layer to extract intermediate states
        layer_features = []
        x = fused
        for layer in self.transformer_encoder.layers:
            x = layer(x, src_key_padding_mask=triple_mask)
            layer_features.append(x)

        # Apply final LayerNorm if there's one in TransformerEncoder
        if self.transformer_encoder.norm is not None:
            layer_features = [self.transformer_encoder.norm(feat) for feat in layer_features]

        return layer_features, triple_mask
