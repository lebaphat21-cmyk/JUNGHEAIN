"""Caption Decoder for Image Captioning.

Implements a Transformer Decoder that attends to fused visual-semantic features
and auto-regressively generates image captions.

Key design decisions:
    - Sinusoidal positional encoding (no learnable PE) for better generalisation
      to unseen lengths.
    - Weight tying between the word embedding and the output projection, which
      reduces parameters and has been shown to improve performance (Press &
      Wolf, 2017).
    - Beam search with length normalisation for inference.
    - Greedy decode and multinomial sampling for SCST (Self-Critical Sequence
      Training).

Shapes cheat-sheet:
    fused_features : (batch, N_fused, d_model)
    fused_mask     : (batch, N_fused) bool – True = **padded** position
    captions       : (batch, max_len)  – token indices, starts with <start>
    caption_lengths: (batch,)          – actual lengths *including* <start>
"""

from __future__ import annotations

import math
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================================ #
#  Sinusoidal Positional Encoding
# ============================================================================ #
class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017).

    Produces a (1, max_len, d_model) buffer of position embeddings that is
    added to the token embeddings before the decoder stack.  The buffer is
    registered as a non-learnable parameter so it moves to GPU automatically.

    Args:
        d_model:  Hidden dimension (must be even).
        max_len:  Maximum sequence length to pre-compute.
        dropout:  Dropout applied *after* adding positional encoding.
    """

    def __init__(
        self,
        d_model: int = 512,
        max_len: int = 200,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Build the sinusoidal table: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        # Non-learnable buffer – follows .to(device) calls.
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to *x*.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model) with positional signal added.
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ============================================================================ #
#  Caption Decoder
# ============================================================================ #
class CaptionDecoder(nn.Module):
    """Transformer Decoder for generating image captions.

    Architecture overview::

        Word Embedding + Sinusoidal PE
                   ↓
        ┌──────────────────────────────────────┐
        │  Transformer Decoder Layer  × N      │
        │    ├─ Masked Self-Attention           │
        │    ├─ Cross-Attention (→ fused feats) │
        │    └─ Position-wise FFN               │
        └──────────────────────────────────────┘
                   ↓
        Linear projection → vocab logits
        (weights tied with embedding)

    Args:
        vocab_size:  Size of the token vocabulary.
        d_model:     Hidden / embedding dimension.
        n_heads:     Number of attention heads.
        n_layers:    Number of decoder layers.
        d_ff:        Feed-forward intermediate dimension.
        dropout:     Dropout probability.
        max_len:     Maximum caption length (for PE pre-computation).
        pad_idx:     Index of the ``<pad>`` token (used for embedding padding).
        pretrained_embeddings: Optional pre-trained embedding matrix
            ``(vocab_size, d_model)`` to initialise the word embedding.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 52,
        pad_idx: int = 0,
        pretrained_embeddings: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_idx = pad_idx

        # ---- Word Embedding ------------------------------------------------
        self.word_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=pad_idx,
        )
        if pretrained_embeddings is not None:
            self._init_pretrained_embeddings(pretrained_embeddings)

        # ---- Positional Encoding -------------------------------------------
        self.pos_encoding = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_len=max_len,
            dropout=dropout,
        )

        # ---- Embedding scale factor (Vaswani et al.) -----------------------
        self.embed_scale = math.sqrt(d_model)

        # ---- Transformer Decoder Stack -------------------------------------
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,  # post-norm (classic Transformer)
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )

        # ---- Output Projection ---------------------------------------------
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share embedding and output-projection weights.
        # This requires that the embedding dim equals d_model, which is always
        # true in our architecture.
        self.output_projection.weight = self.word_embedding.weight

        # ---- Initialisation ------------------------------------------------
        self._reset_parameters()

        logger.info(
            "CaptionDecoder: vocab=%d, d_model=%d, heads=%d, layers=%d, "
            "d_ff=%d, dropout=%.2f",
            vocab_size, d_model, n_heads, n_layers, d_ff, dropout,
        )

    # ------------------------------------------------------------------ #
    #  Initialisation helpers
    # ------------------------------------------------------------------ #
    def _reset_parameters(self) -> None:
        """Xavier-uniform init for non-embedding parameters."""
        for name, p in self.named_parameters():
            if "word_embedding" in name:
                continue  # embedding is init'd separately / by pre-trained
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _init_pretrained_embeddings(
        self, pretrained: torch.Tensor
    ) -> None:
        """Load a pre-trained embedding matrix into `self.word_embedding`.

        If the pre-trained matrix has a different dimensionality than
        ``d_model``, a linear projection is applied.

        Args:
            pretrained: Tensor of shape ``(vocab_size, embed_dim)``.
        """
        pretrained = pretrained.float()
        src_dim = pretrained.size(1)

        if src_dim == self.d_model:
            self.word_embedding.weight.data.copy_(pretrained)
            logger.info(
                "Loaded pre-trained embeddings directly (dim=%d).", src_dim
            )
        else:
            # Project from src_dim → d_model.
            self._embed_proj = nn.Linear(src_dim, self.d_model, bias=False)
            projected = self._embed_proj(pretrained)
            self.word_embedding.weight.data.copy_(projected.detach())
            logger.info(
                "Projected pre-trained embeddings %d → %d.", src_dim, self.d_model
            )

    # ================================================================== #
    #  Mask helpers
    # ================================================================== #
    @staticmethod
    def _generate_causal_mask(
        seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """Create an upper-triangular causal (look-ahead) mask.

        PyTorch's ``nn.TransformerDecoder`` expects the ``tgt_mask`` to be
        ``(seq_len, seq_len)`` of dtype ``float`` where ``-inf`` means
        *masked*.

        Returns:
            (seq_len, seq_len) float tensor with ``-inf`` above the diagonal.
        """
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()
        return mask.float().masked_fill(mask, float("-inf"))

    @staticmethod
    def _make_pad_mask(
        tokens: torch.Tensor, pad_idx: int
    ) -> torch.Tensor:
        """Create a boolean padding mask.

        Args:
            tokens: (batch, seq_len) token indices.
            pad_idx: The padding token index.

        Returns:
            (batch, seq_len) bool tensor – ``True`` for padded positions.
        """
        return tokens == pad_idx

    # ================================================================== #
    #  Forward (Teacher Forcing)
    # ================================================================== #
    def forward(
        self,
        fused_features: torch.Tensor,
        fused_mask: Optional[torch.Tensor],
        captions: torch.Tensor,
        caption_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forced forward pass for training.

        The decoder receives ``captions[:, :-1]`` as input (shifted right)
        and predicts ``captions[:, 1:]`` as targets, effectively learning to
        predict the *next* token at every position.

        Args:
            fused_features: (batch, N_fused, d_model) – encoder memory.
            fused_mask:     (batch, N_fused) bool – True = padded position
                            in the encoder memory.  ``None`` if no padding.
            captions:       (batch, max_len) – ground-truth caption indices,
                            starting with ``<start>``.
            caption_lengths:(batch,) – true lengths **including** ``<start>``
                            and ``<end>`` tokens.

        Returns:
            logits: (batch, max_len-1, vocab_size) – predictions aligned
                    with ``captions[:, 1:]``.
        """
        # Shift: input = captions without last token, target = without first.
        tgt_input = captions[:, :-1]  # (batch, T-1)  T = max_len
        seq_len = tgt_input.size(1)

        # ---- Masks ----
        # Causal mask: prevents attending to future tokens.
        tgt_mask = self._generate_causal_mask(seq_len, tgt_input.device)

        # Padding mask for the target (decoder input).
        tgt_key_padding_mask = self._make_pad_mask(tgt_input, self.pad_idx)

        # Memory (encoder) key-padding mask.
        memory_key_padding_mask = fused_mask  # may be None

        # ---- Embed & add PE ----
        x = self.word_embedding(tgt_input) * self.embed_scale  # (B, T-1, d)
        x = self.pos_encoding(x)

        # ---- Transformer Decoder ----
        decoder_output = self.transformer_decoder(
            tgt=x,
            memory=fused_features,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )  # (batch, T-1, d_model)

        # ---- Project to vocabulary ----
        logits = self.output_projection(decoder_output)  # (B, T-1, vocab)
        return logits

    # ================================================================== #
    #  Greedy Decode
    # ================================================================== #
    @torch.no_grad()
    def greedy_decode(
        self,
        fused_features: torch.Tensor,
        fused_mask: Optional[torch.Tensor],
        max_len: int = 30,
        start_idx: int = 1,
        end_idx: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Greedy (argmax) decoding — used as the SCST baseline.

        Args:
            fused_features: (batch, N_fused, d_model).
            fused_mask:     (batch, N_fused) bool or None.
            max_len:  Maximum number of tokens to generate.
            start_idx: Index of ``<start>`` token.
            end_idx:   Index of ``<end>`` token.

        Returns:
            captions: (batch, max_len) – generated token indices (0-padded).
            scores:   (batch,) – sum of log-probabilities for each sequence.
        """
        batch_size = fused_features.size(0)
        device = fused_features.device

        # Start with <start> token for every sample.
        generated = torch.full(
            (batch_size, 1), start_idx, dtype=torch.long, device=device
        )
        scores = torch.zeros(batch_size, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for step in range(max_len):
            seq_len = generated.size(1)

            # Masks.
            tgt_mask = self._generate_causal_mask(seq_len, device)
            tgt_key_padding_mask = self._make_pad_mask(generated, self.pad_idx)

            # Embed + PE.
            x = self.word_embedding(generated) * self.embed_scale
            x = self.pos_encoding(x)

            # Decode.
            out = self.transformer_decoder(
                tgt=x,
                memory=fused_features,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=fused_mask,
            )

            # Only care about the last position.
            logits = self.output_projection(out[:, -1, :])  # (B, vocab)
            log_probs = F.log_softmax(logits, dim=-1)

            # Greedy selection.
            next_tokens = log_probs.argmax(dim=-1)  # (B,)
            next_log_probs = log_probs.gather(
                1, next_tokens.unsqueeze(1)
            ).squeeze(1)

            # Accumulate scores for non-finished sequences.
            scores += next_log_probs * (~finished).float()

            # Mark finished.
            finished = finished | (next_tokens == end_idx)

            # Append token.
            generated = torch.cat(
                [generated, next_tokens.unsqueeze(1)], dim=1
            )

            # Early stop if all sequences have ended.
            if finished.all():
                break

        # Pad to max_len + 1 (including <start>) if stopped early.
        if generated.size(1) < max_len + 1:
            pad_len = max_len + 1 - generated.size(1)
            generated = F.pad(generated, (0, pad_len), value=self.pad_idx)

        # Remove the leading <start> token so output is (batch, max_len).
        captions = generated[:, 1:]
        return captions, scores

    # ================================================================== #
    #  Multinomial Sampling (for SCST)
    # ================================================================== #
    def sample(
        self,
        fused_features: torch.Tensor,
        fused_mask: Optional[torch.Tensor],
        max_len: int = 30,
        start_idx: int = 1,
        end_idx: int = 2,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample captions via multinomial sampling (for SCST training).

        Unlike greedy or beam search, this method samples from the predicted
        distribution at each time step, which is needed for the REINFORCE
        policy gradient estimator.

        Args:
            fused_features: (batch, N_fused, d_model).
            fused_mask:     (batch, N_fused) bool or None.
            max_len:   Maximum tokens to generate.
            start_idx: ``<start>`` token index.
            end_idx:   ``<end>`` token index.
            temperature: Softmax temperature (>1 = more random).

        Returns:
            sampled_ids: (batch, max_len) – sampled token indices.
            log_probs:   (batch, max_len) – per-step log-probabilities of the
                         sampled tokens (masked to 0 after ``<end>``).
        """
        batch_size = fused_features.size(0)
        device = fused_features.device

        generated = torch.full(
            (batch_size, 1), start_idx, dtype=torch.long, device=device
        )
        all_log_probs: list[torch.Tensor] = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for step in range(max_len):
            seq_len = generated.size(1)

            tgt_mask = self._generate_causal_mask(seq_len, device)
            tgt_key_padding_mask = self._make_pad_mask(generated, self.pad_idx)

            x = self.word_embedding(generated) * self.embed_scale
            x = self.pos_encoding(x)

            out = self.transformer_decoder(
                tgt=x,
                memory=fused_features,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=fused_mask,
            )

            logits = self.output_projection(out[:, -1, :])  # (B, vocab)

            # Apply temperature.
            if temperature != 1.0:
                logits = logits / temperature

            probs = F.softmax(logits, dim=-1)
            log_prob_dist = F.log_softmax(logits, dim=-1)

            # Sample from the distribution.
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            step_log_probs = log_prob_dist.gather(
                1, next_tokens.unsqueeze(1)
            ).squeeze(1)

            # Zero-out log-probs for already-finished sequences.
            step_log_probs = step_log_probs * (~finished).float()
            all_log_probs.append(step_log_probs)

            finished = finished | (next_tokens == end_idx)
            generated = torch.cat(
                [generated, next_tokens.unsqueeze(1)], dim=1
            )

            if finished.all():
                break

        # Stack log-probs → (batch, actual_steps).
        log_probs_tensor = torch.stack(all_log_probs, dim=1)

        # Pad to max_len if early-stopped.
        if log_probs_tensor.size(1) < max_len:
            pad_len = max_len - log_probs_tensor.size(1)
            log_probs_tensor = F.pad(log_probs_tensor, (0, pad_len), value=0.0)

        # Remove <start> from generated; pad to max_len.
        sampled = generated[:, 1:]
        if sampled.size(1) < max_len:
            pad_len = max_len - sampled.size(1)
            sampled = F.pad(sampled, (0, pad_len), value=self.pad_idx)

        return sampled, log_probs_tensor

    # ================================================================== #
    #  Beam Search
    # ================================================================== #
    @torch.no_grad()
    def generate(
        self,
        fused_features: torch.Tensor,
        fused_mask: Optional[torch.Tensor],
        max_len: int = 30,
        beam_size: int = 5,
        start_idx: int = 1,
        end_idx: int = 2,
        length_penalty: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Beam search decoding for inference.

        Processes each sample in the batch independently because beam search
        expands to ``beam_size`` hypotheses per sample.

        Length normalisation follows Wu et al. (2016):
            score = log_prob / ((5 + length) / 6) ^ alpha

        Args:
            fused_features: (batch, N_fused, d_model).
            fused_mask:     (batch, N_fused) bool or None.
            max_len:        Maximum number of tokens to generate.
            beam_size:      Beam width.
            start_idx:      ``<start>`` token index.
            end_idx:        ``<end>`` token index.
            length_penalty: Length-normalisation exponent (α).

        Returns:
            captions: (batch, max_len) – best caption for each sample.
            scores:   (batch,) – normalised log-probability scores.
        """
        batch_size = fused_features.size(0)
        device = fused_features.device

        all_captions: list[torch.Tensor] = []
        all_scores: list[float] = []

        for b in range(batch_size):
            # Extract single-sample encoder outputs.
            memory = fused_features[b].unsqueeze(0)  # (1, N_fused, d)
            mem_mask = (
                fused_mask[b].unsqueeze(0) if fused_mask is not None else None
            )

            best_seq, best_score = self._beam_search_single(
                memory=memory,
                memory_mask=mem_mask,
                max_len=max_len,
                beam_size=beam_size,
                start_idx=start_idx,
                end_idx=end_idx,
                length_penalty=length_penalty,
            )
            all_captions.append(best_seq)
            all_scores.append(best_score)

        # Pad / stack into batch tensors.
        captions = torch.zeros(
            batch_size, max_len, dtype=torch.long, device=device
        )
        for b, seq in enumerate(all_captions):
            length = min(len(seq), max_len)
            captions[b, :length] = seq[:length]

        scores = torch.tensor(all_scores, device=device)
        return captions, scores

    def _beam_search_single(
        self,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor],
        max_len: int,
        beam_size: int,
        start_idx: int,
        end_idx: int,
        length_penalty: float,
    ) -> Tuple[torch.Tensor, float]:
        """Run beam search for a single sample.

        Args:
            memory:      (1, N_fused, d_model).
            memory_mask: (1, N_fused) bool or None.

        Returns:
            best_seq:   1-D LongTensor of token indices (without <start>).
            best_score: Normalised log-probability.
        """
        device = memory.device

        # Expand memory for beam_size hypotheses.
        # (beam_size, N_fused, d_model)
        memory_exp = memory.expand(beam_size, -1, -1)
        mem_mask_exp = (
            memory_mask.expand(beam_size, -1)
            if memory_mask is not None
            else None
        )

        # Each hypothesis: (tokens, cumulative_log_prob)
        # Start with just the <start> token.
        sequences = torch.full(
            (beam_size, 1), start_idx, dtype=torch.long, device=device
        )
        scores = torch.zeros(beam_size, device=device)

        # Track which beams are active (not yet produced <end>).
        active_mask = torch.ones(beam_size, dtype=torch.bool, device=device)

        # Completed hypotheses: list of (sequence_tensor, normalised_score).
        completed: list[Tuple[torch.Tensor, float]] = []

        for step in range(max_len):
            if not active_mask.any():
                break

            n_active = active_mask.sum().item()
            active_idx = active_mask.nonzero(as_tuple=False).squeeze(-1)

            # Gather active beams.
            active_seqs = sequences[active_idx]   # (n_active, cur_len)
            active_scores = scores[active_idx]     # (n_active,)
            active_memory = memory_exp[active_idx]
            active_mem_mask = (
                mem_mask_exp[active_idx] if mem_mask_exp is not None else None
            )

            cur_len = active_seqs.size(1)
            tgt_mask = self._generate_causal_mask(cur_len, device)
            tgt_pad_mask = self._make_pad_mask(active_seqs, self.pad_idx)

            x = self.word_embedding(active_seqs) * self.embed_scale
            x = self.pos_encoding(x)

            out = self.transformer_decoder(
                tgt=x,
                memory=active_memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_pad_mask,
                memory_key_padding_mask=active_mem_mask,
            )

            logits = self.output_projection(out[:, -1, :])  # (n_active, V)
            log_probs = F.log_softmax(logits, dim=-1)       # (n_active, V)

            vocab_size = log_probs.size(-1)

            # Candidate scores: (n_active, V)
            candidate_scores = active_scores.unsqueeze(1) + log_probs

            if step == 0 and n_active == beam_size:
                # On the first step all beams are identical; only expand one.
                flat_scores = candidate_scores[0]  # (V,)
                top_scores, top_indices = flat_scores.topk(beam_size)
                beam_indices = torch.zeros(
                    beam_size, dtype=torch.long, device=device
                )
                token_indices = top_indices
            else:
                # Flatten and pick top-k across all active beams.
                flat_scores = candidate_scores.view(-1)  # (n_active * V,)
                top_scores, top_flat_indices = flat_scores.topk(
                    min(beam_size, flat_scores.size(0))
                )
                beam_indices = top_flat_indices // vocab_size
                token_indices = top_flat_indices % vocab_size

            # Build new sequences.
            new_seqs_list: list[torch.Tensor] = []
            new_scores_list: list[float] = []
            new_active: list[bool] = []

            for i in range(top_scores.size(0)):
                bi = beam_indices[i].item()
                ti = token_indices[i].item()
                sc = top_scores[i].item()

                prev_seq = active_seqs[bi]
                new_seq = torch.cat(
                    [prev_seq, torch.tensor([ti], device=device)]
                )

                if ti == end_idx:
                    # Normalise score by length.
                    seq_len = new_seq.size(0) - 1  # exclude <start>
                    norm_score = sc / ((5.0 + seq_len) / 6.0) ** length_penalty
                    completed.append((new_seq[1:], norm_score))  # drop <start>
                    new_active.append(False)
                else:
                    new_active.append(True)

                new_seqs_list.append(new_seq)
                new_scores_list.append(sc)

            # If we have enough completed hypotheses, we can stop.
            if len(completed) >= beam_size:
                break

            # Rebuild beam tensors.
            # Pad all sequences to the same length.
            max_seq_len = max(s.size(0) for s in new_seqs_list)
            new_sequences = torch.full(
                (beam_size, max_seq_len),
                self.pad_idx,
                dtype=torch.long,
                device=device,
            )
            new_scores_tensor = torch.full(
                (beam_size,), float("-inf"), device=device
            )
            new_active_mask = torch.zeros(
                beam_size, dtype=torch.bool, device=device
            )

            count = 0
            for i in range(len(new_seqs_list)):
                if new_active[i] and count < beam_size:
                    slen = new_seqs_list[i].size(0)
                    new_sequences[count, :slen] = new_seqs_list[i]
                    new_scores_tensor[count] = new_scores_list[i]
                    new_active_mask[count] = True
                    count += 1

            sequences = new_sequences
            scores = new_scores_tensor
            active_mask = new_active_mask

            # Expand memory to match current beam size.
            memory_exp = memory.expand(beam_size, -1, -1)
            mem_mask_exp = (
                memory_mask.expand(beam_size, -1)
                if memory_mask is not None
                else None
            )

        # If no hypothesis completed, take the best active beam.
        if not completed:
            best_idx = scores.argmax().item()
            best_seq = sequences[best_idx, 1:]  # drop <start>
            # Remove trailing padding.
            non_pad = (best_seq != self.pad_idx).nonzero(as_tuple=False)
            if non_pad.numel() > 0:
                best_seq = best_seq[: non_pad[-1].item() + 1]
            seq_len = best_seq.size(0)
            best_score = scores[best_idx].item() / (
                (5.0 + seq_len) / 6.0
            ) ** length_penalty
        else:
            # Pick the completed hypothesis with the highest normalised score.
            completed.sort(key=lambda x: x[1], reverse=True)
            best_seq, best_score = completed[0]

        return best_seq, best_score
