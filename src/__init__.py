"""
NCKH Image Captioning System
=============================

A Transformer-based Image Captioning system that fuses Visual features
(from ResNet-101) and Semantic features (from Scene Graph triples) using
multiple fusion strategies.

Fusion Strategies:
    - Concatenation
    - Cross-Attention
    - Gated Fusion
    - Co-Attention

Architecture:
    Visual Backbone (ResNet-101) ──→ Visual Features (batch, 49, 512)
                                          ↓
                                    ┌─────────────┐
    Scene Graph Triples ──→         │   Fusion     │ ──→ Fused Features ──→ Decoder ──→ Caption
    Semantic Features (batch,20,512)│   Module     │
                                    └─────────────┘

Dataset: MS COCO Captions (Karpathy split)
Training: Two-stage (Cross-Entropy → SCST with CIDEr reward)
Evaluation: BLEU, METEOR, ROUGE-L, CIDEr, SPICE
"""

__version__ = "1.0.0"
__author__ = "NCKH Research Team"
__description__ = "Image Captioning with Visual-Semantic Fusion via Transformers"

# Shared constants used across all modules
SHARED_CONSTANTS = {
    "d_model": 512,
    "n_heads": 8,
    "N_v": 49,           # Visual feature count (7x7 grid)
    "N_s": 20,           # Max semantic triples
    "max_caption_len": 30,
    "pad_token_id": 0,
    "start_token_id": 1,
    "end_token_id": 2,
    "unk_token_id": 3,
}
