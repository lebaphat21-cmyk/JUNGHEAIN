"""Model modules for NCKH Image Captioning.

Components:
    - VisualEncoder: Extracts spatial visual features from images via ResNet.
    - SemanticEncoder: Encodes scene graph triples into semantic features.
    - Fusion modules: Concatenation, Cross-Attention, Gated, Co-Attention.
    - CaptionDecoder: Transformer decoder with beam search for caption generation.
    - ImageCaptioningModel: Full end-to-end model assembly.
"""
