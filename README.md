# 🖼️ Image Captioning with Visual-Semantic Fusion via Transformers

## Mô tả tổng quan / Project Overview

**Tiếng Việt:**
Hệ thống sinh chú thích ảnh tự động sử dụng kiến trúc Transformer encoder-decoder, kết hợp đặc trưng thị giác (Visual features) từ ResNet-101 và đặc trưng ngữ nghĩa (Semantic features) từ Scene Graph triples. Dự án nghiên cứu khoa học (NCKH) so sánh 4 chiến lược kết hợp đặc trưng: Concatenation, Cross-Attention, Gated Fusion, và Co-Attention.

**English:**
An automatic image captioning system using a Transformer encoder-decoder architecture that fuses Visual features (from ResNet-101) and Semantic features (from Scene Graph triples). This research project (NCKH) compares four feature fusion strategies: Concatenation, Cross-Attention, Gated Fusion, and Co-Attention.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT IMAGE                                  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
  ┌───────────────────┐         ┌───────────────────────┐
  │   ResNet-101 CNN  │         │  Scene Graph Generator │
  │   (Backbone)      │         │  (Triple Extraction)   │
  └────────┬──────────┘         └───────────┬───────────┘
           │                                │
           ▼                                ▼
  ┌───────────────────┐         ┌───────────────────────┐
  │  Linear Projection│         │  Triple Encoder       │
  │  2048 → 512       │         │  (GloVe + MLP)        │
  └────────┬──────────┘         │  300 → 512            │
           │                    └───────────┬───────────┘
           ▼                                ▼
  Visual Features               Semantic Features
  (batch, 49, 512)              (batch, 20, 512)
           │                                │
           └──────────┬─────────────────────┘
                      ▼
       ┌──────────────────────────┐
       │     FUSION MODULE        │
       │  ┌────────────────────┐  │
       │  │ • Concatenation    │  │
       │  │ • Cross-Attention  │  │
       │  │ • Gated Fusion     │  │
       │  │ • Co-Attention     │  │
       │  └────────────────────┘  │
       └────────────┬─────────────┘
                    ▼
           Fused Features
           (batch, N_fused, 512)
                    │
                    ▼
       ┌────────────────────────┐
       │  Transformer Encoder   │
       │  (3 layers, 8 heads)   │
       └────────────┬───────────┘
                    │
                    ▼
           Encoder Memory
                    │
                    ▼
       ┌────────────────────────┐        ┌──────────────┐
       │  Transformer Decoder   │◄───────│  Captions    │
       │  (3 layers, 8 heads)   │        │  (shifted)   │
       └────────────┬───────────┘        └──────────────┘
                    │
                    ▼
       ┌────────────────────────┐
       │  Output Linear + Softmax│
       │  512 → vocab_size      │
       └────────────┬───────────┘
                    │
                    ▼
            Generated Caption
    "A dog playing with a ball in the park"
```

---

## 🔬 Fusion Strategies

| Strategy | Description | N_fused |
|----------|-------------|---------|
| **Concatenation** | Concatenate visual and semantic features along the sequence dimension, then project | N_v + N_s = 69 |
| **Cross-Attention** | Visual features attend to semantic features (and vice versa) via multi-head cross-attention | N_v = 49 |
| **Gated Fusion** | Learnable sigmoid gate controls the blend of visual and semantic information | N_v = 49 |
| **Co-Attention** | Parallel attention streams with shared interactions between modalities | N_v = 49 |

---

## 📦 Installation

### Prerequisites
- Python 3.9+
- CUDA 11.8+ (recommended for GPU training)
- Git

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd NCKH

# Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (required for evaluation)
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### Data Preparation

1. **MS COCO Images**: Download from [COCO Dataset](https://cocodataset.org/)
2. **Karpathy Split**: Download from [Karpathy's page](https://cs.stanford.edu/people/karpathy/deepimagesent/)
3. **GloVe Embeddings**: Download from [GloVe](https://nlp.stanford.edu/projects/glove/)

```bash
# Expected data structure:
data/
├── coco/
│   ├── images/
│   │   ├── train2014/
│   │   └── val2014/
│   ├── dataset_coco.json       # Karpathy split
│   ├── scene_graphs/           # Pre-extracted scene graphs
│   └── visual_features/        # Pre-extracted ResNet features (optional)
└── glove/
    └── glove.6B.300d.txt
```

---

## 🚀 Usage

### Training

```bash
# Stage 1: Cross-Entropy Training
python train.py --config configs/base_config.yaml \
    --set fusion.type=cross_attention

# Stage 2: SCST Fine-tuning (automatically follows Stage 1)
python train.py --config configs/base_config.yaml \
    --set training.stage2.enabled=true

# Override specific parameters
python train.py --config configs/base_config.yaml \
    --set model.d_model=256 training.stage1.lr=5e-5
```

### Evaluation

```bash
# Evaluate a trained model
python evaluate.py --config configs/base_config.yaml \
    --checkpoint outputs/checkpoints/best_model.pth

# Evaluate with specific metrics
python evaluate.py --checkpoint outputs/checkpoints/best_model.pth \
    --metrics bleu meteor cider
```

### Inference

```bash
# Generate captions for images
python inference.py --image path/to/image.jpg \
    --checkpoint outputs/checkpoints/best_model.pth \
    --beam_size 5
```

### 🎬 Web Demo (Video Captioning & Text-to-Speech)

```bash
# Khởi chạy giao diện Web Demo
python web_demo.py --checkpoint outputs/checkpoints/best_model.pth

# Với các tuỳ chọn bổ sung
python web_demo.py --checkpoint outputs/checkpoints/best_model.pth \
    --sg_method heuristic \
    --gpu 0 \
    --port 7860 \
    --share  # Tạo liên kết chia sẻ công khai
```

Truy cập giao diện tại: `http://127.0.0.1:7860`

**Tính năng:**
- 🎥 **Video Captioning**: Tải video lên → Tự động trích xuất khung hình → Sinh chú thích bằng Temporal Average Pooling + Transformer.
- 🖼️ **Image Captioning**: Giữ nguyên tính năng gốc, tải ảnh lên và sinh chú thích.
- 🔊 **Text-to-Speech**: Đọc kết quả chú thích bằng giọng nói tự nhiên (gTTS online / pyttsx3 offline).
- 🔗 **Scene Graph Visualization**: Hiển thị các mối quan hệ ngữ cảnh trích xuất được từ keyframe.

---

## 📂 Project Structure

```
NCKH/
├── configs/
│   └── base_config.yaml          # Base configuration
│
├── data/                          # Dataset directory (not tracked by git)
│   ├── coco/
│   └── glove/
│
├── src/
│   ├── __init__.py                # Package init with shared constants
│   │
│   ├── data/                      # Data loading & preprocessing
│   │   ├── __init__.py
│   │   ├── dataset.py             # COCO caption dataset
│   │   ├── vocabulary.py          # Vocabulary builder
│   │   ├── transforms.py          # Image transforms
│   │   └── scene_graph.py         # Scene graph processing
│   │
│   ├── models/                    # Model components
│   │   ├── __init__.py
│   │   ├── encoder.py             # Transformer encoder
│   │   ├── decoder.py             # Transformer decoder
│   │   ├── visual_extractor.py    # ResNet-101 visual features
│   │   ├── semantic_extractor.py  # Scene graph semantic features
│   │   ├── captioning_model.py    # Full captioning model
│   │   └── fusion/                # Fusion strategies
│   │       ├── __init__.py
│   │       ├── base.py            # Abstract fusion base class
│   │       ├── concat.py          # Concatenation fusion
│   │       ├── cross_attention.py # Cross-Attention fusion
│   │       ├── gated.py           # Gated fusion
│   │       └── co_attention.py    # Co-Attention fusion
│   │
│   ├── training/                  # Training logic
│   │   ├── __init__.py
│   │   ├── trainer.py             # Main training loop
│   │   ├── scst.py                # Self-Critical Sequence Training
│   │   ├── loss.py                # Loss functions
│   │   └── scheduler.py           # Learning rate schedulers
│   │
│   ├── evaluation/                # Evaluation metrics
│   │   ├── __init__.py
│   │   ├── evaluator.py           # Evaluation pipeline
│   │   └── metrics.py             # Metric wrappers
│   │
│   └── utils/                     # Utilities
│       ├── __init__.py
│       ├── config.py              # Configuration management
│       ├── logger.py              # Logging utilities
│       ├── misc.py                # Miscellaneous helpers
│       └── video_processor.py     # Video frame extraction & TTS
│
├── outputs/                       # Training outputs (not tracked by git)
│   ├── checkpoints/
│   ├── logs/
│   ├── results/
│   └── tensorboard/
│
├── scripts/                       # Helper scripts
│   ├── preprocess.py              # Data preprocessing
│   ├── extract_features.py        # Feature extraction
│   └── visualize.py               # Result visualization
│
├── train.py                       # Training entry point
├── evaluate.py                    # Evaluation entry point
├── inference.py                   # Inference entry point
├── web_demo.py                    # 🆕 Web Demo (Video Captioning & TTS)
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 📊 Training Pipeline

```
┌─────────────────────┐     ┌──────────────────────┐
│  Stage 1: XE Loss   │────▶│  Stage 2: SCST       │
│                     │     │                      │
│  • LR: 1e-4        │     │  • LR: 1e-5          │
│  • Epochs: 30      │     │  • Epochs: 20        │
│  • Label Smoothing  │     │  • CIDEr Reward      │
│  • Cosine Scheduler │     │  • REINFORCE          │
│  • Early Stopping   │     │  • Greedy Baseline    │
└─────────────────────┘     └──────────────────────┘
```

---

## 📈 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| BLEU-1/2/3/4 | N-gram precision with brevity penalty |
| METEOR | Harmonic mean of precision and recall with synonym matching |
| ROUGE-L | Longest common subsequence F-measure |
| CIDEr-D | Consensus-based image description evaluation using TF-IDF |
| SPICE | Semantic propositional image caption evaluation |

---

## ⚙️ Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| d_model | 512 | Hidden dimension |
| n_heads | 8 | Attention heads |
| Encoder layers | 3 | Transformer encoder depth |
| Decoder layers | 3 | Transformer decoder depth |
| Visual features | 49 (7×7) | ResNet-101 grid features |
| Semantic features | 20 | Max scene graph triples |
| Max caption length | 30 | Maximum output tokens |
| Batch size | 64 | Training batch size |
| Beam size | 5 | Beam search width |

---

## 📝 Citation

```bibtex
@article{nckh2025imagecaptioning,
    title={Image Captioning with Visual-Semantic Fusion via Transformers},
    author={NCKH Research Team},
    year={2025},
    note={Undergraduate Research Project}
}
```

---

## 📄 License

This project is developed for academic research purposes as part of an undergraduate research program (NCKH).

---

## 🙏 Acknowledgments

- [MS COCO Dataset](https://cocodataset.org/) for image captioning benchmarks
- [Karpathy Split](https://cs.stanford.edu/people/karpathy/deepimagesent/) for standard train/val/test splits
- [GloVe](https://nlp.stanford.edu/projects/glove/) for pre-trained word embeddings
- [pycocoevalcap](https://github.com/salaniz/pycocoevalcap) for evaluation metrics
