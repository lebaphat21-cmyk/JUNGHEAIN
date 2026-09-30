"""Gói đánh giá chất lượng chú thích ảnh sinh ra (Evaluation Module).

Cung cấp các công cụ để tính toán các độ đo chất lượng phổ biến cho bài toán chú thích ảnh
(BLEU-1/2/3/4, METEOR, ROUGE-L, CIDEr, SPICE) thông qua so sánh câu sinh ra từ mô hình
và các câu chú thích tham chiếu do con người viết.
"""

from src.evaluation.metrics import compute_metrics
from src.evaluation.evaluator import CaptionEvaluator

__all__ = [
    "compute_metrics",
    "CaptionEvaluator",
]
