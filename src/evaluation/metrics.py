"""Các độ đo đánh giá chất lượng câu chú thích ảnh sinh ra (Caption Evaluation Metrics).

Triển khai tính toán các độ đo:
1. BLEU-1, -2, -3, -4: Độ chính xác n-gram (bắt cặp từ đơn/từ ghép).
2. METEOR: Đánh giá dựa trên ngữ nghĩa từ (đồng nghĩa, gốc từ, từ điển WordNet).
3. ROUGE-L: Độ tương đồng chuỗi con chung dài nhất (LCS).
4. CIDEr: Độ đo dựa trên trọng số TF-IDF n-gram dành riêng cho mô tả ảnh.
5. SPICE: Đánh giá ngữ nghĩa dựa trên đồ thị quan hệ trong câu.

Có hỗ trợ hàm dự phòng tự tính BLEU bằng thư viện NLTK nếu không gọi được thư viện COCO.
"""

from __future__ import annotations

import sys
from typing import Dict, List, Any


def compute_metrics(
    predictions: Dict[Any, List[str]], 
    references: Dict[Any, List[str]]
) -> Dict[str, float]:
    """Tính toán toàn bộ các chỉ số đánh giá chất lượng chú thích ảnh.

    Định dạng đầu vào:
        predictions: {image_id: [câu_dự_đoán]} - mỗi ảnh có 1 câu dự đoán.
        references: {image_id: [câu_mẫu_1, câu_mẫu_2, ...]} - mỗi ảnh có nhiều câu tham chiếu.

    Args:
        predictions: Dictionary chứa các câu mô tả mô hình tự sinh ra.
        references: Dictionary chứa các câu mô tả chuẩn của con người viết.

    Returns:
        Dictionary chứa kết quả các độ đo (nhân với 100 để hiển thị dạng phần trăm).
    """
    # Làm sạch khóa: đồng bộ khóa thành string cho pycocoevalcap hoạt động ổn định
    gts = {str(k): [{"caption": c} for c in v] for k, v in references.items()}
    res = {str(k): [{"caption": c[0]} if isinstance(c, list) else {"caption": c}] for k, c in predictions.items()}

    # Lọc lại để chỉ đánh giá trên tập giao của các image_id
    common_ids = set(gts.keys()).intersection(set(res.keys()))
    gts = {k: gts[k] for k in common_ids}
    res = {k: res[k] for k in common_ids}

    if not common_ids:
        print("Lỗi: Không tìm thấy image_id chung nào giữa predictions và references để đánh giá!")
        return {}

    metrics_results = {}

    try:
        # Nhập các lớp đánh giá từ thư viện pycocoevalcap
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.cider.cider import Cider
        
        # Thử import Spice (yêu cầu Java cài sẵn trên máy)
        try:
            from pycocoevalcap.spice.spice import Spice
            has_spice = True
        except (ImportError, Exception):
            has_spice = False
            print("Cảnh báo: Không thể nạp mô-đun SPICE (có thể thiếu Java). Bỏ qua SPICE.")

        # Tiến hành token hóa gts và res trước khi đưa vào các bộ tính điểm
        try:
            from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
            tokenizer = PTBTokenizer()
            gts = tokenizer.tokenize(gts)
            res = tokenizer.tokenize(res)
        except Exception as e:
            print(f"Cảnh báo: Không thể khởi chạy PTBTokenizer (có thể thiếu Java). Sử dụng token hóa đơn giản. Chi tiết: {e}")
            # Dự phòng: Chuyển đổi định dạng từ list-of-dicts sang list-of-strings đơn giản
            gts = {k: [str(item['caption']).lower() for item in v] for k, v in gts.items()}
            res = {k: [str(item['caption']).lower() for item in v] for k, v in res.items()}

        # 1. Tính toán BLEU-1, BLEU-2, BLEU-3, BLEU-4
        bleu_scorer = Bleu(4)
        bleu_score, _ = bleu_scorer.compute_score(gts, res)
        for i in range(4):
            metrics_results[f"BLEU-{i+1}"] = float(bleu_score[i]) * 100.0

        # 2. Tính toán METEOR
        try:
            meteor_scorer = Meteor()
            meteor_score, _ = meteor_scorer.compute_score(gts, res)
            metrics_results["METEOR"] = float(meteor_score) * 100.0
        except Exception as e:
            print(f"Cảnh báo: Lỗi khi tính METEOR ({str(e)}). Gán giá trị 0.0.")
            metrics_results["METEOR"] = 0.0

        # 3. Tính toán ROUGE-L
        rouge_scorer = Rouge()
        rouge_score, _ = rouge_scorer.compute_score(gts, res)
        metrics_results["ROUGE-L"] = float(rouge_score) * 100.0

        # 4. Tính toán CIDEr
        cider_scorer = Cider()
        cider_score, _ = cider_scorer.compute_score(gts, res)
        metrics_results["CIDEr"] = float(cider_score) * 100.0

        # 5. Tính toán SPICE
        if has_spice:
            try:
                spice_scorer = Spice()
                spice_score, _ = spice_scorer.compute_score(gts, res)
                metrics_results["SPICE"] = float(spice_score) * 100.0
            except Exception as e:
                print(f"Cảnh báo: Lỗi khi tính SPICE ({str(e)}). Gán giá trị 0.0.")
                metrics_results["SPICE"] = 0.0
        else:
            metrics_results["SPICE"] = 0.0

    except ImportError:
        print("Không tìm thấy pycocoevalcap. Chuyển sang chế độ tính toán dự phòng bằng NLTK...")
        metrics_results = compute_fallback_nltk_metrics(predictions, references)

    return metrics_results


def compute_fallback_nltk_metrics(
    predictions: Dict[Any, List[str]], 
    references: Dict[Any, List[str]]
) -> Dict[str, float]:
    """Hàm đánh giá dự phòng sử dụng thư viện NLTK chỉ để tính BLEU.

    Được gọi khi hệ thống không cài đặt được thư viện pycocoevalcap gốc.

    Args:
        predictions: Các mô tả tự động từ mô hình.
        references: Các mô tả chuẩn từ con người.

    Returns:
        Dict chứa BLEU-1/2/3/4 tự tính.
    """
    import nltk
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    
    # NLTK yêu cầu token hóa các câu thành danh sách từ
    refs_tokenized = []
    preds_tokenized = []
    
    for img_id, pred_caps in predictions.items():
        if img_id not in references:
            continue
            
        pred_sentence = pred_caps[0] if isinstance(pred_caps, list) else pred_caps
        ref_sentences = references[img_id]
        
        # Tokenize từ thô sang list từ bằng split đơn giản
        pred_tokens = str(pred_sentence).lower().split()
        ref_tokens_list = [str(r).lower().split() for r in ref_sentences]
        
        preds_tokenized.append(pred_tokens)
        refs_tokenized.append(ref_tokens_list)
        
    if not preds_tokenized:
        return {}
        
    chencherry = SmoothingFunction()
    
    bleu1 = corpus_bleu(refs_tokenized, preds_tokenized, weights=(1, 0, 0, 0), smoothing_function=chencherry.method1)
    bleu2 = corpus_bleu(refs_tokenized, preds_tokenized, weights=(0.5, 0.5, 0, 0), smoothing_function=chencherry.method1)
    bleu3 = corpus_bleu(refs_tokenized, preds_tokenized, weights=(0.33, 0.33, 0.33, 0), smoothing_function=chencherry.method1)
    bleu4 = corpus_bleu(refs_tokenized, preds_tokenized, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method1)
    
    # Tính ROUGE-L đơn giản (bằng tay) làm đại diện nếu không có coco-caption
    rouge_l_scores = []
    for pred, refs in zip(preds_tokenized, refs_tokenized):
        best_f1 = 0.0
        for ref in refs:
            f1 = compute_sentence_lcs_f1(pred, ref)
            best_f1 = max(best_f1, f1)
        rouge_l_scores.append(best_f1)
    avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0.0
    
    return {
        "BLEU-1": float(bleu1) * 100.0,
        "BLEU-2": float(bleu2) * 100.0,
        "BLEU-3": float(bleu3) * 100.0,
        "BLEU-4": float(bleu4) * 100.0,
        "ROUGE-L": float(avg_rouge_l) * 100.0,
        "CIDEr": 0.0,  # Không thể tự tính toán CIDEr dễ dàng
        "METEOR": 0.0,
        "SPICE": 0.0
    }


def compute_sentence_lcs_f1(x: List[str], y: List[str]) -> float:
    """Tính điểm F1 dựa trên Chuỗi con chung dài nhất (LCS) giữa hai câu.

    Args:
        x: Token danh sách từ câu dự đoán.
        y: Token danh sách từ câu tham chiếu.

    Returns:
        Điểm F1-LCS (0.0 đến 1.0).
    """
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return 0.0
        
    # Tạo bảng quy hoạch động tính độ dài LCS
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if x[i-1] == y[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                
    lcs_len = dp[n][m]
    
    # Tính Precision, Recall và F1
    precision = lcs_len / n
    recall = lcs_len / m
    
    if precision + recall == 0.0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)
