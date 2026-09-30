"""Bộ chạy đánh giá mô hình chú thích ảnh (Caption Evaluator).

Lớp CaptionEvaluator chịu trách nhiệm quản lý quy trình chạy mô hình qua tập dữ liệu kiểm thử
(Validation hoặc Test set), thực hiện giải mã chùm (beam search) để sinh câu mô tả,
chuyển đổi từ dạng chỉ số số nguyên về dạng câu chữ tự nhiên và tính toán các độ đo chất lượng.
"""

from __future__ import annotations

import os
import json
from typing import Dict, List, Tuple, Any, Optional
import torch
from tqdm import tqdm

from src.evaluation.metrics import compute_metrics


class CaptionEvaluator:
    """Quy trình đánh giá chất lượng chú thích ảnh tích hợp."""

    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        vocab: Any,
        device: torch.device,
        beam_size: int = 5,
        max_len: int = 30,
    ) -> None:
        """Khởi tạo bộ đánh giá.

        Args:
            model: Mô hình ImageCaptioningModel cần đánh giá.
            dataloader: DataLoader của tập kiểm thử.
            vocab: Bộ từ điển (Vocabulary) của dự án.
            device: Thiết bị chạy tính toán ('cuda' hoặc 'cpu').
            beam_size: Độ rộng chùm trong thuật toán Beam Search.
            max_len: Chiều dài câu chú thích tối đa.
        """
        self.model = model.to(device)
        self.dataloader = dataloader
        self.vocab = vocab
        self.device = device
        self.beam_size = beam_size
        self.max_len = max_len

    def evaluate(self) -> Tuple[Dict[str, float], Dict[int, str]]:
        """Chạy đánh giá mô hình trên toàn bộ tập dữ liệu của dataloader.

        Returns:
            Tuple chứa:
                - **metrics**: Dictionary kết quả các độ đo (BLEU, CIDEr, v.v.).
                - **predictions**: Dictionary ánh xạ từ image_id (int) sang câu dự đoán (str).
        """
        self.model.eval()
        predictions = {}
        references = {}

        print(f"Đang sinh chú thích ảnh bằng thuật toán Beam Search (size={self.beam_size})...")

        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="Giải mã sinh câu"):
                # Giải nén batch dữ liệu từ COCO DataLoader
                images = batch["image"].to(self.device)
                triples = batch["triples"].to(self.device)
                triple_mask = batch["triple_mask"].to(self.device)
                image_ids = batch["image_id"]
                raw_captions = batch["raw_captions"]  # Danh sách các câu mô tả chuẩn dạng text

                # Sinh chú thích từ mô hình bằng Beam Search
                # caps_pred: (B, max_len)
                caps_pred, _ = self.model.generate(
                    images=images,
                    triples=triples,
                    triple_mask=triple_mask,
                    max_len=self.max_len,
                    beam_size=self.beam_size,
                    start_idx=self.vocab.word2idx["<start>"],
                    end_idx=self.vocab.word2idx["<end>"],
                )

                # Chuyển đổi chuỗi chỉ số token sang văn bản
                for idx, img_id in enumerate(image_ids):
                    # Chuyển đổi khóa về kiểu int hoặc giữ nguyên
                    img_key = int(img_id.item()) if isinstance(img_id, torch.Tensor) else img_id
                    
                    # Lấy chuỗi chỉ số của ảnh hiện tại và giải mã
                    pred_token_ids = caps_pred[idx].cpu().tolist()
                    pred_caption = self.vocab.decode(pred_token_ids)
                    
                    # Lưu lại câu dự đoán (lấy câu đầu tiên duy nhất)
                    predictions[img_key] = [pred_caption]
                    
                    # Lưu lại danh sách các câu chú thích tham chiếu chuẩn của ảnh đó
                    references[img_key] = raw_captions[idx]

        # Tính toán toàn bộ các chỉ số đánh giá bằng pycocoevalcap
        metrics = compute_metrics(predictions, references)

        # Chuyển đổi định dạng predictions từ list của 1 phần tử về string thô cho gọn
        flat_predictions = {k: v[0] for k, v in predictions.items()}
        
        return metrics, flat_predictions

    def evaluate_and_save(self, output_dir: str, split_name: str = "test") -> Dict[str, float]:
        """Chạy đánh giá và tự động lưu trữ các file kết quả chi tiết.

        Tạo ra:
        1. File JSON chứa toàn bộ các câu dự đoán: `predictions_[split].json`
        2. File JSON chứa kết quả độ đo tổng hợp: `metrics_[split].json`
        3. File text lưu các mẫu ví dụ để phân tích định tính: `qualitative_[split].txt`

        Args:
            output_dir: Thư mục đích để lưu trữ kết quả.
            split_name: Tên của tập dữ liệu đang chạy (ví dụ: 'val', 'test').

        Returns:
            Dictionary chứa kết quả các độ đo.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        metrics, predictions = self.evaluate()
        
        # In bảng kết quả đẹp ra màn hình
        self.print_results(metrics)
        
        # 1. Lưu file predictions
        pred_path = os.path.join(output_dir, f"predictions_{split_name}.json")
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
            
        # 2. Lưu file metrics
        metrics_path = os.path.join(output_dir, f"metrics_{split_name}.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
            
        # 3. Lưu qualitative analysis (phân tích định tính)
        sample_path = os.path.join(output_dir, f"qualitative_{split_name}.txt")
        self.save_sample_results(predictions, sample_path, n_samples=50)
        
        print(f"Đã lưu kết quả dự đoán tại:  {pred_path}")
        print(f"Đã lưu các chỉ số đánh giá tại: {metrics_path}")
        print(f"Đã lưu 50 mẫu định tính tại:   {sample_path}")
        
        return metrics

    def print_results(self, metrics: Dict[str, float]) -> None:
        """In bảng kết quả các chỉ số đánh giá theo định dạng Markdown.

        Args:
            metrics: Dictionary chứa kết quả các độ đo.
        """
        print("\n" + "=" * 50)
        print(" KẾT QUẢ ĐÁNH GIÁ CHẤT LƯỢNG CHÚ THÍCH ẢNH ")
        print("=" * 50)
        print("| Chỉ số (Metric)   | Giá trị (%) / Điểm số |")
        print("|-------------------|------------------------|")
        for k, v in metrics.items():
            print(f"| {k:<17} | {v:14.2f} |")
        print("=" * 50 + "\n")

    def save_sample_results(
        self, 
        predictions: Dict[int, str], 
        output_path: str, 
        n_samples: int = 50
    ) -> None:
        """Lưu lại một số mẫu để phân tích định tính (so sánh dự đoán và nhãn gốc).

        Ghi thông tin image_id, câu dự đoán từ mô hình và danh sách các câu chú thích
        tham chiếu chuẩn của con người viết.

        Args:
            predictions: Dictionary câu chú thích dự đoán của mô hình.
            output_path: Đường dẫn lưu file text.
            n_samples: Số lượng mẫu tối đa muốn lưu (mặc định 50).
        """
        # Lấy tham chiếu gốc từ DataLoader để ghi nhận nhãn chuẩn
        dataset = self.dataloader.dataset
        
        # Tạo ánh xạ nhanh từ image_id sang danh sách các câu chuẩn trong tập dữ liệu
        id_to_refs = {}
        for i in range(len(dataset)):
            item = dataset[i]
            # Tùy thuộc vào thiết kế dataset trả về image_id dạng nào
            img_id = item["image_id"]
            img_key = int(img_id.item()) if isinstance(img_id, torch.Tensor) else img_id
            
            raw_caps = item["raw_captions"]
            id_to_refs[img_key] = raw_caps
            
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(" PHÂN TÍCH ĐỊNH TÍNH CHẤT LƯỢNG CHÚ THÍCH (QUALITATIVE SAMPLES)\n")
            f.write("=" * 80 + "\n\n")
            
            count = 0
            for img_id, pred_cap in predictions.items():
                if count >= n_samples:
                    break
                    
                refs = id_to_refs.get(img_id, ["Không tìm thấy câu tham chiếu"])
                
                f.write(f"Mẫu số {count + 1} | Image ID: {img_id}\n")
                f.write(f"  [DỰ ĐOÁN] - {pred_cap}\n")
                f.write("  [THAM CHIẾU CHUẨN]:\n")
                for i, r in enumerate(refs):
                    f.write(f"    {i+1}. {r}\n")
                f.write("-" * 80 + "\n")
                count += 1
