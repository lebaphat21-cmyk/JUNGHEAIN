"""Mô hình chú thích ảnh hoàn chỉnh (Image Captioning Model).

Lắp ghép các thành phần:
1. Visual Encoder (Bộ mã hóa thị giác): Trích xuất đặc trưng ảnh bằng ResNet-101 + Transformer Encoder.
2. Semantic Encoder (Bộ mã hóa ngữ nghĩa): Trích xuất và mã hóa các bộ ba Scene Graph (đối tượng, quan hệ, đối tượng).
3. Fusion Module (Bộ hợp nhất đặc trưng): Hợp nhất đặc trưng thị giác và ngữ nghĩa theo 4 chiến lược (Concat, Cross-Attn, Gated, Co-Attn).
4. Caption Decoder (Bộ giải mã chú thích): Sinh câu chú thích bằng Transformer Decoder với Beam Search hoặc SCST Sampling.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Optional, List
import torch
import torch.nn as nn

from src.models.visual_encoder import VisualEncoder
from src.models.semantic_encoder import SemanticEncoder
from src.models.fusion import build_fusion
from src.models.caption_decoder import CaptionDecoder


class ImageCaptioningModel(nn.Module):
    """Mô hình chú thích ảnh kết hợp đặc trưng thị giác và ngữ nghĩa.

    Lớp này đóng vai trò là kiến trúc tổng thể, quản lý luồng dữ liệu đi qua
    bộ mã hóa ảnh, bộ mã hóa scene graph, mô-đun hợp nhất đặc trưng và
    bộ giải mã văn bản.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_enc_layers: int = 3,
        n_dec_layers: int = 3,
        d_ff: int = 2048,
        dropout: float = 0.1,
        fusion_type: str = "cross_attention",
        visual_backbone: str = "resnet101",
        freeze_cnn: bool = True,
        max_triples: int = 20,
        use_confidence: bool = False,
        multi_scale_fusion: bool = False,
        fusion_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Khởi tạo mô hình chú thích ảnh.

        Args:
            vocab_size: Kích thước từ điển.
            d_model: Số chiều ẩn của mô hình (mặc định 512).
            n_heads: Số đầu attention trong Transformer (mặc định 8).
            n_enc_layers: Số lớp Encoder cho visual/semantic (mặc định 3).
            n_dec_layers: Số lớp Decoder (mặc định 3).
            d_ff: Kích thước lớp FeedForward ẩn (mặc định 2048).
            dropout: Tỷ lệ dropout (mặc định 0.1).
            fusion_type: Loại chiến lược hợp nhất ('concat', 'cross_attention', 'gated',
                'co_attention', 'gated_co_attention', 'adaptive').
            visual_backbone: Tên backbone thị giác (mặc định 'resnet101').
            freeze_cnn: Nếu True, đóng băng trọng số của CNN backbone ở đầu quá trình huấn luyện.
            max_triples: Số lượng bộ ba tối đa được giữ lại từ Scene Graph.
            use_confidence: Nếu True, bật cơ chế ước lượng độ tin cậy (confidence estimation)
                cho từng bộ ba Scene Graph để lọc nhiễu.
            multi_scale_fusion: Nếu True, bật Residual Multi-scale Fusion — hợp nhất đặc trưng
                tại từng tầng Encoder (thay vì chỉ tầng cuối), dùng residual connection.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.fusion_type = fusion_type
        self.multi_scale_fusion = multi_scale_fusion
        self.max_triples = max_triples
        fusion_kwargs = dict(fusion_kwargs or {})

        # 1. Bộ mã hóa thị giác (Visual Encoder)
        self.visual_encoder = VisualEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_enc_layers,
            d_ff=d_ff,
            dropout=dropout,
            freeze_cnn=freeze_cnn,
            backbone=visual_backbone,
        )

        # 2. Bộ mã hóa ngữ nghĩa (Semantic Encoder)
        self.semantic_encoder = SemanticEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=max(1, n_enc_layers - 1),  # Thường dùng ít lớp hơn một chút
            d_ff=d_ff,
            max_triples=max_triples,
            dropout=dropout,
            use_confidence=use_confidence,
        )

        # 3. Mô-đun hợp nhất đặc trưng (Fusion Module)
        self.fusion_module = build_fusion(
            fusion_type=fusion_type,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            **fusion_kwargs,
        )

        # 3.5. Multi-scale Fusion: thêm các fusion module cho từng tầng encoder
        if self.multi_scale_fusion:
            # Số tầng của Visual Encoder (quyết định số scale)
            n_visual_layers = n_enc_layers
            # Mỗi tầng dùng chung loại fusion (shared weights để tiết kiệm params)
            self.scale_fusion_modules = nn.ModuleList([
                build_fusion(
                    fusion_type=fusion_type,
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                    **fusion_kwargs,
                )
                for _ in range(n_visual_layers)
            ])
            # LayerNorm cuối cùng để chuẩn hoá đặc trưng multi-scale tích luỹ
            self.ms_output_norm = nn.LayerNorm(d_model)

        # 4. Bộ giải mã chú thích (Caption Decoder)
        self.caption_decoder = CaptionDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_dec_layers,
            d_ff=d_ff,
            dropout=dropout,
        )

    @classmethod
    def from_config(cls, config: Any, vocab_size: int) -> ImageCaptioningModel:
        """Tạo đối tượng mô hình từ file cấu hình.

        Args:
            config: Đối tượng chứa các tham số cấu hình.
            vocab_size: Kích thước từ điển.

        Returns:
            Thể hiện của lớp ImageCaptioningModel.
        """
        fusion_type = config.fusion.type
        fusion_section = config.fusion.get(fusion_type)
        # Accept the short aliases used by older configs.
        if fusion_section is None and fusion_type == "gated_co_attn":
            fusion_section = config.fusion.get("gated_co_attention")
        fusion_kwargs = fusion_section.to_dict() if fusion_section is not None else {}
        # These dimensions belong to the model, while the remaining options
        # (linear_attention, proj_len, types, ...) come from the fusion block.
        fusion_kwargs.pop("n_heads", None)
        fusion_kwargs["max_v_len"] = int(config.model.get("grid_size", 7)) ** 2
        fusion_kwargs["max_s_len"] = int(config.model.max_triples)

        return cls(
            vocab_size=vocab_size,
            d_model=config.model.d_model,
            n_heads=config.model.n_heads,
            n_enc_layers=config.model.n_encoder_layers,
            n_dec_layers=config.model.n_decoder_layers,
            d_ff=config.model.d_ff,
            dropout=config.model.dropout,
            fusion_type=fusion_type,
            visual_backbone=config.model.visual_backbone,
            freeze_cnn=config.model.get("freeze_cnn", config.model.get("freeze_backbone", True)),
            max_triples=config.model.max_triples,
            use_confidence=config.model.get("use_confidence", False),
            multi_scale_fusion=config.model.get("multi_scale_fusion", False),
            fusion_kwargs=fusion_kwargs,
        )

    def freeze_backbone(self) -> None:
        """Đóng băng CNN backbone để trích xuất đặc trưng tĩnh."""
        self.visual_encoder.freeze_backbone()

    def unfreeze_backbone(self) -> None:
        """Mở băng CNN backbone để huấn luyện end-to-end (tinh chỉnh/fine-tune)."""
        self.visual_encoder.unfreeze_backbone()

    def forward(
        self,
        images: torch.Tensor,
        triples: torch.Tensor,
        triple_mask: torch.Tensor,
        captions: torch.Tensor,
        caption_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Lan truyền xuôi (Forward pass) trong quá trình huấn luyện (sử dụng Teacher Forcing).

        Args:
            images: Batch ảnh đầu vào, kích thước (B, 3, H, W).
            triples: Batch các bộ ba ngữ nghĩa dạng chỉ số từ điển, kích thước (B, K, 3).
            triple_mask: Mask của các bộ ba (True tại vị trí pad), kích thước (B, K).
            captions: Các câu chú thích đích (bao gồm token bắt đầu <start>), kích thước (B, L).
            caption_lengths: Độ dài thực tế của từng câu chú thích, kích thước (B,).

        Returns:
            Logits dự đoán phân phối từ vựng cho từng vị trí sinh, kích thước (B, L-1, vocab_size).
        """
        # 1. Trích xuất đặc trưng thị giác và ngữ nghĩa
        if self.multi_scale_fusion:
            # Residual Multi-scale Fusion: lấy đặc trưng từng tầng encoder
            v_layers, visual_mask = self.visual_encoder.forward_intermediate(images)
            s_layers, semantic_mask = self.semantic_encoder.forward_intermediate(triples, triple_mask)
            fused_feats, fused_mask = self._multi_scale_fuse(
                v_layers, s_layers, visual_mask, semantic_mask
            )
        else:
            # Single-scale fusion (mặc định)
            visual_feats, visual_mask = self.visual_encoder(images)
            semantic_feats, semantic_mask = self.semantic_encoder(triples, triple_mask)
            fused_feats, fused_mask = self.fusion_module(
                visual_feats, semantic_feats, visual_mask, semantic_mask
            )

        # 4. Giải mã sinh từ (Teacher forcing)
        # logits: (B, L-1, vocab_size)
        logits = self.caption_decoder(
            fused_feats, fused_mask, captions, caption_lengths
        )

        return logits

    def generate(
        self,
        images: torch.Tensor,
        triples: torch.Tensor,
        triple_mask: torch.Tensor,
        max_len: int = 30,
        beam_size: int = 5,
        start_idx: int = 1,
        end_idx: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sinh câu chú thích ảnh bằng thuật toán Beam Search (quá trình Inference).

        Args:
            images: Batch ảnh đầu vào, kích thước (B, 3, H, W).
            triples: Batch bộ ba ngữ nghĩa, kích thước (B, K, 3).
            triple_mask: Mask của bộ ba, kích thước (B, K).
            max_len: Độ dài tối đa của câu chú thích sinh ra.
            beam_size: Độ rộng của chùm (beam size).
            start_idx: Chỉ số của token bắt đầu <start> (mặc định 1).
            end_idx: Chỉ số của token kết thúc <end> (mặc định 2).

        Returns:
            Tuple chứa:
                - **captions**: Chỉ số các từ trong câu sinh ra, kích thước (B, max_len).
                - **scores**: Điểm số log-probability tương ứng của các câu, kích thước (B,).
        """
        # Trích xuất và hợp nhất đặc trưng
        if self.multi_scale_fusion:
            v_layers, visual_mask = self.visual_encoder.forward_intermediate(images)
            s_layers, semantic_mask = self.semantic_encoder.forward_intermediate(triples, triple_mask)
            fused_feats, fused_mask = self._multi_scale_fuse(
                v_layers, s_layers, visual_mask, semantic_mask
            )
        else:
            visual_feats, visual_mask = self.visual_encoder(images)
            semantic_feats, semantic_mask = self.semantic_encoder(triples, triple_mask)
            fused_feats, fused_mask = self.fusion_module(
                visual_feats, semantic_feats, visual_mask, semantic_mask
            )

        # Gọi hàm sinh từ bằng Beam Search trong decoder
        return self.caption_decoder.generate(
            fused_features=fused_feats,
            fused_mask=fused_mask,
            max_len=max_len,
            beam_size=beam_size,
            start_idx=start_idx,
            end_idx=end_idx,
        )

    def sample(
        self,
        images: torch.Tensor,
        triples: torch.Tensor,
        triple_mask: torch.Tensor,
        max_len: int = 30,
        start_idx: int = 1,
        end_idx: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Lấy mẫu ngẫu nhiên có trọng số (Multinomial Sampling) để phục vụ huấn luyện SCST.

        Args:
            images: Batch ảnh đầu vào.
            triples: Batch bộ ba ngữ nghĩa.
            triple_mask: Mask bộ ba.
            max_len: Độ dài tối đa câu chú thích.
            start_idx: Chỉ số token <start>.
            end_idx: Chỉ số token <end>.

        Returns:
            Tuple chứa:
                - **sampled_ids**: Chỉ số các từ được lấy mẫu, kích thước (B, max_len).
                - **log_probs**: Log-probability của các từ được chọn, kích thước (B, max_len).
        """
        # Trích xuất và hợp nhất đặc trưng
        if self.multi_scale_fusion:
            v_layers, visual_mask = self.visual_encoder.forward_intermediate(images)
            s_layers, semantic_mask = self.semantic_encoder.forward_intermediate(triples, triple_mask)
            fused_feats, fused_mask = self._multi_scale_fuse(
                v_layers, s_layers, visual_mask, semantic_mask
            )
        else:
            visual_feats, visual_mask = self.visual_encoder(images)
            semantic_feats, semantic_mask = self.semantic_encoder(triples, triple_mask)
            fused_feats, fused_mask = self.fusion_module(
                visual_feats, semantic_feats, visual_mask, semantic_mask
            )

        # Gọi hàm lấy mẫu trong decoder
        return self.caption_decoder.sample(
            fused_features=fused_feats,
            fused_mask=fused_mask,
            max_len=max_len,
            start_idx=start_idx,
            end_idx=end_idx,
        )

    def _multi_scale_fuse(
        self,
        v_layers: List[torch.Tensor],
        s_layers: List[torch.Tensor],
        visual_mask: Optional[torch.Tensor],
        semantic_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Residual Multi-scale Fusion: hợp nhất đặc trưng visual và semantic ở nhiều tầng.

        Với mỗi tầng l của Visual Encoder:
          - Lấy đặc trưng semantic tại tầng l (hoặc tầng cuối nếu l vượt quá số tầng semantic).
          - Fuse visual_l với semantic_l.
          - Cộng dồn (residual) vào kết quả tích luỹ.
        Áp dụng LayerNorm cuối cùng trên tổng tích luỹ.

        Args:
            v_layers: Danh sách đặc trưng visual mỗi tầng, mỗi phần tử (B, N_v, d).
            s_layers: Danh sách đặc trưng semantic mỗi tầng, mỗi phần tử (B, N_s, d).
            visual_mask: (B, N_v) bool mask hoặc None.
            semantic_mask: (B, N_s) bool mask hoặc None.

        Returns:
            Tuple:
                - **fused_acc**: ``(B, N_fused, d_model)`` đặc trưng tổng hợp.
                - **fused_mask**: ``(B, N_fused)`` mask hoặc None.
        """
        fused_acc: Optional[torch.Tensor] = None
        fused_mask: Optional[torch.Tensor] = None
        n_sem_layers = len(s_layers)

        for l, (v_feat, scale_fuser) in enumerate(zip(v_layers, self.scale_fusion_modules)):
            # Lấy semantic tại tầng l, hoặc giữ tầng cuối nếu l vượt quá
            s_feat = s_layers[min(l, n_sem_layers - 1)]

            fused_l, fused_mask = scale_fuser(
                v_feat, s_feat, visual_mask, semantic_mask
            )

            if fused_acc is None:
                fused_acc = fused_l
            else:
                # Residual: cộng dồn các scale, đảm bảo cùng chiều sequence
                min_len = min(fused_acc.size(1), fused_l.size(1))
                fused_acc = fused_acc[:, :min_len, :] + fused_l[:, :min_len, :]
                if fused_mask is not None:
                    fused_mask = fused_mask[:, :min_len]

        # Chuẩn hoá output tổng hợp
        fused_acc = self.ms_output_norm(fused_acc)
        return fused_acc, fused_mask

    def get_parameter_groups(
        self, lr_backbone: float, lr_rest: float
    ) -> List[Dict[str, Any]]:
        """Chia các nhóm tham số với learning rate khác nhau cho quá trình tối ưu.

        Thường dùng khi fine-tune: Backbone CNN chạy với LR nhỏ hơn nhiều để tránh
        phá vỡ các đặc trưng đã học từ trước, các phần còn lại dùng LR lớn hơn.

        Args:
            lr_backbone: Learning rate cho backbone CNN.
            lr_rest: Learning rate cho các phần còn lại của mô hình.

        Returns:
            Danh sách các dictionary định dạng nhóm tham số của PyTorch Optimizer.
        """
        backbone_params = []
        rest_params = []

        # Tách tham số của CNN backbone
        for name, param in self.named_parameters():
            if param.requires_grad:
                if "visual_encoder.backbone" in name:
                    backbone_params.append(param)
                else:
                    rest_params.append(param)

        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": lr_backbone})
        if rest_params:
            groups.append({"params": rest_params, "lr": lr_rest})

        return groups

    def print_summary(self) -> None:
        """In ra thông tin tóm tắt số lượng tham số của mô hình."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        visual_params = sum(p.numel() for p in self.visual_encoder.parameters())
        semantic_params = sum(p.numel() for p in self.semantic_encoder.parameters())
        fusion_params = sum(p.numel() for p in self.fusion_module.parameters())
        decoder_params = sum(p.numel() for p in self.caption_decoder.parameters())

        print("=" * 60)
        print(" TÓM TẮT THAM SỐ MÔ HÌNH CHÚ THÍCH ẢNH ")
        print("=" * 60)
        print(f"Tổng số tham số:              {total_params:,}")
        print(f"Số tham số có thể huấn luyện: {trainable_params:,}")
        print("-" * 60)
        print(f"  - Visual Encoder:           {visual_params:,}")
        print(f"  - Semantic Encoder:         {semantic_params:,}")
        print(f"  - Fusion Module ({self.fusion_type}): {fusion_params:,}")
        if self.multi_scale_fusion:
            ms_params = sum(p.numel() for p in self.scale_fusion_modules.parameters())
            ms_norm_params = sum(p.numel() for p in self.ms_output_norm.parameters())
            print(f"  - Multi-scale Fusion:       {ms_params + ms_norm_params:,}")
        print(f"  - Caption Decoder:          {decoder_params:,}")
        print("=" * 60)
