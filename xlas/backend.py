#!/usr/bin/env python
"""Giao diện Web Demo cho hệ thống Video Captioning & Text-to-Speech.

Script này xây dựng một giao diện web trực quan bằng Gradio, cho phép người dùng:
1. Tải lên một đoạn video bất kỳ.
2. Hệ thống tự động trích xuất các khung hình, phân tích đặc trưng thị giác
   và ngữ nghĩa (Scene Graph), rồi sinh câu chú thích bằng mô hình Transformer.
3. Chuyển câu chú thích thành giọng nói (Text-to-Speech) và phát trực tiếp.

Cách chạy:
    python web_demo.py --checkpoint <đường_dẫn_checkpoint>

Truy cập giao diện tại: http://127.0.0.1:7860
"""

import argparse
import os
import sys
import tempfile
import time
import numpy as np
from PIL import Image
import cv2
import speech_recognition as sr
from google import genai
from openai import OpenAI

import torch
import torchvision.transforms as T

# Thêm thư mục gốc vào PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Đảm bảo in tiếng Việt không lỗi trên console Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

from src.utils.config import load_config
from src.utils.misc import load_checkpoint, resolve_checkpoint_path
from src.data.vocabulary import Vocabulary
from src.data.transforms import get_val_transforms
from src.models.captioning_model import ImageCaptioningModel
from src.scene_graph.extract_sg import create_extractor
from src.scene_graph.sg_utils import encode_triples
from src.utils.video_processor import extract_frames, generate_tts_audio


# ===========================================================================
# Cấu hình đường dẫn lưu trữ outputs của XLAS
# ===========================================================================
XLAS_DIR = os.path.dirname(os.path.abspath(__file__))
XLAS_OUTPUTS_DIR = os.path.join(XLAS_DIR, "outputs")
XLAS_AUDIO_DIR = os.path.join(XLAS_OUTPUTS_DIR, "audio")
XLAS_FACES_DIR = os.path.join(XLAS_OUTPUTS_DIR, "faces")
XLAS_CAPTIONS_DIR = os.path.join(XLAS_OUTPUTS_DIR, "captions")
os.makedirs(XLAS_AUDIO_DIR, exist_ok=True)
os.makedirs(XLAS_FACES_DIR, exist_ok=True)
os.makedirs(XLAS_CAPTIONS_DIR, exist_ok=True)


def get_temp_audio_file():
    """Tạo file âm thanh tạm thời trong thư mục outputs/audio của XLAS."""
    target_dir = XLAS_AUDIO_DIR if os.path.isdir(XLAS_AUDIO_DIR) else tempfile.gettempdir()
    audio_tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False, dir=target_dir)
    audio_tmp.close()
    return audio_tmp.name


# ===========================================================================
# Biến toàn cục cho mô hình (nạp một lần khi khởi động)
# ===========================================================================
MODEL = None
VOCAB = None
CONFIG = None
DEVICE = None
TRANSFORMS = None
SG_EXTRACTOR = None
BLIP_PROCESSOR = None
BLIP_MODEL = None


def load_model_global(checkpoint_path, config_path="configs/base_config.yaml",
                      sg_method="heuristic", gpu=0):
    """Nạp mô hình và các thành phần cần thiết vào bộ nhớ (chỉ gọi một lần)."""
    global MODEL, VOCAB, CONFIG, DEVICE, TRANSFORMS, SG_EXTRACTOR

    # 1. Đọc file cấu hình
    CONFIG = load_config(config_path)

    # 2. Nạp từ điển
    vocab_path = os.path.join(CONFIG.data.vocab_dir, "vocab.json")
    if not os.path.exists(vocab_path):
        # Thử tìm ở vị trí khác
        vocab_path = CONFIG.data.get("vocab_file", "data/vocab/vocab.json")

    if not os.path.exists(vocab_path):
        raise FileNotFoundError(
            f"Không tìm thấy file từ điển tại {vocab_path}!\n"
            "Vui lòng đảm bảo đã chạy bước xây dựng từ điển (build_vocab.py)."
        )

    VOCAB = Vocabulary.load_from_file(vocab_path)
    vocab_size = len(VOCAB)

    # 3. Xác định thiết bị
    DEVICE = torch.device(
        f"cuda:{gpu}" if gpu >= 0 and torch.cuda.is_available() else "cpu"
    )
    print(f"[Web Demo] Thiết bị tính toán: {DEVICE}")

    # 4. Xây dựng mô hình và nạp trọng số
    try:
        checkpoint_path = resolve_checkpoint_path(checkpoint_path)
        print(f"[Web Demo] Đang nạp mô hình chú thích ảnh từ checkpoint: {checkpoint_path}...")
        MODEL = ImageCaptioningModel.from_config(CONFIG, vocab_size=vocab_size)
        MODEL.to(DEVICE)
        load_checkpoint(checkpoint_path, MODEL, map_location=DEVICE)
        MODEL.eval()
        print("[Web Demo] Nạp mô hình thành công!")
    except Exception as e:
        MODEL = None
        print("\n" + "!" * 80)
        print(f"⚠️ CẢNH BÁO: KHÔNG THỂ NẠP FILE CHECKPOINT: {checkpoint_path}")
        print(f"Chi tiết lỗi: {e}")
        print("Giao diện Gradio vẫn sẽ khởi chạy, nhưng các chức năng sinh caption sẽ hiển thị thông báo lỗi này.")
        print("!" * 80 + "\n")

    # 5. Chuẩn bị bộ biến đổi ảnh
    TRANSFORMS = get_val_transforms(
        image_size=CONFIG.data.get("image_size", 256),
        crop_size=CONFIG.data.get("crop_size", 224)
    )

    # 6. Khởi tạo bộ trích xuất Scene Graph
    print(f"[Web Demo] Khởi tạo bộ trích xuất Scene Graph (phương pháp: {sg_method})...")
    try:
        SG_EXTRACTOR = create_extractor(method=sg_method, device=str(DEVICE))
        print("[Web Demo] Khởi tạo bộ trích xuất Scene Graph thành công!")
    except Exception as e:
        SG_EXTRACTOR = None
        print(f"⚠️ Lỗi khi khởi tạo bộ trích xuất Scene Graph: {e}")
    print("[Web Demo] Hoàn tất quá trình khởi tạo!\n")

    # 7. Nạp mô hình BLIP pre-trained nếu mô hình chính không khả dụng
    if MODEL is None:
        load_blip_captioner()


def load_blip_captioner():
    """Nạp mô hình BLIP pre-trained (Salesforce) để sinh chú thích chi tiết cho mọi loại ảnh.

    Mô hình này được sử dụng như fallback khi chưa có checkpoint mô hình NCKH riêng.
    Lần chạy đầu tiên sẽ tự động tải trọng số từ HuggingFace (~990MB).
    """
    global BLIP_PROCESSOR, BLIP_MODEL

    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration

        print("[Web Demo] Đang nạp mô hình BLIP pre-trained để sinh chú thích chi tiết...")
        print("[Web Demo] (Lần đầu tiên sẽ tải trọng số ~990MB từ HuggingFace, vui lòng đợi...)")

        BLIP_PROCESSOR = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        )
        BLIP_MODEL = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large",
            torch_dtype=torch.float32,
        )
        blip_device = DEVICE if DEVICE is not None else torch.device("cpu")
        BLIP_MODEL.to(blip_device)
        BLIP_MODEL.eval()
        print(f"[Web Demo] Nạp mô hình BLIP thành công! (thiết bị: {blip_device})")
    except Exception as e:
        BLIP_PROCESSOR = None
        BLIP_MODEL = None
        print(f"⚠️ Không thể nạp mô hình BLIP: {e}")
        print("   Hệ thống sẽ sử dụng phương pháp mô phỏng từ Scene Graph thay thế.")


def generate_blip_caption(pil_image, detail_level="high"):
    """Sinh chú thích chi tiết cho ảnh bằng mô hình BLIP pre-trained.

    Args:
        pil_image: Ảnh PIL cần sinh chú thích.
        detail_level: Mức độ chi tiết ('high' sinh nhiều câu mô tả khác nhau).

    Returns:
        str: Câu mô tả chi tiết nhất, hoặc None nếu BLIP không khả dụng.
    """
    if BLIP_PROCESSOR is None or BLIP_MODEL is None:
        return None

    try:
        blip_device = DEVICE if DEVICE is not None else torch.device("cpu")
        rgb_image = pil_image.convert("RGB")

        captions = []

        # 1. Caption không điều kiện (unconditional) — mô tả tổng quan
        inputs = BLIP_PROCESSOR(rgb_image, return_tensors="pt").to(blip_device)
        with torch.no_grad():
            out = BLIP_MODEL.generate(
                **inputs,
                max_new_tokens=80,
                num_beams=5,
                repetition_penalty=1.5,
            )
        cap_general = BLIP_PROCESSOR.decode(out[0], skip_special_tokens=True).strip()
        captions.append(cap_general)

        if detail_level == "high":
            # 2. Caption có điều kiện (conditional) — mô tả chi tiết hơn
            prompts = [
                "a detailed photograph of",
                "this image shows",
            ]
            for prompt in prompts:
                inputs = BLIP_PROCESSOR(rgb_image, text=prompt, return_tensors="pt").to(blip_device)
                with torch.no_grad():
                    out = BLIP_MODEL.generate(
                        **inputs,
                        max_new_tokens=80,
                        num_beams=5,
                        repetition_penalty=1.5,
                    )
                cap = BLIP_PROCESSOR.decode(out[0], skip_special_tokens=True).strip()
                captions.append(cap)

        # Chọn câu dài nhất (thường chi tiết nhất)
        best_caption = max(captions, key=len)
        # Viết hoa chữ cái đầu
        best_caption = best_caption[0].upper() + best_caption[1:] if best_caption else best_caption
        # Đảm bảo kết thúc bằng dấu chấm
        if best_caption and not best_caption.endswith("."):
            best_caption += "."

        return best_caption
    except Exception as e:
        print(f"[BLIP Warning] Lỗi khi sinh chú thích bằng BLIP: {e}")
        return None


def draw_scene_graph_on_image(pil_image, triples):
    """Vẽ các hộp bao (bounding boxes) và liên kết quan hệ ngữ cảnh trực tiếp lên ảnh."""
    if pil_image is None or not triples:
        return pil_image

    from PIL import ImageDraw, ImageFont

    # Tạo bản sao của ảnh để không ghi đè lên ảnh gốc
    draw_img = pil_image.copy().convert("RGB")
    draw = ImageDraw.Draw(draw_img)

    # Chọn font mặc định
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # Danh sách màu sắc nổi bật để vẽ các hộp đối tượng khác nhau
    colors = [
        "#E11D48", "#2563EB", "#16A34A", "#D97706", 
        "#7C3AED", "#0891B2", "#EA580C", "#DB2777"
    ]

    # Lọc lấy những bộ ba có thông tin box và sắp xếp theo confidence
    valid_triples = [t for t in triples if "subject_box" in t and "object_box" in t]
    valid_triples = sorted(valid_triples, key=lambda x: x.get("confidence", 0), reverse=True)[:6]

    # Để tránh vẽ đè nhãn lên cùng một vị trí của đối tượng
    drawn_boxes = {}

    for idx, t in enumerate(valid_triples):
        color = colors[idx % len(colors)]
        
        sub = str(t["subject"])
        pred = str(t["predicate"])
        obj = str(t["object"])
        
        sub_box = t["subject_box"]
        obj_box = t["object_box"]

        # Vẽ hộp bao của Subject nếu chưa vẽ
        sub_box_tuple = tuple(sub_box)
        if sub_box_tuple not in drawn_boxes:
            draw.rectangle(sub_box, outline=color, width=3)
            draw.text((sub_box[0] + 4, sub_box[1] + 4), sub, fill=color, font=font)
            drawn_boxes[sub_box_tuple] = sub

        # Vẽ hộp bao của Object nếu chưa vẽ
        obj_box_tuple = tuple(obj_box)
        if obj_box_tuple not in drawn_boxes:
            draw.rectangle(obj_box, outline=color, width=3)
            draw.text((obj_box[0] + 4, obj_box[1] + 4), obj, fill=color, font=font)
            drawn_boxes[obj_box_tuple] = obj

        # Chỉ vẽ đường nối nếu subject và object khác nhau
        if sub_box_tuple != obj_box_tuple:
            center_sub = ((sub_box[0] + sub_box[2]) / 2, (sub_box[1] + sub_box[3]) / 2)
            center_obj = ((obj_box[0] + obj_box[2]) / 2, (obj_box[1] + obj_box[3]) / 2)
            
            # Vẽ đường nét liền mảnh kết nối 2 tâm
            draw.line([center_sub, center_obj], fill=color, width=2)
            
            # Vẽ nhãn quan hệ ở giữa đường nối
            mid_x = (center_sub[0] + center_obj[0]) / 2
            mid_y = (center_sub[1] + center_obj[1]) / 2
            
            # Tạo hộp màu tối nhỏ làm nền cho chữ quan hệ để tăng độ tương phản
            text_w = len(pred) * 6 + 10
            draw.rectangle([mid_x - text_w/2, mid_y - 8, mid_x + text_w/2, mid_y + 8], fill="#1E293B")
            draw.text((mid_x - text_w/2 + 5, mid_y - 6), pred, fill="#FFFFFF", font=font)

    return draw_img


def analyze_image_context(pil_image):
    """Phân tích ngữ cảnh ảnh qua đặc trưng màu sắc và phân bổ pixel.

    Returns:
        dict chứa thông tin môi trường (environment), tỷ lệ màu, v.v.
    """
    if pil_image is None:
        return {"environment": "general"}

    try:
        img_small = pil_image.resize((32, 32)).convert("RGB")
        img_arr = np.array(img_small)  # shape: (32, 32, 3)
        pixels = img_arr.reshape(-1, 3).tolist()  # list of [R, G, B]
        n = len(pixels)

        avg_r = sum(p[0] for p in pixels) / n
        avg_g = sum(p[1] for p in pixels) / n
        avg_b = sum(p[2] for p in pixels) / n

        # Phân tích vùng trên (bầu trời) và vùng dưới (mặt đất)
        top_pixels = pixels[:n // 2]
        bot_pixels = pixels[n // 2:]
        bot_g = sum(p[1] for p in bot_pixels) / len(bot_pixels)
        bot_r = sum(p[0] for p in bot_pixels) / len(bot_pixels)

        # Đếm tỷ lệ pixel theo nhóm màu
        green_count = sum(1 for p in pixels if p[1] > p[0] * 0.9 and p[1] > p[2] and p[1] > 70)
        blue_count = sum(1 for p in pixels if p[2] > p[0] and p[2] > p[1] and p[2] > 100)
        brown_count = sum(1 for p in pixels if p[0] > p[2] and abs(p[0] - p[1]) < 50 and p[0] > 80 and p[2] < 120)
        gray_count = sum(1 for p in pixels if abs(p[0] - p[1]) < 25 and abs(p[1] - p[2]) < 25)

        context = {
            "avg_r": avg_r, "avg_g": avg_g, "avg_b": avg_b,
            "green_ratio": green_count / n,
            "blue_ratio": blue_count / n,
            "brown_ratio": brown_count / n,
            "gray_ratio": gray_count / n,
        }

        # Phát hiện môi trường dựa trên tỷ lệ màu sắc
        if context["green_ratio"] > 0.20 or (avg_g > 90 and avg_g > avg_b * 1.15 and avg_r > avg_b):
            # Vùng xanh lá: đồng ruộng, vườn, công viên
            if bot_g > 80 and bot_r > bot_g * 0.7:
                context["environment"] = "rice_field"
            else:
                context["environment"] = "outdoor_green"
        elif context["blue_ratio"] > 0.25:
            context["environment"] = "sky_or_water"
        elif context["gray_ratio"] > 0.4:
            context["environment"] = "indoor_or_urban"
        elif avg_r > 150 and avg_g > 150 and avg_b > 150:
            context["environment"] = "bright_outdoor"
        else:
            context["environment"] = "general_outdoor"

        return context
    except Exception as e:
        print(f"[Heuristic Warning] Lỗi phân tích đặc trưng ảnh: {e}")
        return {"environment": "general"}


def generate_simulated_caption(triples, pil_image=None):
    """Sinh câu mô tả tiếng Anh chi tiết dựa trên phân tích Scene Graph + đặc trưng thị giác của ảnh.

    Hệ thống kết hợp:
    1. Thử sinh chú thích cực kỳ chi tiết bằng mô hình BLIP pre-trained
    2. Phương án dự phòng: Thuật toán Rule-based + Scene Graph phân tích màu sắc
    """
    # 1. Ưu tiên sử dụng mô hình BLIP sinh chú thích chi tiết động
    if pil_image is not None:
        blip_caption = generate_blip_caption(pil_image)
        if blip_caption:
            return blip_caption

    # 2. Dự phòng: Phân tích ngữ cảnh ảnh qua màu sắc
    context = analyze_image_context(pil_image) if pil_image else {}
    env = context.get("environment", "general")

    # ========================
    # 2. Thu thập tất cả đối tượng và quan hệ từ Scene Graph
    # ========================
    all_subjects = []
    all_objects_list = []
    all_predicates = []
    unique_entities = set()

    for t in triples:
        sub = str(t.get("subject", "")).lower().strip()
        pred = str(t.get("predicate", "")).lower().strip()
        obj = str(t.get("object", "")).lower().strip()
        if sub:
            all_subjects.append(sub)
            unique_entities.add(sub)
        if obj:
            all_objects_list.append(obj)
            unique_entities.add(obj)
        if sub and pred and obj:
            all_predicates.append((sub, pred, obj))

    # Đếm số lượng từng loại đối tượng
    entity_counts = {}
    for e in all_subjects + all_objects_list:
        entity_counts[e] = entity_counts.get(e, 0) + 1

    person_count = entity_counts.get("person", 0)
    # Đảm bảo tối thiểu 1 nếu có "person" trong danh sách
    if "person" in unique_entities and person_count == 0:
        person_count = 1

    # Danh sách các đối tượng không phải người
    non_person_entities = [e for e in unique_entities if e != "person"]

    # ========================
    # 3. Xây dựng mô tả chủ thể (Subject Description)
    # ========================
    if person_count >= 3:
        subject_desc = f"a group of {person_count} people"
    elif person_count == 2:
        subject_desc = "two people"
    elif person_count == 1:
        subject_desc = "a person"
    else:
        subject_desc = ""

    # ========================
    # 4. Xây dựng mô tả hành động (Action Description)
    # ========================
    action_parts = []
    seen_actions = set()
    for sub, pred, obj in all_predicates:
        action_key = (pred, obj)
        if action_key in seen_actions:
            continue
        seen_actions.add(action_key)

        if pred in ["wearing", "holding", "riding", "eating", "carrying",
                     "playing", "driving", "watching", "looking at", "sitting on",
                     "standing on", "walking on", "using"]:
            action_parts.append(f"{pred} a {obj}")
        elif pred in ["on", "in", "near", "next to", "behind", "under",
                       "above", "beside", "along"] and obj != "person":
            action_parts.append(f"{pred} a {obj}")

    action_desc = ""
    if action_parts:
        action_desc = " and ".join(action_parts[:2])

    # ========================
    # 5. Xây dựng mô tả môi trường (Environment Description)
    # ========================
    env_descriptions = {
        "rice_field": "in a rice paddy field, surrounded by green rice plants and rural houses in the distance",
        "outdoor_green": "in a lush green outdoor area with natural vegetation",
        "sky_or_water": "near a body of water under a wide open sky",
        "indoor_or_urban": "in an indoor or urban setting with buildings nearby",
        "bright_outdoor": "in a bright, sunlit outdoor environment",
        "general_outdoor": "in an outdoor setting",
        "general": "",
    }
    env_desc = env_descriptions.get(env, "")

    # ========================
    # 6. Sinh câu chú thích đặc biệt cho cảnh đồng ruộng + người
    # ========================
    if env == "rice_field" and person_count >= 1:
        if person_count >= 3:
            return (f"A group of {person_count} farmers are working together "
                    f"in a vast green rice paddy field, with traditional rural "
                    f"houses and lush vegetation visible in the background.")
        elif person_count == 2:
            return ("Two farmers are harvesting rice in a green rice paddy field, "
                    "surrounded by rows of rice plants with local village houses "
                    "and trees in the background.")
        else:
            return ("A farmer is working alone in a vast rice paddy field, "
                    "bending down among the green rice plants with rural houses "
                    "and natural landscape stretching out in the background.")

    # ========================
    # 7. Xây dựng câu mô tả tổng quát (General Caption)
    # ========================
    parts = []

    # Chủ thể
    if subject_desc:
        parts.append(subject_desc)

    # Hành động
    if action_desc:
        if parts:
            parts[-1] = parts[-1] + " " + action_desc
        else:
            parts.append(action_desc)

    # Các đối tượng khác
    if non_person_entities:
        other_objs = ", ".join(f"a {o}" for o in list(non_person_entities)[:3])
        if parts:
            parts.append(f"with {other_objs} visible in the scene")
        else:
            parts.append(f"{other_objs}")

    # Môi trường
    if env_desc:
        parts.append(env_desc)

    # Tổ hợp câu cuối cùng
    if not parts:
        return "A photograph capturing a scene with various objects and activities."

    caption_body = ", ".join(parts)
    # Viết hoa chữ cái đầu
    return f"A photograph of {caption_body}."


def speak_custom_text(text, lang_name):
    """Sinh file âm thanh TTS cho đoạn văn bản tùy chỉnh được nhập từ người dùng."""
    if not text or not text.strip():
        return None

    lang_code = "vi" if "Vietnamese" in lang_name else "en"

    # Tạo file âm thanh trong thư mục outputs/audio của XLAS
    audio_output_path = get_temp_audio_file()

    audio_path, _ = generate_tts_audio(
        text=text,
        lang=lang_code,
        output_path=audio_output_path
    )
    return audio_path


# Thứ tự ưu tiên các Gemini model (từ nhẹ nhất → mạnh nhất)
# Lite model có quota free tier cao hơn, ít bị giới hạn hơn
GEMINI_MODEL_FALLBACK_ORDER = [
    "gemini-flash-latest",     # Đã kiểm tra và chạy thành công trên tài khoản của bạn
    "gemini-flash-lite-latest",# Đã kiểm tra và chạy thành công trên tài khoản của bạn
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
]


def clean_markdown_text(text):
    """Loại bỏ các ký tự định dạng Markdown (như **, *, #, -, _) để văn bản hiển thị sạch và TTS đọc tự nhiên."""
    import re
    if not text:
        return ""
    # 1. Loại bỏ các dòng tiêu đề hoặc ký tự # ở đầu dòng
    text = re.sub(r'(?m)^#+\s*', '', text)
    # 2. Loại bỏ ký tự list marker như * hoặc - ở đầu dòng
    text = re.sub(r'(?m)^\s*[\*\-]\s+', '', text)
    # 3. Loại bỏ ký hiệu in đậm/in nghiêng **, *, __, _
    text = re.sub(r'\*\*|__', '', text)
    text = re.sub(r'\*|_', '', text)
    # 4. Chuẩn hóa khoảng trắng thừa
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)
    return text.strip()


def call_gemini_with_fallback(gemini_key, prompt_parts, max_retries=2, retry_delay=5):
    """
    Gọi Gemini API với cơ chế fallback model và retry tự động.
    Thử lần lượt từng model trong GEMINI_MODEL_FALLBACK_ORDER.
    Nếu bị lỗi 429 (quota) sẽ chờ và thử lại, hoặc chuyển model khác.

    Args:
        gemini_key: API key Gemini
        prompt_parts: List nội dung truyền vào (text + PIL image)
        max_retries: Số lần thử lại mỗi model khi bị 429
        retry_delay: Số giây chờ giữa mỗi lần retry

    Returns:
        (text, model_used): Nội dung phản hồi và tên model đã dùng thành công
    """
    import time as _time

    client = genai.Client(api_key=gemini_key.strip())

    last_error = None
    for model_name in GEMINI_MODEL_FALLBACK_ORDER:
        for attempt in range(max_retries + 1):
            try:
                # Client riêng cho từng request, tránh chia sẻ API key giữa người dùng.
                response = client.models.generate_content(model=model_name, contents=prompt_parts)
                return response.text.strip(), model_name
            except Exception as e:
                err_str = str(e)
                is_quota = "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower()
                is_not_found = "404" in err_str or "not found" in err_str.lower()

                if is_not_found:
                    # Model không tồn tại, bỏ qua sang model tiếp theo
                    print(f"[Gemini] Model {model_name} không khả dụng, thử model khác...")
                    last_error = e
                    break  # Break khỏi vòng retry, sang model khác

                elif is_quota and attempt < max_retries:
                    wait = retry_delay * (attempt + 1)
                    print(f"[Gemini] {model_name} bị giới hạn quota, chờ {wait}s rồi thử lại (lần {attempt+1}/{max_retries})...")
                    _time.sleep(wait)
                    last_error = e
                    continue  # Thử lại cùng model

                elif is_quota:
                    # Hết lượt retry, thử model nhẹ hơn
                    print(f"[Gemini] {model_name} hết quota sau {max_retries} lần thử, chuyển sang model khác...")
                    last_error = e
                    break

                else:
                    # Lỗi khác (không phải quota), ném ngay
                    raise e

    # Tất cả models đều thất bại
    raise Exception(
        f"❌ Tất cả Gemini models đều không phản hồi được.\n"
        f"Lỗi cuối: {last_error}\n\n"
        f"💡 Gợi ý:\n"
        f"  • Kiểm tra lại API Key tại: https://aistudio.google.com/apikey\n"
        f"  • Tạo API Key mới nếu key hiện tại hết quota miễn phí\n"
        f"  • Hoặc kích hoạt thanh toán tại: https://console.cloud.google.com/billing\n"
        f"  • Hoặc dùng OpenAI GPT-4 Vision thay thế"
    )


def transcribe_audio(audio_path, openai_api_key=None):
    """Nhận diện giọng nói từ file ghi âm sang văn bản.

    Hỗ trợ cả tiếng Anh và tiếng Việt. Ưu tiên OpenAI Whisper nếu có key,
    ngược lại sử dụng Google Speech Recognition miễn phí.
    """
    if not audio_path:
        return ""
    
    try:
        # 1. Sử dụng OpenAI Whisper nếu có API Key
        if openai_api_key and openai_api_key.strip():
            client = OpenAI(api_key=openai_api_key.strip())
            with open(audio_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            return transcript.text.strip()
        
        # 2. Sử dụng thư viện SpeechRecognition (Google API miễn phí)
        r = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            # Điều chỉnh nhiễu
            r.adjust_for_ambient_noise(source)
            audio_data = r.record(source)
        
        # Thử nhận diện bằng tiếng Việt trước, nếu lỗi chuyển sang tiếng Anh
        try:
            return r.recognize_google(audio_data, language="vi-VN").strip()
        except Exception:
            return r.recognize_google(audio_data, language="en-US").strip()
            
    except Exception as e:
        print(f"[SpeechRecognition] Lỗi nhận diện giọng nói: {e}")
        return ""


def run_qa_assistant(image, question_text, question_audio, api_provider, gemini_key, openai_key):
    """Trợ lý Hỏi - Đáp Đa phương thức: Nhận câu hỏi (chữ hoặc giọng nói), phân tích ảnh và trả lời."""
    if image is None:
        return "", "⚠️ Vui lòng tải lên hoặc chụp một bức ảnh.", None

    # 1. Nhận diện câu hỏi từ âm thanh giọng nói nếu có
    transcribed_q = ""
    if question_audio:
        transcribed_q = transcribe_audio(question_audio, openai_key if api_provider == "OpenAI GPT-4 Vision" else None)
    
    # Ưu tiên câu hỏi giọng nói vừa nhận diện, nếu trống thì dùng câu hỏi text
    final_question = transcribed_q if transcribed_q else question_text
    if not final_question or not final_question.strip():
        return "", "⚠️ Vui lòng nhập câu hỏi hoặc ghi âm câu hỏi của bạn.", None

    answer_text = ""
    # Chuyển đổi ảnh sang PIL Image
    if isinstance(image, np.ndarray):
        pil_img = Image.fromarray(image).convert("RGB")
    else:
        pil_img = image.convert("RGB")

    # 2. Xử lý câu hỏi qua mô hình được chọn
    try:
        if api_provider == "Gemini AI" and gemini_key and gemini_key.strip():
            # Sử dụng Gemini API với fallback model tự động khi bị giới hạn quota
            answer_text, model_used = call_gemini_with_fallback(
                gemini_key, [final_question, pil_img]
            )
            print(f"[Q&A] Gemini model đã dùng: {model_used}")
            
        elif api_provider == "OpenAI GPT-4 Vision" and openai_key and openai_key.strip():
            # Sử dụng OpenAI GPT-4 Vision API
            import base64
            from io import BytesIO
            
            buffered = BytesIO()
            pil_img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            client = OpenAI(api_key=openai_key.strip())
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": final_question},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_str}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            answer_text = response.choices[0].message.content.strip()
            
        else:
            # Fallback: Chế độ demo thông minh dựa trên đồ thị ngữ cảnh
            # Trích xuất scene graph cục bộ để làm dữ liệu nền
            triples = []
            if SG_EXTRACTOR is not None:
                triples = SG_EXTRACTOR.extract(pil_img)
            
            entities = set()
            relations = []
            for t in triples:
                sub = t.get("subject", "").lower()
                pred = t.get("predicate", "").lower()
                obj = t.get("object", "").lower()
                if sub: entities.add(sub)
                if obj: entities.add(obj)
                relations.append(f"{sub} {pred} {obj}")
                
            q_lower = final_question.lower()
            if "how many" in q_lower or "mấy" in q_lower or "bao nhiêu" in q_lower:
                if "person" in q_lower or "người" in q_lower:
                    p_count = sum(1 for e in entities if "person" in e or "man" in e or "woman" in e)
                    answer_text = f"Based on local scene analysis, I can detect {max(1, p_count)} person(s) in this image."
                else:
                    detected_matches = [e for e in entities if e in q_lower]
                    if detected_matches:
                        answer_text = f"I detected the object '{detected_matches[0]}' in the image."
                    else:
                        answer_text = "I can see several objects in the scene but cannot count the specific item you asked about without an API key."
            elif "what" in q_lower or "cái gì" in q_lower:
                if entities:
                    answer_text = f"This image mainly contains: {', '.join(list(entities)[:4])}."
                else:
                    answer_text = "I see objects in the image, but I need an API key to describe them in detail."
            else:
                answer_text = (
                    f"💬 [Chế độ Demo/Mô phỏng] Bạn đã hỏi: '{final_question}'\n\n"
                    f"Để có câu trả lời phân tích chuyên sâu nhất, vui lòng nhập API Key của Gemini AI hoặc OpenAI GPT-4 ở mục Cài đặt phía trên.\n"
                    f"Hiện tại, tôi phát hiện các thực thể sau trong ảnh: {', '.join(list(entities)[:5])}."
                )
    except Exception as e:
        answer_text = f"❌ Lỗi khi xử lý câu hỏi với API: {e}"

    # 3. Tạo âm thanh giọng đọc TTS từ câu trả lời
    answer_text = clean_markdown_text(answer_text)
    audio_path = None
    if answer_text:
        # Xác định ngôn ngữ đọc dựa trên câu trả lời
        is_vietnamese = any(char in answer_text for char in "áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ")
        lang = "vi" if is_vietnamese else "en"
        
        audio_output_path = get_temp_audio_file()
        audio_path, _ = generate_tts_audio(answer_text, lang=lang, output_path=audio_output_path)

    return transcribed_q, answer_text, audio_path


def run_ocr_reader(image, api_provider, gemini_key, openai_key):
    """Đọc văn bản ngắn & Tài liệu (OCR) bằng mô hình Vision API."""
    if image is None:
        return "⚠️ Vui lòng tải lên hoặc chụp một bức ảnh chứa tài liệu/chữ viết.", None

    if isinstance(image, np.ndarray):
        pil_img = Image.fromarray(image).convert("RGB")
    else:
        pil_img = image.convert("RGB")

    prompt = "Read and extract all visible text in this image, including signs, document pages, menus, or labels. Return the extracted text accurately."
    ocr_text = ""

    try:
        if api_provider == "Gemini AI" and gemini_key and gemini_key.strip():
            # Sử dụng Gemini API với fallback model tự động khi bị giới hạn quota
            ocr_text, model_used = call_gemini_with_fallback(
                gemini_key, [prompt, pil_img]
            )
            print(f"[OCR] Gemini model đã dùng: {model_used}")
        elif api_provider == "OpenAI GPT-4 Vision" and openai_key and openai_key.strip():
            import base64
            from io import BytesIO
            
            buffered = BytesIO()
            pil_img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            client = OpenAI(api_key=openai_key.strip())
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
                        ]
                    }
                ],
                max_tokens=500
            )
            ocr_text = response.choices[0].message.content.strip()
        else:
            ocr_text = (
                "📝 [Chế độ Demo/Mô phỏng OCR]\n\n"
                "Vui lòng cung cấp API Key của Gemini hoặc OpenAI để quét văn bản.\n"
                " Vision API sẽ đọc cực kỳ chính xác các biển báo tiếng Việt, nhãn chai thuốc hoặc toàn bộ trang tài liệu."
            )
    except Exception as e:
        ocr_text = f"❌ Lỗi khi đọc OCR qua API: {e}"

    ocr_text = clean_markdown_text(ocr_text)
    # Phát âm thanh đọc văn bản quét được
    audio_path = None
    if ocr_text:
        is_vietnamese = any(char in ocr_text for char in "áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ")
        lang = "vi" if is_vietnamese else "en"
        
        audio_output_path = get_temp_audio_file()
        audio_path, _ = generate_tts_audio(ocr_text, lang=lang, output_path=audio_output_path)

    return ocr_text, audio_path


def run_face_analyzer(image, api_provider, gemini_key, openai_key):
    """Nhận diện khuôn mặt cục bộ bằng OpenCV và phân tích tuổi/giới tính/cảm xúc qua Vision API."""
    if image is None:
        return None, "⚠️ Vui lòng cung cấp ảnh đầu vào.", None

    if isinstance(image, np.ndarray):
        img_cv = image.copy()
    else:
        img_cv = np.array(image)

    # 1. Phát hiện khuôn mặt bằng Haar Cascade của OpenCV (Cục bộ & Nhanh chóng)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Vẽ khung hình chữ nhật quanh khuôn mặt phát hiện được
    for (x, y, w, h) in faces:
        cv2.rectangle(img_cv, (x, y), (x+w, y+h), (22, 163, 74), 3) # Màu xanh lá cây

    n_faces = len(faces)
    face_details = ""
    
    # 2. Phân tích chi tiết khuôn mặt bằng Vision API
    if n_faces > 0:
        pil_img = Image.fromarray(img_cv).convert("RGB")
        prompt = (
            f"There are {n_faces} faces detected in this image. "
            "Describe visible facial expressions for each detected face. If suggesting an emotion, clearly mark it as uncertain and based only on visible expression. Do not infer identity or gender identity. "
            "Respond briefly in Vietnamese, structured for each person."
        )
        
        try:
            if api_provider == "Gemini AI" and gemini_key and gemini_key.strip():
                # Sử dụng Gemini API với fallback model tự động khi bị giới hạn quota
                face_details, model_used = call_gemini_with_fallback(
                    gemini_key, [prompt, pil_img]
                )
                print(f"[Face] Gemini model đã dùng: {model_used}")
            elif api_provider == "OpenAI GPT-4 Vision" and openai_key and openai_key.strip():
                import base64
                from io import BytesIO
                
                buffered = BytesIO()
                pil_img.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                
                client = OpenAI(api_key=openai_key.strip())
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
                            ]
                        }
                    ],
                    max_tokens=300
                )
                face_details = response.choices[0].message.content.strip()
            else:
                face_details = (
                    f"👤 [Chế độ Demo] Phát hiện thành công {n_faces} khuôn mặt cục bộ trong ảnh.\n\n"
                    "Để mô tả biểu cảm khuôn mặt bằng AI, vui lòng cung cấp API Key ở mục Cài đặt.\n"
                    "Kết quả AI là ước đoán từ biểu cảm nhìn thấy, không xác định chắc chắn cảm xúc thực tế."
                )
        except Exception as e:
            face_details = f"👤 Phát hiện {n_faces} khuôn mặt. Lỗi gọi API phân tích cảm xúc: {e}"
    else:
        face_details = "❌ Không tìm thấy khuôn mặt nào trong bức ảnh."

    face_details = clean_markdown_text(face_details)
    # Phát âm thanh đọc mô tả cảm xúc
    audio_path = None
    if face_details:
        is_vietnamese = any(char in face_details for char in "áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ")
        lang = "vi" if is_vietnamese else "en"
        
        audio_output_path = get_temp_audio_file()
        audio_path, _ = generate_tts_audio(face_details, lang=lang, output_path=audio_output_path)

    return img_cv, face_details, audio_path


def process_single_image(pil_image):
    """Xử lý một ảnh PIL đơn lẻ qua pipeline mô hình.

    Returns:
        Tuple (visual_features_tensor, triples_tensor, triple_mask_tensor, triples_raw)
    """
    if SG_EXTRACTOR is None:
        return None, None, None, []

    # Trích xuất Scene Graph
    triples_raw = SG_EXTRACTOR.extract(pil_image)

    # Mã hóa bộ ba
    encoded_triples, triple_mask = encode_triples(
        triples=triples_raw,
        vocab=VOCAB,
        max_triples=CONFIG.model.max_triples
    )
    triples_tensor = torch.tensor(encoded_triples, dtype=torch.long).unsqueeze(0).to(DEVICE)
    triple_mask_tensor = torch.tensor(triple_mask, dtype=torch.bool).unsqueeze(0).to(DEVICE)

    # Tiền xử lý ảnh
    img_tensor = TRANSFORMS(pil_image).unsqueeze(0).to(DEVICE)

    return img_tensor, triples_tensor, triple_mask_tensor, triples_raw


def generate_caption_from_video(video_path, num_frames=8, beam_size=5):
    """Hàm chính: Sinh chú thích cho video và tạo file âm thanh TTS.

    Pipeline:
        1. Trích xuất N khung hình cách đều từ video.
        2. Trích xuất đặc trưng thị giác từ tất cả khung hình (ResNet-101).
        3. Tính trung bình cộng (Temporal Average Pooling) đặc trưng thị giác.
        4. Trích xuất Scene Graph từ khung hình đại diện (keyframe).
        5. Hợp nhất đặc trưng và sinh chú thích bằng Beam Search (hoặc mô phỏng từ Scene Graph nếu chưa nạp checkpoint).
        6. Chuyển chú thích thành giọng nói (TTS).

    Args:
        video_path: Đường dẫn tới file video.
        num_frames: Số khung hình cần lấy mẫu.
        beam_size: Độ rộng beam search.

    Returns:
        Tuple (caption_text, audio_path, keyframe_image, visualized_keyframe, scene_graph_text, processing_info, timeline_text)
    """
    if video_path is None:
        return "⚠️ Vui lòng tải lên một đoạn video.", None, None, None, "", "", ""

    start_time = time.time()
    num_frames = int(num_frames)
    beam_size = int(beam_size)

    # ========================
    # Bước 1: Trích xuất khung hình từ video
    # ========================
    try:
        frames, keyframe_idx, duration = extract_frames(video_path, num_frames=num_frames)
    except Exception as e:
        return f"❌ Lỗi khi đọc video: {str(e)}", None, None, None, "", "", ""

    keyframe = frames[keyframe_idx]
    n_extracted = len(frames)

    # ========================
    # Bước 2: Trích xuất Scene Graph từ keyframe
    # ========================
    triples_raw = []
    if SG_EXTRACTOR is not None:
        triples_raw = SG_EXTRACTOR.extract(keyframe)

    # ========================
    # Bước 3: Sinh chú thích (Transformer hoặc Mô phỏng từ Scene Graph)
    # ========================
    if MODEL is not None:
        # Cách tiếp cận mạng nơ-ron đầy đủ (khi có checkpoint)
        with torch.no_grad():
            all_visual_feats = []
            for frame in frames:
                img_tensor = TRANSFORMS(frame).unsqueeze(0).to(DEVICE)
                visual_feats, visual_mask = MODEL.visual_encoder(img_tensor)
                all_visual_feats.append(visual_feats)

            stacked = torch.cat(all_visual_feats, dim=0)  # (N, 49, d_model)
            pooled_visual_feats = stacked.mean(dim=0, keepdim=True)  # (1, 49, d_model)

            pooled_visual_mask = torch.zeros(
                1, pooled_visual_feats.size(1), dtype=torch.bool, device=DEVICE
            )

            # Mã hóa các bộ ba scene graph
            encoded_triples, triple_mask = encode_triples(
                triples=triples_raw,
                vocab=VOCAB,
                max_triples=CONFIG.model.max_triples
            )
            triples_tensor = torch.tensor(encoded_triples, dtype=torch.long).unsqueeze(0).to(DEVICE)
            triple_mask_tensor = torch.tensor(triple_mask, dtype=torch.bool).unsqueeze(0).to(DEVICE)

            # Trích xuất đặc trưng ngữ nghĩa
            semantic_feats, semantic_mask = MODEL.semantic_encoder(triples_tensor, triple_mask_tensor)

            # Hợp nhất đặc trưng thị giác và ngữ nghĩa
            fused_feats, fused_mask = MODEL.fusion_module(
                pooled_visual_feats, semantic_feats, pooled_visual_mask, semantic_mask
            )

            # Sinh chú thích bằng Beam Search
            caps_pred, scores = MODEL.caption_decoder.generate(
                fused_features=fused_feats,
                fused_mask=fused_mask,
                max_len=CONFIG.inference.max_len,
                beam_size=beam_size,
                start_idx=VOCAB.word2idx["<start>"],
                end_idx=VOCAB.word2idx["<end>"]
            )

        caption_text = VOCAB.decode(caps_pred[0].cpu().tolist())
        score_value = scores[0].item()
        mode_desc = "Mô hình Transformer (Visual-Semantic Fusion)"
    else:
        # Chế độ Mô phỏng / Demo (khi chưa có checkpoint mô hình)
        caption_text = generate_simulated_caption(triples_raw, keyframe)
        score_value = 0.0
        mode_desc = "Mô phỏng (Suy luận trực tiếp từ Scene Graph + BLIP)"

    # ========================
    # Bước 4: Chuyển chú thích thành giọng nói (TTS)
    # ========================
    audio_output_path = get_temp_audio_file()
    audio_path, tts_engine = generate_tts_audio(
        text=caption_text,
        lang="en",
        output_path=audio_output_path
    )

    # ========================
    # Bước 5: Phân tích dòng thời gian video (Timeline Analysis)
    # ========================
    timeline_text = ""
    if len(frames) >= 3:
        time_step = duration / (len(frames) - 1) if len(frames) > 1 else 0.0
        t_start = 0.0
        t_mid = (len(frames) // 2) * time_step
        t_end = duration

        # Trích xuất caption nhanh cho 3 khung hình chính
        cap_start = generate_blip_caption(frames[0], detail_level="low")
        cap_mid = generate_blip_caption(frames[len(frames) // 2], detail_level="low")
        cap_end = generate_blip_caption(frames[-1], detail_level="low")

        if not cap_start:
            cap_start = generate_simulated_caption([], frames[0])
        if not cap_mid:
            cap_mid = generate_simulated_caption([], frames[len(frames) // 2])
        if not cap_end:
            cap_end = generate_simulated_caption([], frames[-1])

        timeline_text = (
            f"⏱️ **0.0s (Bắt đầu):** {cap_start}\n\n"
            f"⏱️ **{t_mid:.1f}s (Giữa):** {cap_mid}\n\n"
            f"⏱️ **{t_end:.1f}s (Kết thúc):** {cap_end}"
        )
    else:
        timeline_text = "⏱️ *Không đủ số lượng khung hình để thực hiện phân tích timeline.*"

    # ========================
    # Tổng hợp thông tin xử lý
    # ========================
    elapsed = time.time() - start_time

    # Chuẩn bị chuỗi hiển thị Scene Graph
    sg_lines = []
    for i, t in enumerate(triples_raw[:15]):
        sg_lines.append(
            f"  {i+1}. ({t['subject']}, **{t['predicate']}**, {t['object']}) "
            f"— conf: {t.get('confidence', 0):.2f}"
        )
    sg_text = "\n".join(sg_lines) if sg_lines else "Không trích xuất được mối quan hệ nào."

    info_text = (
        f"📊 **Thông tin xử lý:**\n"
        f"- Chế độ chạy: **{mode_desc}**\n"
        f"- Thời lượng video: **{duration:.1f}s**\n"
        f"- Số khung hình đã trích xuất: **{n_extracted}**\n"
        f"- Điểm log-probability: **{score_value:.4f}**\n"
        f"- Công cụ TTS: **{tts_engine or 'N/A'}**\n"
        f"- Thời gian xử lý: **{elapsed:.2f}s**"
    )

    # Chuyển đổi keyframe sang hình trực quan vẽ bounding box
    keyframe_visualized = draw_scene_graph_on_image(keyframe, triples_raw)
    
    keyframe_np = np.array(keyframe)
    keyframe_visualized_np = np.array(keyframe_visualized)

    return caption_text, audio_path, keyframe_np, keyframe_visualized_np, sg_text, info_text, timeline_text


def generate_caption_from_image(image, beam_size=5):
    """Hàm xử lý cho chế độ Image Captioning (giữ nguyên tính năng gốc).

    Args:
        image: Ảnh PIL hoặc numpy array từ Gradio.
        beam_size: Độ rộng beam search.

    Returns:
        Tuple (caption_text, audio_path, visualized_image_np, scene_graph_text, processing_info)
    """
    if image is None:
        return "⚠️ Vui lòng tải lên một bức ảnh.", None, None, "", ""

    start_time = time.time()
    beam_size = int(beam_size)

    # Chuyển numpy array sang PIL nếu cần
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image).convert("RGB")
    else:
        pil_image = image.convert("RGB")

    # Trích xuất Scene Graph
    triples_raw = []
    if SG_EXTRACTOR is not None:
        triples_raw = SG_EXTRACTOR.extract(pil_image)

    # Sinh chú thích (Transformer hoặc Mô phỏng từ Scene Graph)
    if MODEL is not None:
        # Xử lý qua pipeline của mô hình thật
        img_tensor, triples_tensor, triple_mask_tensor, _ = process_single_image(pil_image)

        with torch.no_grad():
            caps_pred, scores = MODEL.generate(
                images=img_tensor,
                triples=triples_tensor,
                triple_mask=triple_mask_tensor,
                max_len=CONFIG.inference.max_len,
                beam_size=beam_size,
                start_idx=VOCAB.word2idx["<start>"],
                end_idx=VOCAB.word2idx["<end>"]
            )
        caption_text = VOCAB.decode(caps_pred[0].cpu().tolist())
        score_value = scores[0].item()
        mode_desc = "Mô hình Transformer (Visual-Semantic Fusion)"
    else:
        # Chế độ Mô phỏng / Demo (khi chưa có checkpoint mô hình)
        caption_text = generate_simulated_caption(triples_raw, pil_image)
        score_value = 0.0
        mode_desc = "Mô phỏng (Suy luận trực tiếp từ Scene Graph + BLIP)"

    # TTS
    audio_output_path = get_temp_audio_file()
    audio_path, tts_engine = generate_tts_audio(
        text=caption_text, lang="en", output_path=audio_output_path
    )

    elapsed = time.time() - start_time

    # Scene Graph text
    sg_lines = []
    for i, t in enumerate(triples_raw[:15]):
        sg_lines.append(
            f"  {i+1}. ({t['subject']}, **{t['predicate']}**, {t['object']}) "
            f"— conf: {t.get('confidence', 0):.2f}"
        )
    sg_text = "\n".join(sg_lines) if sg_lines else "Không trích xuất được mối quan hệ nào."

    info_text = (
        f"📊 **Thông tin xử lý:**\n"
        f"- Chế độ chạy: **{mode_desc}**\n"
        f"- Beam Search size: **{beam_size}**\n"
        f"- Điểm log-probability: **{score_value:.4f}**\n"
        f"- Công cụ TTS: **{tts_engine or 'N/A'}**\n"
        f"- Thời gian xử lý: **{elapsed:.2f}s**"
    )

    # Tạo ảnh vẽ Scene Graph trực quan
    visualized_image = draw_scene_graph_on_image(pil_image, triples_raw)
    visualized_image_np = np.array(visualized_image)

    return caption_text, audio_path, visualized_image_np, sg_text, info_text



