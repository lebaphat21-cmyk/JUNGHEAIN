import os
import cv2
from PIL import Image

def extract_frames(video_path, num_frames=8):
    """Trích xuất các khung hình cách đều nhau từ một file video.

    Args:
        video_path: Đường dẫn tới file video đầu vào.
        num_frames: Số lượng khung hình cần trích xuất.

    Returns:
        Tuple chứa:
            - **frames**: Danh sách các PIL Image đã được trích xuất.
            - **keyframe_idx**: Vị trí của khung hình đại diện (keyframe) ở giữa.
            - **duration**: Thời lượng của video (giây).
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Không tìm thấy file video tại: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Không thể mở file video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0.0

    if total_frames <= 0:
        cap.release()
        raise ValueError("Video không chứa khung hình nào hợp lệ.")

    # Tính toán chỉ số các khung hình cần lấy mẫu cách đều nhau
    if num_frames >= total_frames:
        indices = list(range(total_frames))
    else:
        # Lấy đều từ đầu đến cuối video
        indices = [int(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)] if num_frames > 1 else [total_frames // 2]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            # Thử đọc khung hình tiếp theo nếu lỗi
            ret, frame = cap.read()
            if not ret:
                continue
        # Chuyển BGR sang RGB và chuyển thành PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))

    cap.release()

    if not frames:
        raise ValueError("Không thể trích xuất được khung hình nào từ video.")

    # Chọn khung hình ở giữa danh sách làm keyframe để trích xuất scene graph
    keyframe_idx = len(frames) // 2

    return frames, keyframe_idx, duration


def generate_tts_audio(text, lang="en", output_path="outputs/results/output_audio.mp3"):
    """Chuyển văn bản thành giọng nói và lưu thành file âm thanh.

    Cố gắng sử dụng gTTS (yêu cầu Internet) để có giọng đọc chất lượng cao.
    Nếu thất bại, tự động chuyển sang pyttsx3 (chạy offline).

    Args:
        text: Văn bản cần đọc.
        lang: Mã ngôn ngữ ('en' hoặc 'vi').
        output_path: Đường dẫn lưu file âm thanh đầu ra.

    Returns:
        Tuple (output_path, tts_engine_name) hoặc (None, None) nếu thất bại.
    """
    # Tạo thư mục chứa file đầu ra nếu chưa có
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # 1. Thử dùng gTTS (online)
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(output_path)
        return output_path, "gTTS (online)"
    except Exception as e:
        print(f"[gTTS Warning] Không thể sử dụng gTTS (có thể do offline hoặc lỗi kết nối): {e}")
        print("Đang chuyển hướng sang công cụ pyttsx3 (offline)...")

    # 2. Dự phòng bằng pyttsx3 (offline)
    try:
        import pyttsx3
        # Khởi tạo engine pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)  # Tốc độ đọc
        
        # Thiết lập ngôn ngữ nếu được hỗ trợ
        # Thư viện pyttsx3 dùng giọng đọc cài sẵn của hệ điều hành
        voices = engine.getProperty("voices")
        selected_voice = None
        for voice in voices:
            if lang == "vi" and "vietnam" in voice.name.lower():
                selected_voice = voice.id
                break
            elif lang == "en" and "english" in voice.name.lower():
                selected_voice = voice.id
                break
        
        if selected_voice:
            engine.setProperty("voice", selected_voice)

        engine.save_to_file(text, output_path)
        engine.runAndWait()
        
        # Một số phiên bản pyttsx3 trên Windows có thể sinh file dạng .wav hoặc không tự đóng file ngay lập tức,
        # nhưng cơ bản là lưu ra đúng output_path đã truyền.
        return output_path, "pyttsx3 (offline)"
    except Exception as e:
        print(f"[TTS Error] Lỗi nghiêm trọng khi chuyển đổi giọng nói bằng pyttsx3: {e}")
        return None, None
