"""XLAS: giao diện riêng, kế thừa pipeline từ web_demo.py."""
import argparse
import os
from pathlib import Path

import gradio as gr
import backend

ROOT = Path(__file__).resolve().parent.parent
XLAS_DIR = Path(__file__).resolve().parent
css_path = XLAS_DIR / "styles.css"
CSS = css_path.read_text(encoding="utf-8") if css_path.is_file() else ""
SAMPLES_DIR = XLAS_DIR / "data" / "samples"
SAMPLE_IMAGES = sorted([str(p) for p in SAMPLES_DIR.glob("*.jpg")]) if SAMPLES_DIR.exists() else []


def make_theme():
    theme = gr.themes.Soft(primary_hue="teal", neutral_hue="slate", font=["Segoe UI", "Arial", "sans-serif"])
    theme.set(body_background_fill="#f3f6fa", block_background_fill="#ffffff",
              body_text_color="#15263d", body_text_color_subdued="#53657b",
              block_border_color="#e0e7ef", input_background_fill="#f8fafc",
              button_primary_background_fill="#087f75", button_primary_text_color="#ffffff",
              button_primary_background_fill_hover="#06685f")
    # Giữ bảng màu sáng nhất quán cả khi trình duyệt đang bật dark mode.
    values = theme.to_dict()["theme"]
    theme.set(**{key: values[key[:-5]] for key in values if key.endswith("_dark") and key[:-5] in values})
    return theme


def answer_question(image, text, audio, provider, gemini_key, openai_key):
    if image is None or not (text or audio):
        return backend.run_qa_assistant(image, text, audio, provider, gemini_key, openai_key)
    if provider != "Demo cục bộ":
        key = gemini_key if provider == "Gemini AI" else openai_key
        if not key or not key.strip():
            return "", "Vui lòng nhập API key của AI đã chọn trong Kết nối AI.", None
    result = backend.run_qa_assistant(image, text, audio, provider, gemini_key, openai_key)
    if provider == "Demo cục bộ":
        return result[0], "[Demo mô phỏng — không phải câu trả lời AI đã xác minh]\n" + result[1], result[2]
    return result


def panel_heading(number, title, subtitle):
    gr.HTML(f'<div class="panel-heading"><span class="step">{number}</span><div><h3>{title}</h3><p>{subtitle}</p></div></div>')


def section_heading(tag, title, description):
    gr.HTML(f'<div class="section-heading"><span class="section-tag">{tag}</span><h2>{title}</h2><p>{description}</p></div>')


def build_app():
    with gr.Blocks(title="XLAS · AI Studio") as app:
        gr.HTML('''<header class="topbar"><div class="brand"><span class="brand-icon">x</span><strong>XLAS<span> / AI STUDIO</span></strong></div><span class="topbar-note">Một góc nhìn, nhiều khám phá.</span></header>
        <section class="welcome"><div><div class="eyebrow">KHÔNG GIAN SÁNG TẠO VỚI AI</div><h1>Thấy nhiều hơn.<br><span>Hiểu rõ hơn.</span></h1><p>Từ hình ảnh đến lời nói — khám phá, đặt câu hỏi<br class="desktop-break"> và lắng nghe trong một không gian.</p></div><div class="welcome-art" aria-hidden="true"><div class="orbit orbit-one"></div><div class="orbit orbit-two"></div><div class="art-tile tile-image">▧</div><div class="art-tile tile-sound">▂ ▅ ▇ ▃ ▆ ▂</div><div class="art-tile tile-spark">✦</div><span class="art-label">IMAGE · VOICE · EXPRESSION</span></div></section>''')
        with gr.Accordion("Kết nối AI · Nhập API key tại đây", open=False, elem_id="ai-settings"):
            gr.Markdown("Dùng Gemini hoặc OpenAI cho hỏi đáp và mô tả biểu cảm. Key chỉ dùng trong phiên làm việc; ảnh và câu hỏi được gửi đến nhà cung cấp bạn chọn khi xử lý.")
            with gr.Row():
                gemini = gr.Textbox(label="Gemini API key", type="password", placeholder="Nhập Gemini API key…")
                openai = gr.Textbox(label="OpenAI API key", type="password", placeholder="Nhập OpenAI API key…")
        providers = ["Gemini AI", "OpenAI GPT-4 Vision", "Demo cục bộ"]
        with gr.Tabs(elem_id="workspace-tabs"):
            with gr.Tab("01  Hình ảnh", id="image"):
                section_heading("IMAGE CAPTIONING", "Một bức ảnh, một câu chuyện.", "Tạo chú thích, khám phá nội dung và nghe ảnh được kể bằng lời.")
                with gr.Row(equal_height=False, elem_classes="workspace-row"):
                    with gr.Column(scale=1, min_width=300, elem_classes="studio-panel"):
                        panel_heading("01", "Ảnh của bạn", "Tải lên, dán ảnh hoặc sử dụng camera")
                        picture = gr.Image(label="Ảnh đầu vào", show_label=False, type="numpy", sources=["upload", "webcam", "clipboard"], height=300, elem_classes="image-input")
                        with gr.Accordion("Tùy chỉnh nâng cao", open=False, elem_classes="subtle-accordion"):
                            beam = gr.Slider(1, 10, value=5, step=1, label="Beam Search Size", info="Số phương án mô hình cân nhắc khi tạo chú thích.")
                        caption_button = gr.Button("Tạo chú thích  →", variant="primary", size="lg")
                        clear_image = gr.ClearButton(value="Làm mới", size="sm")
                        if SAMPLE_IMAGES:
                            gr.Examples(examples=[[p] for p in SAMPLE_IMAGES], inputs=picture, label="Ảnh mẫu thử nghiệm (data/samples)")
                    with gr.Column(scale=1, min_width=300, elem_classes="studio-panel result-panel"):
                        panel_heading("02", "Khám phá kết quả", "Đọc, chỉnh sửa và nghe chú thích của bạn")
                        with gr.Tabs(elem_classes="result-tabs"):
                            with gr.Tab("Chú thích"):
                                caption = gr.Textbox(label="Nội dung chú thích", placeholder="Câu chuyện của bức ảnh sẽ xuất hiện ở đây…", lines=5, interactive=True, buttons=["copy"], elem_classes="result-text")
                                with gr.Row(elem_classes="playback-row"):
                                    language = gr.Dropdown(["Tiếng Anh (English)", "Tiếng Việt (Vietnamese)"], value="Tiếng Anh (English)", label="Ngôn ngữ đọc", scale=2, min_width=160)
                                    speak = gr.Button("Đọc lại", scale=1, min_width=100)
                                caption_audio = gr.Audio(label="Nghe chú thích", type="filepath", interactive=False)
                            with gr.Tab("Scene Graph"):
                                graph_image = gr.Image(label="Các mối quan hệ trong ảnh", height=280, interactive=False)
                                graph = gr.Markdown("Các mối quan hệ sẽ xuất hiện sau khi xử lý ảnh.")
                        with gr.Accordion("Chi tiết xử lý", open=False, elem_classes="subtle-accordion"):
                            info = gr.Markdown("Chưa có kết quả xử lý.")
                        if backend.MODEL is None and backend.BLIP_MODEL is None:
                            gr.HTML('<div class="mode-note"><span>i</span> Chế độ mô phỏng · Chưa tải mô hình chú thích ảnh.</div>')
                image_outputs = [caption, caption_audio, graph_image, graph, info]
                caption_button.click(backend.generate_caption_from_image, [picture, beam], image_outputs)
                speak.click(backend.speak_custom_text, [caption, language], caption_audio)
                clear_image.add([picture, *image_outputs])
            with gr.Tab("02  Hỏi đáp", id="voice"):
                section_heading("VOICE Q&A ASSISTANT", "Bạn hỏi. AI cùng khám phá.", "Hỏi về hình ảnh bằng văn bản hoặc giọng nói tiếng Việt, tiếng Anh.")
                with gr.Row(equal_height=False, elem_classes="workspace-row"):
                    with gr.Column(scale=1, min_width=300, elem_classes="studio-panel"):
                        panel_heading("01", "Ảnh & câu hỏi", "Chọn ảnh, rồi đặt điều bạn muốn biết")
                        qa_image = gr.Image(label="Ảnh để hỏi đáp", show_label=False, type="numpy", sources=["upload", "webcam"], height=240, elem_classes="image-input")
                        qa_provider = gr.Dropdown(providers, value="Gemini AI", label="Trợ lý AI")
                        question = gr.Textbox(label="Câu hỏi của bạn", placeholder="Ví dụ: Hãy mô tả những gì có trong ảnh này…", lines=2)
                        with gr.Accordion("Dùng giọng nói thay cho bàn phím", open=False, elem_classes="subtle-accordion"):
                            question_audio = gr.Audio(label="Ghi âm hoặc tải câu hỏi", sources=["microphone", "upload"], type="filepath", format="wav")
                            gr.Markdown("Nếu có ghi âm, trợ lý sẽ ưu tiên câu hỏi trong bản ghi.")
                        ask = gr.Button("Gửi câu hỏi  →", variant="primary", size="lg")
                        clear_qa = gr.ClearButton(value="Cuộc hỏi đáp mới", size="sm")
                        if SAMPLE_IMAGES:
                            gr.Examples(examples=[[p] for p in SAMPLE_IMAGES[:2]], inputs=qa_image, label="Ảnh mẫu thử nghiệm")
                    with gr.Column(scale=1, min_width=300, elem_classes="studio-panel result-panel"):
                        panel_heading("02", "Lời giải đáp", "Thông tin từ hình ảnh, theo câu hỏi của bạn")
                        answer = gr.Textbox(label="Câu trả lời", placeholder="Chọn một bức ảnh và đặt câu hỏi để bắt đầu cuộc trò chuyện.", lines=10, interactive=False, buttons=["copy"], elem_classes="result-text")
                        answer_audio = gr.Audio(label="Nghe câu trả lời", type="filepath", interactive=False)
                        with gr.Accordion("Văn bản nhận diện từ ghi âm", open=False, elem_classes="subtle-accordion"):
                            transcript = gr.Textbox(label="Câu hỏi đã nghe", interactive=False, lines=2)
                ask.click(answer_question, [qa_image, question, question_audio, qa_provider, gemini, openai], [transcript, answer, answer_audio])
                clear_qa.add([qa_image, question, question_audio, transcript, answer, answer_audio])
            with gr.Tab("03  Khuôn mặt", id="face"):
                section_heading("FACE EMOTION ANALYZER", "Quan sát từng biểu cảm.", "Phát hiện khuôn mặt và khám phá những biểu cảm nhìn thấy trong ảnh.")
                with gr.Row(equal_height=False, elem_classes="workspace-row"):
                    with gr.Column(scale=1, min_width=300, elem_classes="studio-panel"):
                        panel_heading("01", "Ảnh chân dung", "Ảnh rõ nét, đủ sáng cho kết quả tốt hơn")
                        face_image = gr.Image(label="Ảnh khuôn mặt", show_label=False, type="numpy", sources=["upload", "webcam"], height=300, elem_classes="image-input")
                        face_provider = gr.Dropdown(providers, value="Gemini AI", label="AI phân tích")
                        analyze = gr.Button("Phân tích biểu cảm  →", variant="primary", size="lg")
                        clear_face = gr.ClearButton(value="Làm mới", size="sm")
                        if SAMPLE_IMAGES:
                            gr.Examples(examples=[[p] for p in SAMPLE_IMAGES], inputs=face_image, label="Ảnh mẫu thử nghiệm")
                    with gr.Column(scale=1, min_width=300, elem_classes="studio-panel result-panel"):
                        panel_heading("02", "Bức tranh biểu cảm", "Khuôn mặt được phát hiện và mô tả từ AI")
                        with gr.Tabs(elem_classes="result-tabs"):
                            with gr.Tab("Phân tích"):
                                details = gr.Textbox(label="Mô tả biểu cảm", placeholder="Kết quả phân tích sẽ xuất hiện khi bạn gửi ảnh.", lines=7, interactive=False, buttons=["copy"], elem_classes="result-text")
                                face_audio = gr.Audio(label="Nghe kết quả", type="filepath", interactive=False)
                            with gr.Tab("Vùng khuôn mặt"):
                                detected = gr.Image(label="Khuôn mặt được phát hiện", height=300, interactive=False)
                        gr.HTML('<div class="mode-note"><span>i</span> Biểu cảm là gợi ý, không khẳng định cảm xúc thực tế.</div>')
                analyze.click(backend.run_face_analyzer, [face_image, face_provider, gemini, openai], [detected, details, face_audio])
                clear_face.add([face_image, detected, details, face_audio])
        gr.HTML('<div class="studio-footer"><span><b>XLAS</b> · Designed for discovery</span><span>Hình ảnh · Giọng nói · Biểu cảm</span></div>')
    return app


def main():
    parser = argparse.ArgumentParser(description="XLAS AI Studio")
    parser.add_argument("--checkpoint", default="auto")
    parser.add_argument("--config", default="configs/base_config.yaml")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sg_method", choices=["heuristic", "reltr"], default="heuristic")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--lightweight", action="store_true", help="Bỏ qua tải mô hình; caption chỉ mô phỏng.")
    args = parser.parse_args()
    os.chdir(ROOT)
    if not args.lightweight:
        backend.load_model_global(args.checkpoint, args.config, args.sg_method, args.gpu)
    build_app().queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", server_port=args.port, theme=make_theme(), css=CSS,
        inbrowser=False, show_error=True,
    )


if __name__ == "__main__":
    main()
