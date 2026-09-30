"""Gói hỗ trợ trích xuất và xử lý Scene Graph (Đồ thị ngữ cảnh) từ ảnh.

Export các lớp và hàm chính để làm việc với Scene Graph:
- SceneGraphExtractor: Lớp cơ sở trích xuất đồ thị ngữ cảnh từ ảnh thô.
- create_extractor: Hàm factory tạo bộ trích xuất theo thuật toán lựa chọn.
- load_scene_graphs, save_scene_graphs: Hàm đọc/ghi danh sách đồ thị ngữ cảnh.
- filter_triples, encode_triples: Các tiện ích tiền xử lý bộ ba ngữ nghĩa.
"""

from src.scene_graph.sg_utils import (
    load_scene_graphs,
    save_scene_graphs,
    filter_triples,
    encode_triples,
    get_predicate_statistics,
    get_object_statistics,
    visualize_scene_graph,
    triple_to_text,
)

from src.scene_graph.extract_sg import (
    SceneGraphExtractor,
    RelTRExtractor,
    SimpleObjectDetectorSGG,
    create_extractor,
)

__all__ = [
    "SceneGraphExtractor",
    "RelTRExtractor",
    "SimpleObjectDetectorSGG",
    "create_extractor",
    "load_scene_graphs",
    "save_scene_graphs",
    "filter_triples",
    "encode_triples",
    "get_predicate_statistics",
    "get_object_statistics",
    "visualize_scene_graph",
    "triple_to_text",
]
