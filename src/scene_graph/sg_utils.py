"""Các hàm tiện ích phục vụ xử lý và trực quan hóa Đồ thị ngữ cảnh (Scene Graph).

Cung cấp các công cụ cần thiết cho việc lọc độ tin cậy, chuyển đổi bộ ba từ dạng chữ
sang dạng chỉ số trong từ điển, thống kê tần suất xuất hiện và vẽ đồ thị minh họa.
"""

from __future__ import annotations

import json
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np


def load_scene_graphs(filepath: str) -> Dict[int, List[Dict[str, Any]]]:
    """Tải dữ liệu Scene Graph đã được trích xuất sẵn từ file JSON.

    Args:
        filepath: Đường dẫn tới file JSON chứa scene graphs.

    Returns:
        Dictionary ánh xạ từ image_id (int) sang danh sách các bộ ba (List[Dict]).
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Chuyển đổi khóa string trong JSON thành int cho image_id
    scene_graphs = {}
    for k, v in data.items():
        try:
            scene_graphs[int(k)] = v
        except ValueError:
            scene_graphs[k] = v
            
    return scene_graphs


def save_scene_graphs(
    scene_graphs: Dict[Any, List[Dict[str, Any]]], filepath: str
) -> None:
    """Lưu trữ dữ liệu Scene Graph của các ảnh vào file JSON.

    Args:
        scene_graphs: Dictionary chứa thông tin đồ thị ngữ cảnh các ảnh.
        filepath: Đường dẫn lưu file JSON.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(scene_graphs, f, ensure_ascii=False, indent=2)


def filter_triples(
    triples: List[Dict[str, Any]],
    min_confidence: float = 0.3,
    max_triples: int = 20,
) -> List[Dict[str, Any]]:
    """Lọc danh sách bộ ba dựa trên ngưỡng độ tin cậy và giới hạn số lượng tối đa.

    Args:
        triples: Danh sách các bộ ba thô dạng dict.
        min_confidence: Ngưỡng độ tin cậy tối thiểu (mặc định 0.3).
        max_triples: Số lượng bộ ba tối đa được giữ lại (mặc định 20).

    Returns:
        Danh sách các bộ ba đã được lọc và sắp xếp theo độ tin cậy giảm dần.
    """
    # Sắp xếp các bộ ba theo độ tin cậy giảm dần nếu có thông tin confidence
    sorted_triples = sorted(
        triples, key=lambda x: x.get("confidence", 1.0), reverse=True
    )

    # Lọc theo ngưỡng độ tin cậy
    filtered = [t for t in sorted_triples if t.get("confidence", 1.0) >= min_confidence]

    # Giới hạn số lượng bộ ba tối đa
    return filtered[:max_triples]


def encode_triples(
    triples: List[Dict[str, Any]],
    vocab: Any,
    max_triples: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mã hóa các bộ ba dạng chuỗi văn bản thành chỉ số trong từ điển.

    Mỗi bộ ba (subject, predicate, object) được ánh xạ thành bộ ba chỉ số nguyên.
    Nếu số bộ ba ít hơn max_triples, thực hiện đệm (padding) bằng giá trị 0 (<pad>).
    Nếu nhiều hơn, thực hiện cắt bỏ.

    Args:
        triples: Danh sách các bộ ba thô.
        vocab: Đối tượng từ điển (Vocabulary) dùng để tra cứu chỉ số.
        max_triples: Số lượng bộ ba tối đa sau khi đệm (mặc định 20).

    Returns:
        Tuple chứa:
            - **encoded**: Mảng numpy kích thước (max_triples, 3) chứa chỉ số từ điển.
            - **mask**: Mảng boolean kích thước (max_triples,) với True biểu thị vị trí pad.
    """
    encoded = np.zeros((max_triples, 3), dtype=np.int64)  # Mặc định đệm là 0 (<pad>)
    mask = np.ones(max_triples, dtype=bool)  # Mặc định tất cả đều là mask (True)

    for i, t in enumerate(triples[:max_triples]):
        # Lấy nhãn của đối tượng và quan hệ
        subj = t.get("subject", "<pad>")
        pred = t.get("predicate", "<pad>")
        obj = t.get("object", "<pad>")

        # Tra cứu chỉ số trong từ điển
        subj_idx = vocab.word2idx.get(str(subj).lower(), vocab.word2idx.get("<unk>", 3))
        pred_idx = vocab.word2idx.get(str(pred).lower(), vocab.word2idx.get("<unk>", 3))
        obj_idx = vocab.word2idx.get(str(obj).lower(), vocab.word2idx.get("<unk>", 3))

        encoded[i] = [subj_idx, pred_idx, obj_idx]
        mask[i] = False  # Vị trí dữ liệu thật, không phải pad

    return encoded, mask


def get_predicate_statistics(
    scene_graphs: Dict[Any, List[Dict[str, Any]]]
) -> Dict[str, int]:
    """Thống kê tần suất xuất hiện của các quan hệ (predicates) trong bộ dữ liệu.

    Args:
        scene_graphs: Bộ dữ liệu Scene Graph.

    Returns:
        Dictionary ánh xạ từ quan hệ (str) sang số lần xuất hiện (int).
    """
    stats = {}
    for sg in scene_graphs.values():
        triples = sg.get("triples", sg) if isinstance(sg, dict) else sg
        for t in triples:
            pred = t.get("predicate", "").lower()
            if pred:
                stats[pred] = stats.get(pred, 0) + 1
    return dict(sorted(stats.items(), key=lambda x: x[1], reverse=True))


def get_object_statistics(
    scene_graphs: Dict[Any, List[Dict[str, Any]]]
) -> Dict[str, int]:
    """Thống kê tần suất xuất hiện của các loại đối tượng (objects/subjects) trong bộ dữ liệu.

    Args:
        scene_graphs: Bộ dữ liệu Scene Graph.

    Returns:
        Dictionary ánh xạ từ tên đối tượng (str) sang số lần xuất hiện (int).
    """
    stats = {}
    for sg in scene_graphs.values():
        triples = sg.get("triples", sg) if isinstance(sg, dict) else sg
        for t in triples:
            subj = t.get("subject", "").lower()
            obj = t.get("object", "").lower()
            if subj:
                stats[subj] = stats.get(subj, 0) + 1
            if obj:
                stats[obj] = stats.get(obj, 0) + 1
    return dict(sorted(stats.items(), key=lambda x: x[1], reverse=True))


def triple_to_text(triple: Dict[str, Any]) -> str:
    """Chuyển đổi một bộ ba dạng dict thành chuỗi văn bản dễ đọc.

    Ví dụ: {"subject": "dog", "predicate": "on", "object": "grass"} -> "dog on grass"

    Args:
        triple: Bộ ba dạng dictionary.

    Returns:
        Chuỗi văn bản mô tả bộ ba.
    """
    return f"{triple.get('subject', '')} {triple.get('predicate', '')} {triple.get('object', '')}"


def visualize_scene_graph(
    triples: List[Dict[str, Any]],
    image: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> None:
    """Trực quan hóa Đồ thị ngữ cảnh bằng thư viện NetworkX và Matplotlib.

    Vẽ đồ thị có hướng mô tả các đối tượng làm các nút (nodes) và các mối quan hệ
    làm các cạnh nối (edges). Nếu có ảnh gốc đi kèm, hiển thị song song.

    Args:
        triples: Danh sách các bộ ba cần vẽ.
        image: Ảnh gốc tương ứng (mảng numpy RGB), tùy chọn.
        save_path: Đường dẫn lưu ảnh kết quả vẽ đồ thị, tùy chọn.
    """
    try:
        import networkx as nx
    except ImportError:
        print("Cảnh báo: Cần cài đặt thư viện 'networkx' để trực quan hóa đồ thị. Chạy 'pip install networkx'.")
        return

    # Tạo đồ thị có hướng
    G = nx.DiGraph()

    # Thêm cạnh và nút từ các bộ ba
    edge_labels = {}
    for t in triples:
        subj = str(t.get("subject", "")).lower()
        pred = str(t.get("predicate", "")).lower()
        obj = str(t.get("object", "")).lower()
        
        if subj and obj and pred:
            # Tạo nhãn nút duy nhất phòng khi trùng tên đối tượng
            G.add_edge(subj, obj)
            edge_labels[(subj, obj)] = pred

    # Thiết lập kích thước vẽ
    plt.figure(figsize=(10, 8) if image is None else (16, 8))

    if image is not None:
        # Nếu có ảnh, vẽ ảnh bên trái, đồ thị bên phải
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.axis("off")
        plt.title("Ảnh đầu vào")
        plt.subplot(1, 2, 2)

    # Bố cục nút dạng spring layout
    pos = nx.spring_layout(G, seed=42)

    # Vẽ nút
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color="lightblue", alpha=0.9)
    # Vẽ nhãn nút
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", font_family="sans-serif")
    # Vẽ cạnh (mũi tên hướng)
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=15, edge_color="gray", width=1.5)
    # Vẽ nhãn cạnh (quan hệ)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, font_color="darkred")

    plt.axis("off")
    plt.title("Đồ thị ngữ cảnh của ảnh (Scene Graph)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Đã lưu đồ thị ngữ cảnh vẽ được vào: {save_path}")
    else:
        plt.show()
    plt.close()
