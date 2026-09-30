"""MS COCO Captions dataset with Scene Graph support (Karpathy split).

This module provides :class:`COCODataset`, a PyTorch :class:`Dataset` that
loads images, captions, and (optionally) pre-extracted scene graph triples
for the MS COCO dataset using the *Karpathy split* JSON format.

Karpathy split JSON schema::

    {
      "images": [
        {
          "split": "train" | "restval" | "val" | "test",
          "filename": "COCO_train2014_000000XXXX.jpg",
          "imgid": 12345,
          "sentences": [
            {"raw": "A cat sitting on a mat.", "tokens": ["a","cat",...]},
            ...
          ]
        },
        ...
      ]
    }

Scene graph JSON schema (one file, list of dicts)::

    [
      {
        "image_id": 12345,
        "triples": [
          {"subject": "cat", "predicate": "on", "object": "mat",
           "confidence": 0.92},
          ...
        ]
      },
      ...
    ]
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset  # type: ignore[import]
from torchvision import transforms as T  # type: ignore[import]

from src.data.transforms import get_train_transforms, get_val_transforms
from src.data.vocabulary import (
    END_IDX,
    PAD_IDX,
    START_IDX,
    UNK_IDX,
    Vocabulary,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
MAX_CAPTION_LENGTH: int = 30  # including <start> and <end>
MAX_TRIPLES: int = 20         # max scene-graph triples per image
TRIPLE_TOKEN_LEN: int = 3     # (subject, predicate, object)

# Mapping from split name → Karpathy split tags.
# "restval" images are folded into the training set following standard
# practice (Karpathy & Fei-Fei, 2015).
_SPLIT_MAP: Dict[str, List[str]] = {
    "train": ["train", "restval"],
    "val": ["val"],
    "test": ["test"],
}


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class COCODataset(Dataset):
    """MS COCO Captions dataset with scene-graph triples.

    Each sample returned by :meth:`__getitem__` is a dict containing:
        - **image**: ``Tensor[3, crop, crop]`` – transformed RGB image.
        - **caption**: ``LongTensor[max_caption_length]`` – padded
          token-index sequence including ``<start>`` and ``<end>``.
          Key is named 'caption' for backward compatibility.
        - **caption_len**: ``int`` – true length (before padding).
        - **triples**: ``LongTensor[max_triples, 3]`` – each row
          is ``(subject_idx, predicate_idx, object_idx)`` encoded via the
          vocabulary. Padded rows are all ``<pad>`` (0).
        - **image_id**: ``int`` – unique COCO image id.
        - **raw_captions**: ``List[str]`` - all reference captions for this image.

    Args:
        karpathy_json: Path to the Karpathy-split JSON.
        image_root: Root directory containing the COCO images.
        vocab: A :class:`Vocabulary` instance for text encoding.
        split: One of ``"train"``, ``"val"``, ``"test"``.
        scene_graph_json: Optional path to the scene-graph JSON file.
        transform: Optional torchvision transform.
        max_caption_length: Maximum caption length (padded/truncated).
        max_triples: Maximum number of scene-graph triples per image.
        captions_per_image: Number of captions to sample per image.
    """

    def __init__(
        self,
        karpathy_json: Union[str, Path],
        image_root: Union[str, Path],
        vocab: Vocabulary,
        split: str = "train",
        scene_graph_json: Optional[Union[str, Path]] = None,
        transform: Optional[T.Compose] = None,
        max_caption_length: int = MAX_CAPTION_LENGTH,
        max_triples: int = MAX_TRIPLES,
        captions_per_image: int = -1,
    ) -> None:
        super().__init__()

        assert split in _SPLIT_MAP, f"Unknown split '{split}'. Choose from {list(_SPLIT_MAP)}"

        self.image_root = Path(image_root)
        self.vocab = vocab
        self.split = split
        self.max_caption_length = max_caption_length
        self.max_triples = max_triples

        # Default transforms if none supplied.
        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = get_train_transforms()
        else:
            self.transform = get_val_transforms()

        # ---- Load Karpathy JSON ----------------------------------------- #
        logger.info("Loading Karpathy split JSON: %s …", karpathy_json)
        with open(karpathy_json, "r", encoding="utf-8") as f:
            karpathy_data = json.load(f)

        valid_splits = set(_SPLIT_MAP[split])

        # Build a flat list of (filename, image_id, raw_caption) tuples.
        self.samples: List[Tuple[str, int, str]] = []
        self.img_id_to_captions: Dict[int, List[str]] = {}
        
        for img_entry in karpathy_data["images"]:
            if img_entry["split"] not in valid_splits:
                continue
            filename: str = img_entry["filename"]
            image_id: int = img_entry["imgid"]
            captions_raw: List[str] = [
                s["raw"] for s in img_entry["sentences"]
            ]
            self.img_id_to_captions[image_id] = captions_raw

            if captions_per_image == -1:
                # Expand every caption as a separate training sample.
                for cap in captions_raw:
                    self.samples.append((filename, image_id, cap))
            else:
                # Take only the first N captions.
                for cap in captions_raw[:captions_per_image]:
                    self.samples.append((filename, image_id, cap))

        logger.info(
            "[%s] Loaded %d samples from %s.",
            split,
            len(self.samples),
            karpathy_json,
        )

        # ---- Load scene graphs ------------------------------------------ #
        # Index: image_id → list of triple dicts.
        self.scene_graphs: Dict[int, List[Dict[str, Any]]] = {}
        if scene_graph_json is not None:
            sg_path = Path(scene_graph_json)
            if sg_path.exists():
                logger.info("Loading scene graphs: %s …", sg_path)
                with open(sg_path, "r", encoding="utf-8") as f:
                    sg_data = json.load(f)
                
                # Robust parsing of both formats (list of dicts vs dictionary)
                if isinstance(sg_data, list):
                    for entry in sg_data:
                        img_id = entry["image_id"]
                        self.scene_graphs[img_id] = entry.get("triples", [])
                elif isinstance(sg_data, dict):
                    for k, v in sg_data.items():
                        try:
                            img_id = int(k)
                        except ValueError:
                            img_id = k
                        if isinstance(v, dict) and "triples" in v:
                            self.scene_graphs[img_id] = v["triples"]
                        else:
                            self.scene_graphs[img_id] = v
                            
                logger.info(
                    "Loaded scene graphs for %d images.", len(self.scene_graphs)
                )
            else:
                logger.warning(
                    "Scene-graph file not found: %s. Proceeding without scene graphs.",
                    sg_path,
                )

    # --------------------------------------------------------------------- #
    # Dataset interface
    # --------------------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        filename, image_id, raw_caption = self.samples[index]

        # ---- Image ------------------------------------------------------ #
        image = self._load_image(filename)

        # ---- Caption ---------------------------------------------------- #
        caption_indices = self.vocab.encode(raw_caption)
        caption_length = min(len(caption_indices), self.max_caption_length)
        # Pad or truncate to fixed length.
        caption_indices = self._pad_or_truncate(
            caption_indices, self.max_caption_length, PAD_IDX
        )
        caption_tensor = torch.LongTensor(caption_indices)

        # ---- Scene graph ------------------------------------------------ #
        sg_tensor = self._encode_scene_graph(image_id)
        raw_captions = self.img_id_to_captions.get(image_id, [raw_caption])

        return {
            "image": image,
            "caption": caption_tensor,
            "caption_len": caption_length,
            "triples": sg_tensor,
            "image_id": image_id,
            "raw_captions": raw_captions
        }

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _load_image(self, filename: str) -> torch.Tensor:
        """Load an image file, convert to RGB, and apply the transform.

        The method looks for the file in several possible locations under
        *image_root* to accommodate different directory layouts.
        """
        candidates = [
            self.image_root / filename,
            self.image_root / "train2014" / filename,
            self.image_root / "val2014" / filename,
            self.image_root / "test2015" / filename,
        ]
        for path in candidates:
            if path.exists():
                image = Image.open(path).convert("RGB")
                if self.transform is not None:
                    image = self.transform(image)
                return image

        raise FileNotFoundError(
            f"Image '{filename}' not found under {self.image_root}. "
            f"Tried: {[str(c) for c in candidates]}"
        )

    def _encode_scene_graph(self, image_id: int) -> torch.LongTensor:
        """Encode scene-graph triples for *image_id* as vocabulary indices.

        Each triple ``(subject, predicate, object)`` is converted to
        ``(subj_idx, pred_idx, obj_idx)`` via the vocabulary.  The result
        is padded/truncated to shape ``(max_triples, 3)``.

        If no scene graph is available for the image, a zero-padded tensor
        is returned (graceful fallback).
        """
        triples_raw = self.scene_graphs.get(image_id, [])

        encoded: List[List[int]] = []
        for triple in triples_raw[: self.max_triples]:
            subj = self._word_to_idx(triple.get("subject", ""))
            pred = self._word_to_idx(triple.get("predicate", ""))
            obj = self._word_to_idx(triple.get("object", ""))
            encoded.append([subj, pred, obj])

        # Pad with zeros if fewer triples than max_triples.
        while len(encoded) < self.max_triples:
            encoded.append([PAD_IDX, PAD_IDX, PAD_IDX])

        return torch.LongTensor(encoded)  # shape: (max_triples, 3)

    def _word_to_idx(self, text: str) -> int:
        """Map a single word/phrase to its vocabulary index.

        For multi-word entity labels (e.g. ``"traffic light"``), we take
        the index of the first word that is in the vocabulary so that the
        representation stays single-token.  Falls back to ``<unk>``.
        """
        text = text.strip().lower()
        if not text:
            return PAD_IDX
        # Try the full text first (handles single-word entities).
        if text in self.vocab.word2idx:
            return self.vocab.word2idx[text]
        # Fall back to the first known word.
        for token in text.split():
            if token in self.vocab.word2idx:
                return self.vocab.word2idx[token]
        return UNK_IDX

    @staticmethod
    def _pad_or_truncate(
        seq: List[int], max_len: int, pad_value: int
    ) -> List[int]:
        """Pad *seq* with *pad_value* or truncate to *max_len*.

        If the sequence must be truncated, we ensure the last token is
        ``<end>`` so that the decoder always sees a proper sentence boundary.
        """
        if len(seq) >= max_len:
            seq = seq[:max_len]
            # Guarantee <end> at the end even after truncation.
            seq[-1] = END_IDX
        else:
            seq = seq + [pad_value] * (max_len - len(seq))
        return seq

    # --------------------------------------------------------------------- #
    # Collate function
    # --------------------------------------------------------------------- #
    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Custom collate for DataLoader.

        Sorts the batch by caption length in *descending* order.

        Returns:
            A dictionary containing:
                - **image**: ``Tensor[B, 3, H, W]``
                - **caption**: ``LongTensor[B, max_caption_length]``
                - **caption_len**: ``LongTensor[B]``
                - **triples**: ``LongTensor[B, max_triples, 3]``
                - **triple_mask**: ``BoolTensor[B, max_triples]``
                - **image_id**: ``LongTensor[B]``
                - **raw_captions**: ``List[List[str]]``
        """
        # Sort by caption length (descending).
        batch = sorted(batch, key=lambda x: x["caption_len"], reverse=True)

        images = torch.stack([x["image"] for x in batch], dim=0)
        captions = torch.stack([x["caption"] for x in batch], dim=0)
        lengths = torch.LongTensor([x["caption_len"] for x in batch])
        scene_graphs = torch.stack([x["triples"] for x in batch], dim=0)
        image_ids = [x["image_id"] for x in batch]
        raw_captions = [x["raw_captions"] for x in batch]

        # Compute triple_mask on the fly: True for padded positions (all three indices are PAD_IDX=0)
        triple_mask = (scene_graphs == PAD_IDX).all(dim=-1)

        # Prevent all-True rows in triple_mask to avoid NaNs and PyTorch NestedTensor RuntimeError
        # If a row is entirely True (no valid triples), we set the first entry to False.
        all_padded = triple_mask.all(dim=-1)
        triple_mask[all_padded, 0] = False

        return {
            "image": images,
            "caption": captions,
            "caption_len": lengths,
            "triples": scene_graphs,
            "triple_mask": triple_mask,
            "image_id": torch.LongTensor(image_ids),
            "raw_captions": raw_captions
        }


# --------------------------------------------------------------------------- #
# Convenience factory
# --------------------------------------------------------------------------- #
def get_dataloader(
    split: str = "train",
    config: Optional[Any] = None,
    vocab: Optional[Vocabulary] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    karpathy_json: Optional[Union[str, Path]] = None,
    image_root: Optional[Union[str, Path]] = None,
    scene_graph_json: Optional[Union[str, Path]] = None,
    max_caption_length: int = MAX_CAPTION_LENGTH,
    max_triples: int = MAX_TRIPLES,
    transform: Optional[T.Compose] = None,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a :class:`DataLoader` for a given split.

    This is a thin convenience wrapper around :class:`COCODataset` and
    :class:`torch.utils.data.DataLoader`.
    """
    # Extract fields from config if provided
    if config is not None:
        karpathy_json = config.data.captions_file
        image_root = config.data.images_dir
        
        # Resolve scene graph file
        sg_dir = config.data.get("scene_graphs_dir", "data/coco/scene_graphs")
        if sg_dir:
            scene_graph_json = os.path.join(sg_dir, f"{split}_scene_graphs.json")
            
        batch_size = batch_size if batch_size is not None else config.data.batch_size
        num_workers = num_workers if num_workers is not None else config.data.num_workers
        max_caption_length = config.data.get("max_caption_len", max_caption_length)
        max_triples = config.data.get("max_triples", max_triples)
        pin_memory = config.data.get("pin_memory", pin_memory)

    # Fallback to defaults
    batch_size = batch_size if batch_size is not None else 32
    num_workers = num_workers if num_workers is not None else 4

    if karpathy_json is None or image_root is None or vocab is None:
        raise ValueError("karpathy_json, image_root, and vocab must be provided (directly or via config)")

    dataset = COCODataset(
        karpathy_json=karpathy_json,
        image_root=image_root,
        vocab=vocab,
        split=split,
        scene_graph_json=scene_graph_json,
        transform=transform,
        max_caption_length=max_caption_length,
        max_triples=max_triples,
    )

    shuffle = split == "train"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=COCODataset.collate_fn,
        drop_last=(split == "train"),
    )
