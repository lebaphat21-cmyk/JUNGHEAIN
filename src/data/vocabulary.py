"""Vocabulary module for Image Captioning.

Builds and manages a word-to-index mapping from COCO captions.
Supports encoding/decoding text, saving/loading to JSON, and
constructing GloVe embedding matrices aligned to the vocabulary.

Special tokens:
    <pad>=0, <start>=1, <end>=2, <unk>=3
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import nltk

# Ensure the punkt_tab tokenizer data is available.
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PAD_TOKEN: str = "<pad>"
START_TOKEN: str = "<start>"
END_TOKEN: str = "<end>"
UNK_TOKEN: str = "<unk>"

PAD_IDX: int = 0
START_IDX: int = 1
END_IDX: int = 2
UNK_IDX: int = 3

SPECIAL_TOKENS: List[str] = [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN]


class Vocabulary:
    """Word ↔ index mapping for caption generation.

    Attributes:
        word2idx: Mapping from word string to integer index.
        idx2word: Mapping from integer index to word string.
        min_freq: Minimum word frequency used when building the vocab.
    """

    # --------------------------------------------------------------------- #
    # Construction
    # --------------------------------------------------------------------- #
    def __init__(self) -> None:
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.min_freq: int = 5

        # Always initialise with the four special tokens.
        self._add_special_tokens()

    def _add_special_tokens(self) -> None:
        """Insert the fixed special tokens at their canonical indices."""
        for idx, token in enumerate(SPECIAL_TOKENS):
            self.word2idx[token] = idx
            self.idx2word[idx] = token

    # --------------------------------------------------------------------- #
    # Building
    # --------------------------------------------------------------------- #
    @classmethod
    def build_from_captions(
        cls,
        captions: Sequence[str],
        min_freq: int = 5,
    ) -> "Vocabulary":
        """Build a :class:`Vocabulary` from a list of raw caption strings.

        Args:
            captions: Iterable of raw caption strings.
            min_freq: Only words that appear at least *min_freq* times across
                all captions are included in the vocabulary.

        Returns:
            A new :class:`Vocabulary` instance.
        """
        vocab = cls()
        vocab.min_freq = min_freq

        # Count word frequencies across the entire caption corpus.
        counter: Counter = Counter()
        for caption in captions:
            tokens = _tokenize(caption)
            counter.update(tokens)

        # Add words that meet the minimum-frequency threshold.
        for word, freq in counter.items():
            if freq >= min_freq:
                vocab._add_word(word)

        logger.info(
            "Built vocabulary: %d words (min_freq=%d) from %d captions.",
            len(vocab),
            min_freq,
            len(captions),
        )
        return vocab

    def _add_word(self, word: str) -> int:
        """Add a single word if it is not already present.

        Returns:
            The index assigned to *word*.
        """
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            return idx
        return self.word2idx[word]

    # --------------------------------------------------------------------- #
    # Encode / Decode
    # --------------------------------------------------------------------- #
    def encode(self, text: str) -> List[int]:
        """Convert a raw text string to a list of token indices.

        The output sequence is wrapped with <start> and <end> tokens::

            [<start>, w1, w2, ..., wn, <end>]

        Unknown words are mapped to the ``<unk>`` index.

        Args:
            text: Raw caption string (may be empty).

        Returns:
            List of integer indices.
        """
        if not text or not text.strip():
            return [START_IDX, END_IDX]

        tokens = _tokenize(text)
        indices = [self.word2idx.get(t, UNK_IDX) for t in tokens]
        return [START_IDX] + indices + [END_IDX]

    @staticmethod
    def clean_caption_text(text: str) -> str:
        """Clean and polish raw caption text.

        - Removes <unk>, <pad>, <start>, <end> tokens.
        - Removes consecutive duplicate words.
        - Capitalises first letter and ensures ending period.
        """
        if not text:
            return ""

        # Remove special tokens if present
        for tok in [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN]:
            text = text.replace(tok, "")

        words = text.strip().split()
        if not words:
            return ""

        # Remove consecutive duplicate words
        cleaned_words = []
        for w in words:
            if not cleaned_words or w.lower() != cleaned_words[-1].lower():
                cleaned_words.append(w)

        result = " ".join(cleaned_words)
        if result:
            result = result[0].upper() + result[1:]
            if not result.endswith((".", "!", "?")):
                result += "."
        return result

    def decode(
        self,
        indices: Sequence[int],
        skip_special: bool = True,
        join: bool = True,
        clean: bool = True,
    ) -> Union[str, List[str]]:
        """Convert a sequence of indices back to text.

        Args:
            indices: Iterable of integer token indices.
            skip_special: If *True*, special tokens (<pad>, <start>, <end>)
                are omitted from the output.
            join: If *True* return a single space-joined string; otherwise
                return a list of word strings.
            clean: If *True* and *join* is *True*, run clean_caption_text on the result.

        Returns:
            Decoded string (or list of strings if *join* is *False*).
        """
        skip = {PAD_IDX, START_IDX, END_IDX} if skip_special else set()
        if skip_special:
            skip.add(UNK_IDX)
        words: List[str] = []
        for idx in indices:
            idx = int(idx)  # handle numpy / tensor scalars
            if idx in skip:
                continue
            # Stop at <end> when skipping specials (natural sentence boundary).
            if skip_special and idx == END_IDX:
                break
            w = self.idx2word.get(idx, "")
            if w and w not in SPECIAL_TOKENS:
                words.append(w)
        if not join:
            return words
        raw_text = " ".join(words)
        return self.clean_caption_text(raw_text) if clean else raw_text

    # --------------------------------------------------------------------- #
    # Persistence (JSON)
    # --------------------------------------------------------------------- #
    def save_to_file(self, path: Union[str, Path]) -> None:
        """Serialise the vocabulary to a JSON file.

        The JSON structure stores both mappings and metadata so that
        reconstruction is deterministic.

        Args:
            path: Destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "min_freq": self.min_freq,
            "word2idx": self.word2idx,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info("Vocabulary saved to %s (%d words).", path, len(self))

    @classmethod
    def load_from_file(cls, path: Union[str, Path]) -> "Vocabulary":
        """Load a vocabulary previously saved with :meth:`save_to_file`.

        Args:
            path: Path to the JSON file.

        Returns:
            A reconstructed :class:`Vocabulary` instance.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        vocab = cls()
        vocab.min_freq = data.get("min_freq", 5)
        vocab.word2idx = data["word2idx"]
        # Reconstruct idx2word (JSON keys are always strings).
        vocab.idx2word = {int(idx): word for word, idx in vocab.word2idx.items()}

        logger.info("Vocabulary loaded from %s (%d words).", path, len(vocab))
        return vocab

    # --------------------------------------------------------------------- #
    # GloVe Embeddings
    # --------------------------------------------------------------------- #
    def get_glove_matrix(
        self,
        glove_path: Union[str, Path],
        embed_dim: int = 300,
    ) -> np.ndarray:
        """Build an embedding matrix aligned with this vocabulary.

        Words present in the GloVe file receive their pre-trained vector;
        words absent from GloVe are initialised with a small random normal
        distribution (σ = 0.6 / √embed_dim).

        Args:
            glove_path: Path to a GloVe text file
                (e.g. ``glove.6B.300d.txt``).
            embed_dim: Dimensionality of the GloVe vectors.

        Returns:
            A numpy array of shape ``(vocab_size, embed_dim)`` with
            ``dtype=float32``.  Row 0 (the ``<pad>`` token) is always
            all-zeros.
        """
        glove_path = Path(glove_path)
        if not glove_path.exists():
            raise FileNotFoundError(f"GloVe file not found: {glove_path}")

        vocab_size = len(self)
        # Random init for words not covered by GloVe.
        scale = 0.6 / np.sqrt(embed_dim)
        matrix = np.random.normal(0.0, scale, (vocab_size, embed_dim)).astype(
            np.float32
        )
        # <pad> should remain zero-vector.
        matrix[PAD_IDX] = 0.0

        found = 0
        with open(glove_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                word = parts[0]
                if word in self.word2idx:
                    try:
                        vector = np.array(parts[1:], dtype=np.float32)
                        if vector.shape[0] == embed_dim:
                            matrix[self.word2idx[word]] = vector
                            found += 1
                    except ValueError:
                        # Malformed line – skip silently.
                        continue

        coverage = found / max(vocab_size - len(SPECIAL_TOKENS), 1) * 100
        logger.info(
            "GloVe: loaded %d/%d vectors (%.1f%% coverage).",
            found,
            vocab_size,
            coverage,
        )
        return matrix

    # --------------------------------------------------------------------- #
    # Dunder helpers
    # --------------------------------------------------------------------- #
    def __len__(self) -> int:
        """Return the total number of tokens (including special tokens)."""
        return len(self.word2idx)

    def __contains__(self, word: str) -> bool:
        return word in self.word2idx

    def __repr__(self) -> str:
        return f"Vocabulary(size={len(self)}, min_freq={self.min_freq})"


# --------------------------------------------------------------------------- #
# Module-level helpers
# --------------------------------------------------------------------------- #
def _tokenize(text: str) -> List[str]:
    """Lower-case and tokenise *text* using NLTK's word_tokenize.

    Args:
        text: Raw text string.

    Returns:
        List of lower-cased tokens.
    """
    return word_tokenize(text.lower())
