"""
IAM Online Handwriting Dataset - PyTorch data pipeline.

Parses IAM stroke XML files into sequences of (delta_x, delta_y, pen_up) triples,
normalizes deltas using training-set statistics, and provides a padded, masked
Dataset / DataLoader for variable-length sequences.

Also supports text-conditioned training with character-level encoding and
windowed attention over the conditioning text.
"""

import math
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Character vocabulary
# ---------------------------------------------------------------------------

# Default character set: printable ASCII + space
DEFAULT_CHARSET = (
    " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
    "abcdefghijklmnopqrstuvwxyz{|}~"
)


class CharVocab:
    """Simple character-level vocabulary with encode/decode."""

    def __init__(self, charset: str = DEFAULT_CHARSET):
        self.charset = charset
        self.char_to_idx = {c: i + 1 for i, c in enumerate(charset)}  # 0 = padding
        self.idx_to_char = {i + 1: c for i, c in enumerate(charset)}
        self.idx_to_char[0] = ""
        self.vocab_size = len(charset) + 1  # +1 for padding

    def encode(self, text: str) -> list[int]:
        return [self.char_to_idx.get(c, 0) for c in text]

    def decode(self, indices: list[int]) -> str:
        return "".join(self.idx_to_char.get(i, "") for i in indices)

    def __len__(self) -> int:
        return self.vocab_size


# ---------------------------------------------------------------------------
# XML Parsing
# ---------------------------------------------------------------------------

def parse_iam_xml(xml_path: str | Path) -> tuple[np.ndarray, str]:
    """Parse an IAM online handwriting XML file.

    Returns:
        points: ndarray of shape (N, 3) where each row is [x, y, pen_up]
        text: the transcript string associated with this line (may be empty)
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    # Try to extract text from Line@text attribute
    text = ""
    line_elem = root.find(".//Line")
    if line_elem is not None:
        text = line_elem.get("text", "")
    # Fallback: check root attributes
    if not text:
        text = root.get("text", "")
    # Fallback: try to get from filename (IAM convention: authorId-formId-lineId.xml)
    if not text:
        text = ""

    points: list[list[float]] = []

    for stroke in root.findall(".//Stroke"):
        stroke_points = stroke.findall("Point")
        for i, pt in enumerate(stroke_points):
            x = float(pt.get("x", "0"))
            y = float(pt.get("y", "0"))
            pen_up = 1 if i == len(stroke_points) - 1 else 0
            points.append([x, y, pen_up])

    if not points:
        return np.empty((0, 3), dtype=np.float32), text

    return np.array(points, dtype=np.float32), text


def absolute_to_relative(points: np.ndarray) -> np.ndarray:
    """Convert absolute (x, y, pen_up) to relative (delta_x, delta_y, pen_up).

    The first point has delta_x=0, delta_y=0.
    """
    if len(points) == 0:
        return points

    deltas = np.zeros_like(points)
    deltas[1:, 0] = points[1:, 0] - points[:-1, 0]
    deltas[1:, 1] = points[1:, 1] - points[:-1, 1]
    deltas[:, 2] = points[:, 2]  # pen_up stays the same
    return deltas


def relative_to_absolute(deltas: np.ndarray, start_x: float = 0.0, start_y: float = 0.0) -> np.ndarray:
    """Convert relative (delta_x, delta_y, pen_up) back to absolute (x, y, pen_up)."""
    if len(deltas) == 0:
        return deltas

    abs_points = np.zeros_like(deltas)
    abs_points[0, 0] = start_x + deltas[0, 0]
    abs_points[0, 1] = start_y + deltas[0, 1]
    abs_points[0, 2] = deltas[0, 2]

    for i in range(1, len(deltas)):
        abs_points[i, 0] = abs_points[i - 1, 0] + deltas[i, 0]
        abs_points[i, 1] = abs_points[i - 1, 1] + deltas[i, 1]
        abs_points[i, 2] = deltas[i, 2]

    return abs_points


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def compute_dataset_stats(
    xml_paths: list[str | Path],
) -> tuple[float, float, float, float]:
    """Compute mean and std of delta_x and delta_y over a set of XML files.

    Returns (mean_x, std_x, mean_y, std_y).
    """
    all_dx: list[float] = []
    all_dy: list[float] = []

    for path in xml_paths:
        pts, _ = parse_iam_xml(path)
        if len(pts) < 2:
            continue
        rel = absolute_to_relative(pts)
        all_dx.extend(rel[1:, 0].tolist())
        all_dy.extend(rel[1:, 1].tolist())

    mean_x = float(np.mean(all_dx))
    std_x = float(np.std(all_dx))
    mean_y = float(np.mean(all_dy))
    std_y = float(np.std(all_dy))

    std_x = std_x if std_x > 1e-8 else 1.0
    std_y = std_y if std_y > 1e-8 else 1.0

    return mean_x, std_x, mean_y, std_y


def normalize_deltas(
    deltas: np.ndarray,
    mean_x: float,
    std_x: float,
    mean_y: float,
    std_y: float,
) -> np.ndarray:
    """Apply z-score normalization to delta_x and delta_y columns."""
    out = deltas.copy()
    out[:, 0] = (deltas[:, 0] - mean_x) / std_x
    out[:, 1] = (deltas[:, 1] - mean_y) / std_y
    return out


def denormalize_deltas(
    deltas: np.ndarray,
    mean_x: float,
    std_x: float,
    mean_y: float,
    std_y: float,
) -> np.ndarray:
    """Reverse z-score normalization."""
    out = deltas.copy()
    out[:, 0] = deltas[:, 0] * std_x + mean_x
    out[:, 1] = deltas[:, 1] * std_y + mean_y
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_strokes(
    deltas: np.ndarray,
    title: str = "",
    figsize: tuple[float, float] = (6, 3),
    dpi: int = 100,
) -> plt.Figure:
    """Render a (delta_x, delta_y, pen_up) sequence as a matplotlib figure.

    Returns the Figure object so the caller can save or display it.
    """
    abs_pts = relative_to_absolute(deltas)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Split into individual strokes
    stroke_start = 0
    for i in range(len(abs_pts)):
        if abs_pts[i, 2] == 1:  # pen lift
            stroke = abs_pts[stroke_start : i + 1]
            ax.plot(stroke[:, 0], -stroke[:, 1], color="black", linewidth=1.5)
            stroke_start = i + 1

    # Handle any trailing points
    if stroke_start < len(abs_pts):
        stroke = abs_pts[stroke_start:]
        ax.plot(stroke[:, 0], -stroke[:, 1], color="black", linewidth=1.5)

    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title)

    fig.tight_layout(pad=0)
    return fig


# ---------------------------------------------------------------------------
# Dataset / DataLoader
# ---------------------------------------------------------------------------

class IAMStrokeDataset(Dataset):
    """PyTorch Dataset that loads IAM XML files and returns normalized delta sequences."""

    def __init__(
        self,
        xml_paths: list[str | Path],
        mean_x: float = 0.0,
        std_x: float = 1.0,
        mean_y: float = 0.0,
        std_y: float = 1.0,
        min_seq_len: int = 10,
    ):
        self.mean_x = mean_x
        self.std_x = std_x
        self.mean_y = mean_y
        self.std_y = std_y
        self.min_seq_len = min_seq_len

        self.sequences: list[np.ndarray] = []
        self._load(xml_paths)

    def _load(self, xml_paths: list[str | Path]) -> None:
        for path in xml_paths:
            pts, _ = parse_iam_xml(path)
            if len(pts) < self.min_seq_len:
                continue
            deltas = absolute_to_relative(pts)
            deltas = normalize_deltas(deltas, self.mean_x, self.std_x, self.mean_y, self.std_y)
            self.sequences.append(deltas)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.sequences[idx]


class IAMConditionedDataset(Dataset):
    """Dataset that returns (text, stroke_sequence) pairs for conditioned training."""

    def __init__(
        self,
        xml_paths: list[str | Path],
        vocab: CharVocab | None = None,
        mean_x: float = 0.0,
        std_x: float = 1.0,
        mean_y: float = 0.0,
        std_y: float = 1.0,
        min_seq_len: int = 10,
        min_text_len: int = 1,
    ):
        self.vocab = vocab or CharVocab()
        self.mean_x = mean_x
        self.std_x = std_x
        self.mean_y = mean_y
        self.std_y = std_y
        self.min_seq_len = min_seq_len
        self.min_text_len = min_text_len

        self.samples: list[tuple[list[int], np.ndarray]] = []  # (char_ids, deltas)
        self._load(xml_paths)

    def _load(self, xml_paths: list[str | Path]) -> None:
        for path in xml_paths:
            pts, text = parse_iam_xml(path)
            if len(pts) < self.min_seq_len:
                continue
            text = text.strip()
            if len(text) < self.min_text_len:
                continue

            deltas = absolute_to_relative(pts)
            deltas = normalize_deltas(deltas, self.mean_x, self.std_x, self.mean_y, self.std_y)
            char_ids = self.vocab.encode(text)

            if len(char_ids) == 0:
                continue

            self.samples.append((char_ids, deltas))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[list[int], np.ndarray]:
        return self.samples[idx]


def collate_fn(batch: list[np.ndarray]) -> dict[str, torch.Tensor]:
    """Pad variable-length sequences and create an attention mask.

    Returns:
        data:   FloatTensor of shape (B, T_max, 3)
        mask:   BoolTensor of shape (B, T_max) -- True means valid
        lengths: IntTensor of shape (B,) -- original sequence lengths
    """
    lengths = [len(seq) for seq in batch]
    max_len = max(lengths)

    data = torch.zeros(len(batch), max_len, 3, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)

    for i, seq in enumerate(batch):
        t = lengths[i]
        data[i, :t, :] = torch.from_numpy(seq)
        mask[i, :t] = True

    return {"data": data, "mask": mask, "lengths": torch.tensor(lengths)}


def collate_fn_conditioned(
    batch: list[tuple[list[int], np.ndarray]],
) -> dict[str, torch.Tensor]:
    """Pad both character sequences and stroke sequences.

    Returns:
        data:       FloatTensor (B, T_max, 3)  normalized strokes
        mask:       BoolTensor (B, T_max)      valid stroke timesteps
        lengths:    IntTensor (B,)             stroke lengths
        char_ids:   LongTensor (B, C_max)      character indices
        char_mask:  BoolTensor (B, C_max)      valid characters
        char_lens:  IntTensor (B,)             character sequence lengths
        texts:      list[str]                  decoded text strings
    """
    stroke_lengths = []
    char_lengths = []
    max_stroke_len = 0
    max_char_len = 0

    for char_ids, strokes in batch:
        stroke_lengths.append(len(strokes))
        char_lengths.append(len(char_ids))
        max_stroke_len = max(max_stroke_len, len(strokes))
        max_char_len = max(max_char_len, len(char_ids))

    B = len(batch)
    data = torch.zeros(B, max_stroke_len, 3, dtype=torch.float32)
    mask = torch.zeros(B, max_stroke_len, dtype=torch.bool)
    char_ids_padded = torch.zeros(B, max_char_len, dtype=torch.long)
    char_mask = torch.zeros(B, max_char_len, dtype=torch.bool)
    texts = []

    for i, (char_ids, strokes) in enumerate(batch):
        t_s = stroke_lengths[i]
        t_c = char_lengths[i]
        data[i, :t_s, :] = torch.from_numpy(strokes)
        mask[i, :t_s] = True
        char_ids_padded[i, :t_c] = torch.tensor(char_ids, dtype=torch.long)
        char_mask[i, :t_c] = True

    return {
        "data": data,
        "mask": mask,
        "lengths": torch.tensor(stroke_lengths),
        "char_ids": char_ids_padded,
        "char_mask": char_mask,
        "char_lens": torch.tensor(char_lengths),
        "texts": texts,
    }


def build_dataloader(
    xml_paths: list[str | Path],
    batch_size: int = 32,
    shuffle: bool = True,
    mean_x: float = 0.0,
    std_x: float = 1.0,
    mean_y: float = 0.0,
    std_y: float = 1.0,
    num_workers: int = 0,
    **dataset_kwargs,
) -> DataLoader:
    """Convenience function to build a DataLoader with proper collation."""
    ds = IAMStrokeDataset(
        xml_paths,
        mean_x=mean_x,
        std_x=std_x,
        mean_y=mean_y,
        std_y=std_y,
        **dataset_kwargs,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def build_conditioned_dataloader(
    xml_paths: list[str | Path],
    vocab: CharVocab | None = None,
    batch_size: int = 32,
    shuffle: bool = True,
    mean_x: float = 0.0,
    std_x: float = 1.0,
    mean_y: float = 0.0,
    std_y: float = 1.0,
    num_workers: int = 0,
    **dataset_kwargs,
) -> DataLoader:
    """Build a DataLoader for text-conditioned training."""
    ds = IAMConditionedDataset(
        xml_paths,
        vocab=vocab,
        mean_x=mean_x,
        std_x=std_x,
        mean_y=mean_y,
        std_y=std_y,
        **dataset_kwargs,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn_conditioned,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


# ---------------------------------------------------------------------------
# High-level helpers
# ---------------------------------------------------------------------------

def collect_xml_files(root_dir: str | Path, split_file: Optional[str | Path] = None) -> list[Path]:
    """Recursively collect .xml files under root_dir.

    If split_file is provided, only include files listed in it (one filename per line).
    """
    root = Path(root_dir)
    if split_file is not None:
        with open(split_file) as f:
            allowed = {line.strip() for line in f if line.strip()}
        return [root / name for name in allowed if (root / name).exists()]
    return sorted(root.rglob("*.xml"))


def prepare_splits(
    data_dir: str | Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Split all XML files into train / val / test lists."""
    rng = np.random.default_rng(seed)
    all_xml = collect_xml_files(data_dir)
    rng.shuffle(all_xml)

    n = len(all_xml)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    return all_xml[:n_train], all_xml[n_train : n_train + n_val], all_xml[n_train + n_val :]
