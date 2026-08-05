"""
Animated rendering of handwriting: a GIF of the pen drawing the strokes.

Renders a stroke sequence as a progressively-drawn animation where the pen
traces the trajectory stroke-by-stroke, lifting between strokes (no connector
lines across pen-up events). Uses Pillow as the GIF writer (bundled with the
``api`` / ``demo`` extras), falling back to a graceful error if unavailable.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter

from data import relative_to_absolute
from render import THEMES


def _split_segments(abs_pts: np.ndarray) -> list[np.ndarray]:
    """Split absolute points into pen-lift-delimited segments (as in render.py)."""
    segments: list[np.ndarray] = []
    start = 0
    for i in range(len(abs_pts)):
        if abs_pts[i, 2] == 1:
            segments.append(abs_pts[start : i + 1])
            start = i + 1
    if start < len(abs_pts):
        segments.append(abs_pts[start:])
    return segments


def build_draw_frames(
    deltas: np.ndarray,
    theme: str = "ink",
    step: int = 1,
    dpi: int = 110,
    figsize: tuple[float, float] = (8, 3),
    title: str = "",
) -> tuple[plt.Figure, list[np.ndarray]]:
    """Build a figure plus the per-frame absolute-point slices to reveal.

    Each frame reveals ``step`` additional points. Within a frame, every fully
    completed stroke is shown in full; the stroke currently being drawn is
    shown up to the revealed point; strokes not yet started are hidden.

    Returns:
        (fig, lines, segments, ends, reveals): the figure (with one Line2D
        per stroke), the segment data, their cumulative endpoints, and the
        list of reveal counts for each frame.
    """
    abs_pts = relative_to_absolute(np.asarray(deltas))
    segments = _split_segments(abs_pts)
    if not segments:
        raise ValueError("deltas contains no stroke points to animate")

    theme_obj = THEMES.get(theme, THEMES["ink"])
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(theme_obj.background_color)
    ax.set_facecolor(theme_obj.background_color)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, color=theme_obj.title_color, fontsize=11, pad=8)

    lines = []
    for seg in segments:
        (line,) = ax.plot(
            seg[:, 0],
            -seg[:, 1],
            color=theme_obj.stroke_color,
            linewidth=theme_obj.line_width,
            solid_capstyle="round",
            solid_joinstyle="round",
            visible=False,
        )
        lines.append(line)

    # Cumulative reveal count across segments
    ends = np.cumsum([len(s) for s in segments])
    total = int(ends[-1])
    reveals = list(range(0, total, step))
    if not reveals or reveals[-1] != total - 1:
        reveals.append(total - 1)

    return fig, lines, segments, ends, reveals


def _revealed_frames(lines, segments, ends, reveals):
    """Generator yielding per-frame (ax.set_xlim/ylim, line data) updates."""
    for count in reveals:
        data_updates = []
        for i, (seg, end) in enumerate(zip(segments, ends, strict=False)):
            start = 0 if i == 0 else ends[i - 1]
            if count < start:
                data_updates.append((lines[i], [], []))
            else:
                upto = min(count + 1, end)
                idx = upto - start - 1
                n_pts = max(idx + 1, 0)
                if n_pts <= 0:
                    data_updates.append((lines[i], [], []))
                else:
                    data_updates.append((lines[i], seg[:n_pts, 0], -seg[:n_pts, 1]))
        yield data_updates


def render_handwriting_gif(
    deltas: np.ndarray,
    output_path: str | Path,
    title: str = "",
    theme: str = "ink",
    fps: int = 15,
    step: int = 1,
    dpi: int = 110,
    figsize: tuple[float, float] = (8, 3),
) -> Path:
    """Render a stroke sequence as an animated drawing GIF.

    Args:
        deltas: (N, 3) array of (dx, dy, pen_up) relative strokes.
        output_path: destination ``.gif`` path.
        title: optional title text drawn on the figure.
        theme: name of a theme in ``render.THEMES``.
        fps: playback frames per second.
        step: how many points to reveal per frame.
        dpi: figure resolution.
        figsize: matplotlib figure size.

    Returns:
        The written ``Path``.

    Raises:
        ImportError: if Pillow is not installed.
        ValueError: if the stroke sequence contains no points.
    """
    fig, lines, segments, ends, reveals = build_draw_frames(
        deltas, theme=theme, step=step, dpi=dpi, figsize=figsize, title=title
    )

    frames = list(_revealed_frames(lines, segments, ends, reveals))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=fps)
    with writer.saving(fig, str(output_path), dpi=dpi):
        for _i, data_updates in enumerate(frames):
            for line, xs, ys in data_updates:
                line.set_data(xs, ys)
                line.set_visible(len(xs) > 0)
            writer.grab_frame()

    plt.close(fig)
    return output_path


def render_multi_sample_gif(
    deltas_list: Iterable[np.ndarray],
    output_dir: str | Path,
    stem: str = "sample",
    **kwargs,
) -> list[Path]:
    """Render several stroke sequences as separate GIFs.

    Args:
        deltas_list: iterable of (N, 3) stroke sequences.
        output_dir: directory to write the GIFs into.
        stem: filename prefix; files are named ``{stem}_0.gif``, ...
        kwargs: forwarded to :func:`render_handwriting_gif`.

    Returns:
        List of written ``Path`` objects.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for i, deltas in enumerate(deltas_list):
        path = output_dir / f"{stem}_{i}.gif"
        render_handwriting_gif(deltas, path, **kwargs)
        written.append(path)
    return written
