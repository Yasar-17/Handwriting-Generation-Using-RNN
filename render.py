"""
Advanced rendering utilities for handwriting visualization.

Supports multiple output formats (PNG, SVG) and visual themes.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import relative_to_absolute


@dataclass
class RenderTheme:
    stroke_color: str = "#1a1a2e"
    background_color: str = "#ffffff"
    line_width: float = 1.8
    pen_width: float = 2.2
    show_grid: bool = False
    grid_color: str = "#e0e0e0"
    grid_spacing: float = 50.0
    padding: float = 20.0
    font_family: str = "Georgia, serif"
    title_color: str = "#333333"


THEMES = {
    "classic": RenderTheme(),
    "dark": RenderTheme(
        stroke_color="#e0e0e0",
        background_color="#1a1a2e",
        title_color="#e0e0e0",
        grid_color="#2a2a4e",
    ),
    "blueprint": RenderTheme(
        stroke_color="#0a3d62",
        background_color="#dfe6e9",
        line_width=1.2,
        show_grid=True,
        grid_color="#b2bec3",
        grid_spacing=40.0,
    ),
    "ink": RenderTheme(
        stroke_color="#000000",
        background_color="#faf8f5",
        line_width=2.0,
        pen_width=2.5,
    ),
    "neon": RenderTheme(
        stroke_color="#00ff88",
        background_color="#0d0d0d",
        line_width=2.5,
        pen_width=3.0,
        title_color="#00ff88",
    ),
    "sepia": RenderTheme(
        stroke_color="#5d4e37",
        background_color="#f5e6c8",
        line_width=1.5,
        title_color="#5d4e37",
    ),
}


def render_strokes_svg(
    deltas: np.ndarray,
    title: str = "",
    theme: str | RenderTheme = "classic",
    figsize: tuple[float, float] = (8, 3),
) -> str:
    if isinstance(theme, str):
        theme = THEMES.get(theme, THEMES["classic"])

    abs_pts = relative_to_absolute(deltas)

    if len(abs_pts) == 0:
        return '<svg xmlns="http://www.w3.org/2000/svg"></svg>'

    x_coords = abs_pts[:, 0]
    y_coords = abs_pts[:, 1]
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    pad = theme.padding
    width = (x_max - x_min) + 2 * pad
    height = (y_max - y_min) + 2 * pad

    aspect = figsize[0] / figsize[1]
    data_aspect = width / max(height, 1)

    if data_aspect > aspect:
        svg_width = 800
        svg_height = int(800 / aspect)
        scale = (svg_width - 40) / width
    else:
        svg_height = 300
        svg_width = int(300 * aspect)
        scale = (svg_height - 40) / max(height, 1)

    offset_x = -x_min + pad
    offset_y = -y_min + pad

    svg = ET.Element(
        "svg",
        {
            "xmlns": "http://www.w3.org/2000/svg",
            "width": str(svg_width),
            "height": str(svg_height + (30 if title else 0)),
            "viewBox": f"0 0 {svg_width} {svg_height + (30 if title else 0)}",
        },
    )

    ET.SubElement(
        svg,
        "rect",
        {
            "width": "100%",
            "height": "100%",
            "fill": theme.background_color,
        },
    )

    if theme.show_grid:
        grid_g = ET.SubElement(svg, "g", {"opacity": "0.5"})
        gs = theme.grid_spacing * scale
        x = 0
        while x < svg_width:
            ET.SubElement(
                grid_g,
                "line",
                {
                    "x1": str(x),
                    "y1": "0",
                    "x2": str(x),
                    "y2": str(svg_height),
                    "stroke": theme.grid_color,
                    "stroke-width": "0.5",
                },
            )
            x += gs
        y = 0
        while y < svg_height:
            ET.SubElement(
                grid_g,
                "line",
                {
                    "x1": "0",
                    "y1": str(y),
                    "x2": str(svg_width),
                    "y2": str(y),
                    "stroke": theme.grid_color,
                    "stroke-width": "0.5",
                },
            )
            y += gs

    stroke_start = 0
    for i in range(len(abs_pts)):
        if abs_pts[i, 2] == 1:
            _add_stroke_path(svg, abs_pts[stroke_start : i + 1], offset_x, offset_y, scale, svg_height, theme)
            stroke_start = i + 1

    if stroke_start < len(abs_pts):
        _add_stroke_path(svg, abs_pts[stroke_start:], offset_x, offset_y, scale, svg_height, theme)

    if title:
        text_elem = ET.SubElement(
            svg,
            "text",
            {
                "x": str(svg_width / 2),
                "y": str(svg_height + 20),
                "text-anchor": "middle",
                "font-family": theme.font_family,
                "font-size": "14",
                "fill": theme.title_color,
            },
        )
        text_elem.text = title

    return ET.tostring(svg, encoding="unicode", xml_declaration=True)


def _add_stroke_path(svg, points, offset_x, offset_y, scale, svg_height, theme):
    if len(points) < 2:
        return

    path_d = ""
    for i, pt in enumerate(points):
        sx = (pt[0] + offset_x) * scale
        sy = svg_height - (pt[1] + offset_y) * scale
        if i == 0:
            path_d += f"M {sx:.2f} {sy:.2f}"
        else:
            path_d += f" L {sx:.2f} {sy:.2f}"

    ET.SubElement(
        svg,
        "path",
        {
            "d": path_d,
            "stroke": theme.stroke_color,
            "stroke-width": str(theme.line_width),
            "fill": "none",
            "stroke-linecap": "round",
            "stroke-linejoin": "round",
        },
    )


def render_strokes_themed(
    deltas: np.ndarray,
    title: str = "",
    theme: str | RenderTheme = "classic",
    figsize: tuple[float, float] = (8, 3),
    dpi: int = 150,
) -> plt.Figure:
    if isinstance(theme, str):
        theme = THEMES.get(theme, THEMES["classic"])

    abs_pts = relative_to_absolute(deltas)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(theme.background_color)
    ax.set_facecolor(theme.background_color)

    if theme.show_grid:
        ax.grid(True, color=theme.grid_color, linewidth=0.5, alpha=0.7)

    stroke_start = 0
    for i in range(len(abs_pts)):
        if abs_pts[i, 2] == 1:
            stroke = abs_pts[stroke_start : i + 1]
            ax.plot(
                stroke[:, 0],
                -stroke[:, 1],
                color=theme.stroke_color,
                linewidth=theme.line_width,
                solid_capstyle="round",
                solid_joinstyle="round",
            )
            stroke_start = i + 1

    if stroke_start < len(abs_pts):
        stroke = abs_pts[stroke_start:]
        ax.plot(
            stroke[:, 0],
            -stroke[:, 1],
            color=theme.stroke_color,
            linewidth=theme.line_width,
            solid_capstyle="round",
            solid_joinstyle="round",
        )

    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, color=theme.title_color, fontsize=12, pad=10)

    fig.tight_layout(pad=0)
    return fig


def render_multi_sample(
    deltas_list: list[np.ndarray],
    texts: list[str] | None = None,
    theme: str = "classic",
    figsize: tuple[float, float] = (16, 10),
    dpi: int = 150,
) -> plt.Figure:
    n = len(deltas_list)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    t = THEMES.get(theme, THEMES["classic"])
    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(t.background_color)

    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    for i, deltas in enumerate(deltas_list):
        ax = axes[i]
        ax.set_facecolor(t.background_color)

        abs_pts = relative_to_absolute(deltas)
        stroke_start = 0
        for j in range(len(abs_pts)):
            if abs_pts[j, 2] == 1:
                stroke = abs_pts[stroke_start : j + 1]
                ax.plot(stroke[:, 0], -stroke[:, 1], color=t.stroke_color, linewidth=t.line_width)
                stroke_start = j + 1

        if stroke_start < len(abs_pts):
            stroke = abs_pts[stroke_start:]
            ax.plot(stroke[:, 0], -stroke[:, 1], color=t.stroke_color, linewidth=t.line_width)

        ax.set_aspect("equal")
        ax.axis("off")
        if texts and i < len(texts):
            ax.set_title(texts[i], color=t.title_color, fontsize=11)

    for i in range(n, len(axes)):
        axes[i].axis("off")
        axes[i].set_facecolor(t.background_color)

    fig.tight_layout(pad=2)
    return fig


def save_svg(
    deltas: np.ndarray,
    output_path: str | Path,
    title: str = "",
    theme: str = "classic",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    svg_content = render_strokes_svg(deltas, title=title, theme=theme)
    output_path.write_text(svg_content, encoding="utf-8")
    return output_path


def render_comparison_grid(
    deltas_dict: dict[str, list[np.ndarray]],
    theme: str = "classic",
    figsize: tuple[float, float] = (14, 8),
) -> plt.Figure:
    t = THEMES.get(theme, THEMES["classic"])
    labels = list(deltas_dict.keys())
    max_samples = max(len(v) for v in deltas_dict.values())

    fig, axes = plt.subplots(len(labels), max_samples, figsize=figsize)
    fig.patch.set_facecolor(t.background_color)

    if len(labels) == 1:
        axes = np.array([axes])
    if max_samples == 1:
        axes = axes.reshape(-1, 1)

    for i, label in enumerate(labels):
        for j, deltas in enumerate(deltas_dict[label]):
            ax = axes[i, j]
            ax.set_facecolor(t.background_color)

            abs_pts = relative_to_absolute(deltas)
            stroke_start = 0
            for k in range(len(abs_pts)):
                if abs_pts[k, 2] == 1:
                    stroke = abs_pts[stroke_start : k + 1]
                    ax.plot(stroke[:, 0], -stroke[:, 1], color=t.stroke_color, linewidth=t.line_width)
                    stroke_start = k + 1
            if stroke_start < len(abs_pts):
                stroke = abs_pts[stroke_start:]
                ax.plot(stroke[:, 0], -stroke[:, 1], color=t.stroke_color, linewidth=t.line_width)

            ax.set_aspect("equal")
            ax.axis("off")
            if j == 0:
                ax.set_ylabel(label, color=t.title_color, fontsize=12, rotation=0, labelpad=80)

    fig.tight_layout(pad=1)
    return fig
