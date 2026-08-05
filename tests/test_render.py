"""Tests for the rendering and SVG export module."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from render import (
    THEMES,
    RenderTheme,
    render_comparison_grid,
    render_multi_sample,
    render_strokes_svg,
    render_strokes_themed,
    save_svg,
)


@pytest.fixture
def sample_deltas():
    deltas = np.zeros((50, 3), dtype=np.float32)
    deltas[:, 0] = np.random.randn(50) * 2
    deltas[:, 1] = np.random.randn(50) * 2
    deltas[20, 2] = 1.0
    deltas[49, 2] = 1.0
    return deltas


@pytest.fixture
def empty_deltas():
    return np.empty((0, 3), dtype=np.float32)


class TestRenderTheme:
    def test_default_theme(self):
        theme = RenderTheme()
        assert theme.stroke_color == "#1a1a2e"
        assert theme.background_color == "#ffffff"
        assert theme.line_width > 0

    def test_custom_theme(self):
        theme = RenderTheme(stroke_color="red", line_width=3.0)
        assert theme.stroke_color == "red"
        assert theme.line_width == 3.0

    def test_predefined_themes(self):
        assert "classic" in THEMES
        assert "dark" in THEMES
        assert "blueprint" in THEMES
        assert "ink" in THEMES
        assert "neon" in THEMES
        assert "sepia" in THEMES
        assert len(THEMES) >= 6


class TestSVGRendering:
    def test_svg_output(self, sample_deltas):
        svg = render_strokes_svg(sample_deltas)
        assert svg.startswith("<?xml")
        assert "<svg" in svg
        assert "<path" in svg

    def test_svg_with_title(self, sample_deltas):
        svg = render_strokes_svg(sample_deltas, title="Test Title")
        assert "Test Title" in svg

    def test_svg_with_theme_string(self, sample_deltas):
        svg = render_strokes_svg(sample_deltas, theme="dark")
        assert "<svg" in svg

    def test_svg_with_theme_object(self, sample_deltas):
        theme = RenderTheme(stroke_color="blue", line_width=3.0)
        svg = render_strokes_svg(sample_deltas, theme=theme)
        assert "blue" in svg

    def test_svg_empty_deltas(self, empty_deltas):
        svg = render_strokes_svg(empty_deltas)
        assert "<svg" in svg

    def test_svg_with_grid(self, sample_deltas):
        theme = THEMES["blueprint"]
        svg = render_strokes_svg(sample_deltas, theme=theme)
        assert "<line" in svg

    def test_save_svg(self, sample_deltas):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_svg(sample_deltas, Path(tmpdir) / "test.svg", title="Save Test")
            assert path.exists()
            content = path.read_text()
            assert "<svg" in content


class TestThemedRendering:
    def test_themed_figure(self, sample_deltas):
        fig = render_strokes_themed(sample_deltas, theme="classic")
        assert fig is not None

    def test_themed_with_title(self, sample_deltas):
        fig = render_strokes_themed(sample_deltas, title="Themed", theme="dark")
        assert fig is not None

    def test_all_themes(self, sample_deltas):
        for theme_name in THEMES:
            fig = render_strokes_themed(sample_deltas, theme=theme_name)
            assert fig is not None


class TestMultiSampleRendering:
    def test_multi_sample(self, sample_deltas):
        deltas_list = [sample_deltas, sample_deltas * 0.5, sample_deltas * 1.5]
        fig = render_multi_sample(deltas_list)
        assert fig is not None

    def test_multi_sample_with_texts(self, sample_deltas):
        deltas_list = [sample_deltas, sample_deltas]
        texts = ["Sample A", "Sample B"]
        fig = render_multi_sample(deltas_list, texts=texts)
        assert fig is not None


class TestComparisonGrid:
    def test_comparison_grid(self, sample_deltas):
        deltas_dict = {
            "Model A": [sample_deltas, sample_deltas * 0.8],
            "Model B": [sample_deltas * 1.2, sample_deltas * 0.6],
        }
        fig = render_comparison_grid(deltas_dict)
        assert fig is not None

    def test_single_model_comparison(self, sample_deltas):
        deltas_dict = {"Only Model": [sample_deltas]}
        fig = render_comparison_grid(deltas_dict)
        assert fig is not None
