"""Tests for the animated handwriting GIF rendering module."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from render_animation import (
    _revealed_frames,
    _split_segments,
    build_draw_frames,
    render_handwriting_gif,
    render_multi_sample_gif,
)


@pytest.fixture
def sample_deltas():
    pts = np.zeros((40, 3), dtype=np.float32)
    pts[:, 0] = 1.0
    pts[:, 1] = 0.5
    pts[14, 2] = 1.0
    pts[29, 2] = 1.0
    pts[39, 2] = 1.0
    return pts


@pytest.fixture
def empty_deltas():
    return np.empty((0, 3), dtype=np.float32)


class TestSplitSegments:
    def test_split_to_three(self, sample_deltas):
        segs = _split_segments(sample_deltas)
        assert len(segs) == 3
        assert [len(s) for s in segs] == [15, 15, 10]

    def test_empty(self, empty_deltas):
        assert _split_segments(empty_deltas) == []


class TestBuildDrawFrames:
    def test_returns_lines_and_reveals(self, sample_deltas):
        _fig, lines, _segments, ends, reveals = build_draw_frames(sample_deltas, step=1)
        assert len(lines) == 3
        assert ends[-1] == 40
        assert reveals[0] == 0
        assert reveals[-1] == 39

    def test_empty_raises(self, empty_deltas):
        with pytest.raises(ValueError):
            build_draw_frames(empty_deltas)

    def test_revealed_frames_count(self, sample_deltas):
        _fig, lines, segments, ends, reveals = build_draw_frames(sample_deltas, step=5)
        frames = list(_revealed_frames(lines, segments, ends, reveals))
        assert len(frames) == len(reveals)


class TestRenderGif:
    def test_gif_written(self, sample_deltas):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = render_handwriting_gif(sample_deltas, Path(tmpdir) / "out.gif", fps=10, step=2)
            assert path.exists()
            assert path.read_bytes()[:6] == b"GIF89a"

    def test_gif_with_title_and_theme(self, sample_deltas):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = render_handwriting_gif(sample_deltas, Path(tmpdir) / "t.gif", title="Demo", theme="neon")
            assert path.exists()
            assert path.stat().st_size > 0

    def test_empty_raises(self, empty_deltas):
        with tempfile.TemporaryDirectory() as tmpdir, pytest.raises(ValueError):
            render_handwriting_gif(empty_deltas, Path(tmpdir) / "out.gif")


class TestRenderMultiGif:
    def test_multiple_gifs(self, sample_deltas):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = render_multi_sample_gif([sample_deltas, sample_deltas * 0.5], Path(tmpdir), stem="s")
            assert len(paths) == 2
            assert all(p.read_bytes()[:6] == b"GIF89a" for p in paths)
