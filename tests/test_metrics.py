"""Tests for the stroke-quality metrics module."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from metrics import (
    compare_sample_sets,
    jerk,
    mean_abs_delta,
    mean_speed,
    mean_turning_angle,
    num_strokes,
    pen_up_ratio,
    split_strokes,
    stroke_lengths,
    summarize_sample_set,
    summarize_strokes,
    total_distance,
)


@pytest.fixture
def straight_line():
    """A straight horizontal pen-down run broken into two strokes."""
    pts = np.zeros((30, 3), dtype=np.float32)
    pts[:, 0] = 1.0
    pts[:, 1] = 0.0
    pts[14, 2] = 1.0
    pts[29, 2] = 1.0
    return pts


@pytest.fixture
def zigzag():
    pts = np.zeros((40, 3), dtype=np.float32)
    rng = np.random.default_rng(0)
    pts[:, 0] = rng.normal(loc=0.0, scale=4.0, size=40)
    pts[:, 1] = rng.normal(loc=0.0, scale=4.0, size=40)
    pts[10, 2] = 1.0
    pts[25, 2] = 1.0
    pts[39, 2] = 1.0
    return pts


def test_split_strokes(straight_line):
    strokes = split_strokes(straight_line)
    assert len(strokes) == 2
    assert strokes[0].shape == (15, 3)
    assert strokes[1].shape == (15, 3)


def test_split_strokes_empty():
    assert split_strokes(np.empty((0, 3))) == []


def test_num_strokes(straight_line):
    assert num_strokes(straight_line) == 2


def test_stroke_lengths(straight_line):
    lengths = stroke_lengths(straight_line)
    assert lengths.tolist() == [15, 15]


def test_pen_up_ratio(straight_line):
    # 2 pen-up events out of 30 points
    assert pen_up_ratio(straight_line) == pytest.approx(2.0 / 30.0)


def test_pen_up_ratio_empty():
    assert pen_up_ratio(np.empty((0, 3))) == 0.0


def test_total_distance_straight(straight_line):
    # horizontal dx=1 over 15 pen-down steps, twice
    assert total_distance(straight_line) == pytest.approx(28.0)


def test_total_distance_empty():
    assert total_distance(np.empty((0, 3))) == 0.0


def test_mean_speed(straight_line):
    assert mean_speed(straight_line) == pytest.approx(1.0)


def test_mean_abs_delta_straight(straight_line):
    # |1| + |0| = 1.0 for every pen-down step
    assert mean_abs_delta(straight_line) == pytest.approx(1.0)


def test_jerk_zero_for_constant_velocity(straight_line):
    # Constant deltas have zero second difference -> smooth (low jerk)
    assert jerk(straight_line) == pytest.approx(0.0, abs=1e-6)


def test_jerk_zigzag_greater_than_straight(straight_line, zigzag):
    assert jerk(zigzag) > jerk(straight_line)


def test_mean_turning_angle_straight(straight_line):
    # Straight line has zero turning angle
    assert mean_turning_angle(straight_line) == pytest.approx(0.0, abs=1e-3)


def test_summarize_strokes_keys(straight_line):
    summary = summarize_strokes(straight_line)
    assert set(summary) == {
        "num_strokes",
        "avg_stroke_length",
        "pen_up_ratio",
        "total_distance",
        "mean_speed",
        "mean_abs_delta",
        "jerk",
        "mean_turning_angle",
    }
    assert summary["num_strokes"] == 2
    assert summary["avg_stroke_length"] == 15.0


def test_summarize_sample_set(straight_line, zigzag):
    summary = summarize_sample_set([straight_line, zigzag])
    assert summary["num_samples"] == 2
    assert "jerk" in summary
    assert "mean" in summary["jerk"]
    assert "std" in summary["jerk"]


def test_summarize_sample_set_empty():
    assert summarize_sample_set([]) == {}


def test_compare_sample_sets():
    rng = np.random.default_rng(1)

    def make(seed_variant, scale):
        pts = np.zeros((20, 3), dtype=np.float32)
        pts[:, 0] = rng.normal(scale=scale, size=20)
        pts[:, 1] = rng.normal(scale=scale, size=20)
        pts[19, 2] = 1.0
        return pts

    real = [make(0, 1.0) for _ in range(5)]
    fake = [make(0, 1.5) for _ in range(5)]  # larger displacements
    result = compare_sample_sets(real, fake)
    assert "mean_abs_delta" in result
    assert result["mean_abs_delta"]["abs_diff"] > 0.0
    assert "real" in result["mean_abs_delta"]
    assert "fake" in result["mean_abs_delta"]


def test_compare_sample_sets_empty_fake(straight_line):
    result = compare_sample_sets([straight_line], [])
    assert result == {}
