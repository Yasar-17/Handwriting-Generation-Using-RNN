"""
Quantitative stroke-quality metrics for evaluating generated handwriting.

These are distribution-level summaries computed over stroke sequences
(relative deltas of shape (N, 3) where the last column is the binary
``pen_up`` flag). They let you compare models quantitatively — e.g. the
GAN-refined model vs. the plain MDN-RNN, or generated data vs. the real
training data — rather than relying only on the negative log-likelihood.

Available metrics:

* ``num_strokes`` / ``stroke_lengths`` — segmentation of the trajectory
  into pen-lift-delimited strokes.
* ``pen_up_ratio`` — fraction of timesteps where the pen is lifted.
* ``total_distance`` / ``mean_speed`` — ink-drawing distance and speed.
* ``mean_abs_delta`` — average displacement magnitude (stroke "size").
* ``jerk`` — mean absolute second difference of displacement; a proxy for
  temporal smoothness (low jerk = smooth, high jerk = jittery).
* ``mean_turning_angle`` — average absolute angle between consecutive
  pen-down displacement vectors; a proxy for stroke curvature.

The public entry points are :func:`summarize_strokes`,
:func:`summarize_sample_set`, and :func:`compare_sample_sets`.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def split_strokes(deltas: np.ndarray) -> list[np.ndarray]:
    """Split a stroke sequence into pen-lift-delimited segments.

    Args:
        deltas: (N, 3) array of (dx, dy, pen_up). The point carrying the
            ``pen_up == 1`` flag is included as the final point of its
            stroke (it is where the pen lifts).

    Returns:
        List of (L_i, 3) arrays, one per stroke, preserving order.
    """
    deltas = np.asarray(deltas)
    if deltas.size == 0:
        return []

    strokes: list[np.ndarray] = []
    start = 0
    for i in range(deltas.shape[0]):
        if deltas[i, 2] == 1:
            strokes.append(deltas[start : i + 1])
            start = i + 1
    if start < deltas.shape[0]:
        strokes.append(deltas[start:])
    return strokes


def num_strokes(deltas: np.ndarray) -> int:
    """Number of pen-lift-delimited strokes in the sequence."""
    return len(split_strokes(deltas))


def stroke_lengths(deltas: np.ndarray) -> np.ndarray:
    """Number of points in each stroke (empty array if no strokes)."""
    return np.asarray([s.shape[0] for s in split_strokes(deltas)], dtype=np.int64)


def pen_up_ratio(deltas: np.ndarray) -> float:
    """Fraction of timesteps where the pen is lifted (0.0 if empty)."""
    deltas = np.asarray(deltas)
    if deltas.shape[0] == 0:
        return 0.0
    return float(deltas[:, 2].mean())


def total_distance(deltas: np.ndarray) -> float:
    """Total Euclidean ink-drawing distance (pen-down steps only)."""
    deltas = np.asarray(deltas)
    if deltas.shape[0] == 0:
        return 0.0
    pen_down = deltas[:, 2] < 1.0
    if not pen_down.any():
        return 0.0
    seg = deltas[pen_down, :2].astype(np.float64)
    return float(np.linalg.norm(seg, axis=1).sum())


def mean_speed(deltas: np.ndarray) -> float:
    """Mean per-step displacement magnitude over pen-down steps (0.0 if none)."""
    deltas = np.asarray(deltas)
    if deltas.shape[0] == 0:
        return 0.0
    pen_down = deltas[:, 2] < 1.0
    if not pen_down.any():
        return 0.0
    seg = deltas[pen_down, :2].astype(np.float64)
    return float(np.linalg.norm(seg, axis=1).mean())


def mean_abs_delta(deltas: np.ndarray) -> float:
    """Mean of |dx| + |dy| over pen-down steps (0.0 if none)."""
    deltas = np.asarray(deltas)
    if deltas.shape[0] == 0:
        return 0.0
    pen_down = deltas[:, 2] < 1.0
    if not pen_down.any():
        return 0.0
    return float(np.abs(deltas[pen_down, :2]).sum(axis=1).mean())


def jerk(deltas: np.ndarray) -> float:
    """Mean absolute second difference of displacement (temporal smoothness).

    Low values indicate smooth trajectories; high values indicate jittery,
    accelerated motion. Computed on pen-down steps only.
    """
    deltas = np.asarray(deltas)
    if deltas.shape[0] < 3:
        return 0.0
    pen_down = deltas[:, 2] < 1.0
    if pen_down.sum() < 3:
        return 0.0
    seg = deltas[pen_down, :2].astype(np.float64)
    second_diff = np.abs(np.diff(seg, n=2, axis=0))
    return float(second_diff.mean())


def mean_turning_angle(deltas: np.ndarray) -> float:
    """Mean absolute turning angle (radians) between consecutive pen-down segments.

    Returns the average angle between successive displacement vectors; a value
    of 0 corresponds to perfectly straight strokes. Returns 0.0 when there are
    fewer than 2 usable segments.
    """
    deltas = np.asarray(deltas)
    if deltas.shape[0] < 3:
        return 0.0
    pen_down = deltas[:, 2] < 1.0
    if pen_down.sum() < 2:
        return 0.0
    vecs = deltas[pen_down, :2].astype(np.float64)
    norms = np.linalg.norm(vecs, axis=1)
    mask = norms > 1e-12
    if mask.sum() < 2:
        return 0.0
    vecs = vecs[mask]
    norms = norms[mask]
    unit = vecs / norms[:, None]
    dot = (unit[:-1] * unit[1:]).sum(axis=1).clip(-1.0, 1.0)
    angles = np.arccos(dot)
    return float(np.abs(angles).mean())


def summarize_strokes(deltas: np.ndarray) -> dict[str, Any]:
    """Compute all stroke-quality metrics for a single sequence.

    Args:
        deltas: (N, 3) array of (dx, dy, pen_up).

    Returns:
        dict with keys ``num_strokes``, ``avg_stroke_length``,
        ``pen_up_ratio``, ``total_distance``, ``mean_speed``,
        ``mean_abs_delta``, ``jerk``, ``mean_turning_angle``.
    """
    lengths = stroke_lengths(deltas)
    return {
        "num_strokes": int(lengths.shape[0]),
        "avg_stroke_length": float(lengths.mean()) if lengths.shape[0] else 0.0,
        "pen_up_ratio": pen_up_ratio(deltas),
        "total_distance": total_distance(deltas),
        "mean_speed": mean_speed(deltas),
        "mean_abs_delta": mean_abs_delta(deltas),
        "jerk": jerk(deltas),
        "mean_turning_angle": mean_turning_angle(deltas),
    }


def summarize_sample_set(deltas_list: list[np.ndarray]) -> dict[str, Any]:
    """Aggregate metrics across a set of samples.

    Returns mean and standard deviation of each metric across samples,
    suitable for comparing two models or real vs. generated data.
    """
    per_sample = [summarize_strokes(d) for d in deltas_list]
    if not per_sample:
        return {}

    keys = list(per_sample[0].keys())
    aggregated: dict[str, Any] = {"num_samples": len(per_sample)}
    for key in keys:
        values = np.asarray([s[key] for s in per_sample], dtype=np.float64)
        aggregated[key] = {"mean": float(values.mean()), "std": float(values.std())}
    return aggregated


def compare_sample_sets(
    real_list: list[np.ndarray],
    fake_list: list[np.ndarray],
) -> dict[str, dict[str, float]]:
    """Compare two sets of stroke sequences metric-by-metric.

    Args:
        real_list: reference samples (e.g. real data or the better model).
        fake_list: candidate samples to compare (e.g. a weaker model).

    Returns:
        dict mapping each metric name to ``{"real": mean, "fake": mean,
        "abs_diff": fake - real}``. Only metrics whose keys exist in the
        summaries are included.
    """
    real_summary = summarize_sample_set(real_list)
    fake_summary = summarize_sample_set(fake_list)

    results: dict[str, dict[str, float]] = {}
    metric_keys = {key for key in fake_summary if key != "num_samples" and key in real_summary}
    for key in sorted(metric_keys):
        real_mean = real_summary[key]["mean"]
        fake_mean = fake_summary[key]["mean"]
        results[key] = {
            "real": real_mean,
            "fake": fake_mean,
            "abs_diff": fake_mean - real_mean,
        }
    return results
