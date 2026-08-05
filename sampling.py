"""
Modern decoding strategies for handwriting generation.

Temperature scaling, top-k filtering and top-p (nucleus) filtering are
borrowed from large-language-model decoding. Applied to the MDN mixture
weights they give fine-grained control over the generated strokes:

* ``temperature`` — flattens or sharpens the mixture distribution.
* ``top_k`` — keep only the ``k`` most probable components.
* ``top_p`` — keep only the smallest set of components whose cumulative
  probability reaches ``p`` (nucleus sampling).

Lower values make output more deterministic; higher values add variety.
"""

import numpy as np


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Sharpen/flatten a probability vector by dividing logits by temperature."""
    w = np.asarray(probs, dtype=np.float64) / max(float(temperature), 1e-6)
    w = w - w.max()
    exp_w = np.exp(w)
    return exp_w / exp_w.sum()


def top_k_filter(probs: np.ndarray, k: int) -> np.ndarray:
    """Zero out all but the ``k`` most probable components and renormalize.

    A value of ``k <= 0`` or ``k >= len(probs)`` returns the input unchanged.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if k <= 0 or k >= probs.shape[-1]:
        return probs
    threshold = np.partition(probs, -k)[-k]
    filtered = np.where(probs >= threshold, probs, 0.0)
    total = filtered.sum()
    if total <= 0.0:
        return probs
    return filtered / total


def top_p_filter(probs: np.ndarray, p: float) -> np.ndarray:
    """Keep the smallest set of components with cumulative probability >= p.

    This is nucleus sampling: low-probability "tail" components are pruned
    and the survivors are renormalized. ``p`` outside (0, 1) returns the
    input unchanged.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if not 0.0 < p < 1.0:
        return probs

    order = np.argsort(probs)[::-1]
    sorted_probs = probs[order]
    cumulative = np.cumsum(sorted_probs)

    # Remove components once the cumulative mass passes p, keeping at least one.
    remove = cumulative > p
    remove[1:] = remove[:-1]
    remove[0] = False

    keep = np.zeros_like(probs, dtype=bool)
    keep[order] = ~remove
    filtered = np.where(keep, probs, 0.0)
    total = filtered.sum()
    if total <= 0.0:
        return probs
    return filtered / total


def sample_mixture_component(
    probs: np.ndarray,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    rng: np.random.Generator | None = None,
) -> int:
    """Sample a mixture component index under temperature + top-k/top-p.

    Args:
        probs: (M,) raw mixture weights (softmaxed probabilities).
        temperature: sharpens (< 1) or flattens (> 1) the distribution.
        top_k: if > 0, restrict to the k most probable components.
        top_p: if in (0, 1), nucleus-filter to the top cumulative mass p.
        rng: optional numpy RNG; defaults to ``np.random``.

    Returns:
        Sampled component index in ``[0, M)``.
    """
    weights = apply_temperature(probs, temperature)
    if top_k > 0:
        weights = top_k_filter(weights, top_k)
    if 0.0 < top_p < 1.0:
        weights = top_p_filter(weights, top_p)

    rng = rng if rng is not None else np.random
    return int(rng.choice(weights.shape[-1], p=weights))
