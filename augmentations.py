"""
Data augmentation utilities for handwriting stroke sequences.

Provides augmentation transforms for online handwriting data including:
- Random scaling
- Random rotation
- Random translation
- Gaussian noise injection
- Random time warping
- Stroke dropout
- Composition of multiple augmentations
"""

import numpy as np
import torch


class StrokeAugmentation:
    """Base class for stroke sequence augmentations."""

    def __call__(self, deltas: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class RandomScale(StrokeAugmentation):
    """Randomly scale the stroke sequence.

    Args:
        scale_range: Tuple of (min_scale, max_scale) for uniform sampling.
    """

    def __init__(self, scale_range: tuple[float, float] = (0.8, 1.2)):
        self.scale_range = scale_range

    def __call__(self, deltas: np.ndarray) -> np.ndarray:
        out = deltas.copy()
        scale = np.random.uniform(*self.scale_range)
        out[:, 0] *= scale
        out[:, 1] *= scale
        return out


class RandomRotation(StrokeAugmentation):
    """Randomly rotate the stroke sequence around the origin.

    Args:
        max_angle: Maximum rotation angle in degrees.
    """

    def __init__(self, max_angle: float = 15.0):
        self.max_angle = max_angle

    def __call__(self, deltas: np.ndarray) -> np.ndarray:
        out = deltas.copy()
        angle = np.random.uniform(-self.max_angle, self.max_angle)
        theta = np.radians(angle)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        dx = out[:, 0]
        dy = out[:, 1]
        out[:, 0] = dx * cos_t - dy * sin_t
        out[:, 1] = dx * sin_t + dy * cos_t
        return out


class RandomTranslation(StrokeAugmentation):
    """Add random translation to the first point (shifts entire sequence).

    Note: Since we work with deltas, this only affects the implicit starting
    position. For delta sequences, this is a no-op unless applied to absolute
    coordinates. Kept for API completeness.

    Args:
        max_shift: Maximum shift in each dimension.
    """

    def __init__(self, max_shift: float = 10.0):
        self.max_shift = max_shift

    def __call__(self, deltas: np.ndarray) -> np.ndarray:
        return deltas.copy()


class GaussianNoise(StrokeAugmentation):
    """Add Gaussian noise to the stroke deltas.

    Args:
        std: Standard deviation of the noise.
    """

    def __init__(self, std: float = 0.05):
        self.std = std

    def __call__(self, deltas: np.ndarray) -> np.ndarray:
        out = deltas.copy()
        noise = np.random.normal(0, self.std, size=out[:, :2].shape)
        out[:, :2] += noise
        return out


class RandomTimeWarp(StrokeAugmentation):
    """Randomly subsample or upsample the sequence (simulates writing speed variation).

    Args:
        warp_range: Tuple of (min_ratio, max_ratio) for sequence length change.
    """

    def __init__(self, warp_range: tuple[float, float] = (0.8, 1.2)):
        self.warp_range = warp_range

    def __call__(self, deltas: np.ndarray) -> np.ndarray:
        ratio = np.random.uniform(*self.warp_range)
        original_len = len(deltas)
        new_len = max(10, int(original_len * ratio))

        if new_len == original_len:
            return deltas.copy()

        indices = np.linspace(0, original_len - 1, new_len)
        indices_int = indices.astype(int)
        indices_frac = indices - indices_int

        out = deltas[indices_int].copy()
        if new_len > 1:
            next_idx = np.minimum(indices_int + 1, original_len - 1)
            out[:, :2] = (1 - indices_frac[:, None]) * deltas[indices_int, :2] + indices_frac[:, None] * deltas[
                next_idx, :2
            ]

        out[:, 2] = np.where(out[:, 2] > 0.5, 1.0, 0.0)
        return out


class StrokeDropout(StrokeAugmentation):
    """Randomly drop (zero out) a fraction of stroke points.

    Args:
        dropout_prob: Probability of dropping each point.
    """

    def __init__(self, dropout_prob: float = 0.05):
        self.dropout_prob = dropout_prob

    def __call__(self, deltas: np.ndarray) -> np.ndarray:
        out = deltas.copy()
        mask = np.random.random(len(out)) > self.dropout_prob
        out[~mask, :2] = 0
        return out


class Compose(StrokeAugmentation):
    """Compose multiple augmentations.

    Args:
        transforms: List of augmentation transforms to apply sequentially.
        p: Probability of applying the entire composition.
    """

    def __init__(self, transforms: list[StrokeAugmentation], p: float = 1.0):
        self.transforms = transforms
        self.p = p

    def __call__(self, deltas: np.ndarray) -> np.ndarray:
        if np.random.random() > self.p:
            return deltas.copy()
        out = deltas.copy()
        for t in self.transforms:
            out = t(out)
        return out


class RandomApply(StrokeAugmentation):
    """Randomly apply a single augmentation with given probability.

    Args:
        transform: Augmentation to apply.
        p: Probability of applying the transform.
    """

    def __init__(self, transform: StrokeAugmentation, p: float = 0.5):
        self.transform = transform
        self.p = p

    def __call__(self, deltas: np.ndarray) -> np.ndarray:
        if np.random.random() < self.p:
            return self.transform(deltas)
        return deltas.copy()


def get_default_augmentation() -> Compose:
    """Return a default augmentation pipeline for handwriting data.

    Includes mild scaling, rotation, and noise to simulate natural variation.
    """
    return Compose(
        [
            RandomApply(RandomScale((0.9, 1.1)), p=0.5),
            RandomApply(RandomRotation(max_angle=10.0), p=0.5),
            RandomApply(GaussianNoise(std=0.03), p=0.5),
        ]
    )


def get_strong_augmentation() -> Compose:
    """Return a stronger augmentation pipeline for regularization.

    Includes more aggressive transforms including time warping and dropout.
    """
    return Compose(
        [
            RandomApply(RandomScale((0.7, 1.3)), p=0.7),
            RandomApply(RandomRotation(max_angle=20.0), p=0.7),
            RandomApply(GaussianNoise(std=0.08), p=0.7),
            RandomApply(RandomTimeWarp((0.7, 1.3)), p=0.3),
            RandomApply(StrokeDropout(dropout_prob=0.05), p=0.3),
        ]
    )


class AugmentedStrokeDataset:
    """Wrapper dataset that applies augmentations to stroke sequences.

    Args:
        dataset: Base IAMStrokeDataset or IAMConditionedDataset.
        augmentation: Augmentation pipeline to apply.
    """

    def __init__(
        self,
        dataset,
        augmentation: StrokeAugmentation | None = None,
    ):
        self.dataset = dataset
        self.augmentation = augmentation or get_default_augmentation()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        if isinstance(item, np.ndarray):
            return self.augmentation(item)
        elif isinstance(item, tuple):
            char_ids, deltas = item
            return (char_ids, self.augmentation(deltas))
        return item


def augment_batch(
    data: torch.Tensor,
    mask: torch.Tensor,
    augmentation: StrokeAugmentation | None = None,
) -> torch.Tensor:
    """Apply augmentation to a batch of stroke sequences.

    Args:
        data: Tensor of shape (B, T, 3) with stroke deltas.
        mask: Boolean mask of shape (B, T).
        augmentation: Augmentation to apply. If None, uses default.

    Returns:
        Augmented data tensor of same shape.
    """
    aug = augmentation or get_default_augmentation()
    B, _T, _ = data.shape
    augmented = data.clone()

    for b in range(B):
        valid_len = mask[b].sum().item()
        if valid_len < 10:
            continue
        deltas = data[b, :valid_len].cpu().numpy()
        aug_deltas = aug(deltas)
        augmented[b, :valid_len] = torch.from_numpy(aug_deltas).to(data.device)

    return augmented
