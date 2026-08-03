"""Tests for the data augmentation pipeline."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from augmentations import (
    AugmentedStrokeDataset,
    Compose,
    GaussianNoise,
    RandomApply,
    RandomRotation,
    RandomScale,
    RandomTimeWarp,
    RandomTranslation,
    StrokeDropout,
    augment_batch,
    get_default_augmentation,
    get_strong_augmentation,
)


@pytest.fixture
def sample_deltas():
    deltas = np.zeros((100, 3), dtype=np.float32)
    deltas[:, 0] = np.random.randn(100) * 2
    deltas[:, 1] = np.random.randn(100) * 2
    deltas[30, 2] = 1.0
    deltas[60, 2] = 1.0
    deltas[99, 2] = 1.0
    return deltas


class TestRandomScale:
    def test_output_shape(self, sample_deltas):
        aug = RandomScale((0.8, 1.2))
        result = aug(sample_deltas)
        assert result.shape == sample_deltas.shape

    def test_pen_up_preserved(self, sample_deltas):
        aug = RandomScale((0.8, 1.2))
        result = aug(sample_deltas)
        np.testing.assert_array_equal(result[:, 2], sample_deltas[:, 2])

    def test_identity_scale(self, sample_deltas):
        aug = RandomScale((1.0, 1.0))
        result = aug(sample_deltas)
        np.testing.assert_array_almost_equal(result[:, :2], sample_deltas[:, :2])


class TestRandomRotation:
    def test_output_shape(self, sample_deltas):
        aug = RandomRotation(max_angle=15.0)
        result = aug(sample_deltas)
        assert result.shape == sample_deltas.shape

    def test_zero_rotation(self, sample_deltas):
        aug = RandomRotation(max_angle=0.0)
        result = aug(sample_deltas)
        np.testing.assert_array_almost_equal(result[:, :2], sample_deltas[:, :2])

    def test_pen_up_preserved(self, sample_deltas):
        aug = RandomRotation(max_angle=30.0)
        result = aug(sample_deltas)
        np.testing.assert_array_equal(result[:, 2], sample_deltas[:, 2])


class TestGaussianNoise:
    def test_output_shape(self, sample_deltas):
        aug = GaussianNoise(std=0.05)
        result = aug(sample_deltas)
        assert result.shape == sample_deltas.shape

    def test_noise_added(self, sample_deltas):
        aug = GaussianNoise(std=1.0)
        result = aug(sample_deltas)
        assert not np.allclose(result[:, :2], sample_deltas[:, :2])

    def test_pen_up_unchanged(self, sample_deltas):
        aug = GaussianNoise(std=0.5)
        result = aug(sample_deltas)
        np.testing.assert_array_equal(result[:, 2], sample_deltas[:, 2])


class TestRandomTimeWarp:
    def test_output_shape_varies(self, sample_deltas):
        aug = RandomTimeWarp((0.5, 1.5))
        result = aug(sample_deltas)
        assert result.shape[1] == 3
        assert len(result) >= 10

    def test_pen_up_binary(self, sample_deltas):
        aug = RandomTimeWarp((0.8, 1.2))
        result = aug(sample_deltas)
        assert set(np.unique(result[:, 2])).issubset({0.0, 1.0})


class TestStrokeDropout:
    def test_output_shape(self, sample_deltas):
        aug = StrokeDropout(dropout_prob=0.1)
        result = aug(sample_deltas)
        assert result.shape == sample_deltas.shape

    def test_some_zeroed(self, sample_deltas):
        aug = StrokeDropout(dropout_prob=0.5)
        result = aug(sample_deltas)
        zeroed = np.sum(np.all(result[:, :2] == 0, axis=1))
        assert zeroed > 0


class TestCompose:
    def test_compose_multiple(self, sample_deltas):
        aug = Compose([RandomScale((0.9, 1.1)), GaussianNoise(std=0.01)])
        result = aug(sample_deltas)
        assert result.shape == sample_deltas.shape

    def test_compose_probability(self, sample_deltas):
        aug = Compose([GaussianNoise(std=100.0)], p=0.0)
        result = aug(sample_deltas)
        np.testing.assert_array_almost_equal(result, sample_deltas)


class TestRandomApply:
    def test_always_apply(self, sample_deltas):
        aug = RandomApply(GaussianNoise(std=1.0), p=1.0)
        result = aug(sample_deltas)
        assert not np.allclose(result[:, :2], sample_deltas[:, :2])

    def test_never_apply(self, sample_deltas):
        aug = RandomApply(GaussianNoise(std=100.0), p=0.0)
        result = aug(sample_deltas)
        np.testing.assert_array_almost_equal(result, sample_deltas)


class TestRandomTranslation:
    def test_noop_for_deltas(self, sample_deltas):
        aug = RandomTranslation(max_shift=10.0)
        result = aug(sample_deltas)
        np.testing.assert_array_almost_equal(result, sample_deltas)


class TestDefaultPipelines:
    def test_default_augmentation(self, sample_deltas):
        aug = get_default_augmentation()
        result = aug(sample_deltas)
        assert result.shape == sample_deltas.shape

    def test_strong_augmentation(self, sample_deltas):
        aug = get_strong_augmentation()
        result = aug(sample_deltas)
        assert result.shape[1] == 3


class TestAugmentBatch:
    def test_batch_augmentation(self):
        data = torch.randn(4, 50, 3)
        mask = torch.ones(4, 50, dtype=torch.bool)
        result = augment_batch(data, mask)
        assert result.shape == data.shape

    def test_short_sequence_skipped(self):
        data = torch.randn(1, 5, 3)
        mask = torch.ones(1, 5, dtype=torch.bool)
        result = augment_batch(data, mask)
        torch.testing.assert_close(result, data)
