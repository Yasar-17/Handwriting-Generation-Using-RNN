"""Tests for top-k / top-p nucleus sampling utilities."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from sampling import apply_temperature, sample_mixture_component, top_k_filter, top_p_filter


def make_probs():
    rng = np.random.default_rng(0)
    w = rng.random(20) + 0.01
    return w / w.sum()


class TestTopKFilter:
    def test_scales_down_to_k(self):
        probs = make_probs()
        out = top_k_filter(probs, 5)
        assert int((out > 0).sum()) == 5
        assert np.isclose(out.sum(), 1.0)

    def test_returns_input_when_disabled(self):
        probs = make_probs()
        np.testing.assert_allclose(top_k_filter(probs, 0), probs)
        np.testing.assert_allclose(top_k_filter(probs, len(probs)), probs)

    def test_conserves_total_rank_order(self):
        probs = make_probs()
        out = top_k_filter(probs, 4)
        kept = probs[out > 0]
        # The kept components are exactly the 4 largest of the original.
        np.testing.assert_allclose(np.sort(kept)[::-1], np.sort(probs)[-4:][::-1], rtol=1e-6)


class TestTopPFilter:
    def test_renormalizes_and_prunes_tail(self):
        probs = make_probs()
        out = top_p_filter(probs, 0.6)
        assert np.isclose(out.sum(), 1.0)
        assert int((out > 0).sum()) < len(probs)  # tail was pruned

    def test_returns_input_when_disabled(self):
        probs = make_probs()
        np.testing.assert_allclose(top_p_filter(probs, 1.0), probs)
        np.testing.assert_allclose(top_p_filter(probs, 0.0), probs)

    def test_full_distribution_when_p_is_one(self):
        probs = make_probs()
        out = top_p_filter(probs, 0.999999)
        assert np.isclose(out.sum(), 1.0)


class TestSampleMixtureComponent:
    def test_returns_valid_index(self):
        probs = make_probs()
        idx = sample_mixture_component(probs, temperature=0.5)
        assert 0 <= idx < len(probs)

    def test_matches_distribution_over_many_draws(self):
        probs = make_probs()
        # At temperature=1.0 the sampler reproduces softmax(probs) exactly.
        expected = apply_temperature(probs, 1.0)
        rng = np.random.default_rng(7)
        counts = np.zeros(len(probs))
        for _ in range(40_000):
            counts[sample_mixture_component(probs, temperature=1.0, rng=rng)] += 1
        observed = counts / counts.sum()
        assert np.allclose(observed, expected, atol=0.01)

    def test_top_k_restricts_outcome_set(self):
        probs = make_probs()
        top_indices = set(np.argsort(probs)[-4:].tolist())
        rng = np.random.default_rng(0)
        for _ in range(500):
            idx = sample_mixture_component(probs, top_k=4, rng=rng)
            assert idx in top_indices


class TestApplyTemperature:
    def test_preserves_sum_and_supports_frozen_high_temperature(self):
        probs = make_probs()
        cold = apply_temperature(probs, 0.1)
        assert np.isclose(cold.sum(), 1.0)
        # Very high temperature flattens toward uniform.
        hot = apply_temperature(probs, 100.0)
        assert np.isclose(hot.sum(), 1.0)
        assert hot.max() - hot.min() < 0.02
