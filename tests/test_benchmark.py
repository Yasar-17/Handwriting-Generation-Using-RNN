"""Tests for the benchmarking suite."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark import (
    benchmark_autoregressive_sampling,
    benchmark_discriminator,
    benchmark_forward_pass,
    count_parameters,
    model_size_mb,
)
from data import CharVocab
from models import MDNRNN, MDNRNNConditioned


class TestParameterCounting:
    def test_mdnrnn_parameters(self):
        model = MDNRNN(input_dim=3, hidden_dim=64, num_layers=2, num_mixtures=10)
        total, trainable = count_parameters(model)
        assert total > 0
        assert trainable == total

    def test_conditioned_parameters(self):
        vocab = CharVocab()
        model = MDNRNNConditioned(
            input_dim=3,
            hidden_dim=64,
            num_layers=2,
            num_mixtures=10,
            char_vocab_size=len(vocab),
            char_embed_dim=16,
        )
        total, _trainable = count_parameters(model)
        assert total > 0


class TestModelSize:
    def test_model_size_positive(self):
        model = MDNRNN(input_dim=3, hidden_dim=64, num_mixtures=10)
        size = model_size_mb(model)
        assert size > 0

    def test_larger_model_larger_size(self):
        small = MDNRNN(input_dim=3, hidden_dim=32, num_mixtures=5)
        large = MDNRNN(input_dim=3, hidden_dim=256, num_mixtures=20)
        assert model_size_mb(large) > model_size_mb(small)


class TestBenchmarkForward:
    def test_unconditional_forward(self):
        model = MDNRNN(input_dim=3, hidden_dim=64, num_layers=2, num_mixtures=10)
        latency, throughput = benchmark_forward_pass(
            model,
            batch_size=2,
            seq_len=50,
            num_warmup=2,
            num_runs=5,
        )
        assert latency > 0
        assert throughput > 0

    def test_conditioned_forward(self):
        vocab = CharVocab()
        model = MDNRNNConditioned(
            input_dim=3,
            hidden_dim=64,
            num_layers=2,
            num_mixtures=10,
            char_vocab_size=len(vocab),
            char_embed_dim=16,
        )
        latency, _throughput = benchmark_forward_pass(
            model,
            batch_size=2,
            seq_len=50,
            num_warmup=2,
            num_runs=5,
            conditioned=True,
        )
        assert latency > 0


class TestBenchmarkSampling:
    def test_unconditional_sampling(self):
        model = MDNRNN(input_dim=3, hidden_dim=64, num_layers=2, num_mixtures=10)
        latency = benchmark_autoregressive_sampling(
            model,
            seq_len=20,
            num_runs=2,
        )
        assert latency > 0

    def test_conditioned_sampling(self):
        vocab = CharVocab()
        model = MDNRNNConditioned(
            input_dim=3,
            hidden_dim=64,
            num_layers=2,
            num_mixtures=10,
            char_vocab_size=len(vocab),
            char_embed_dim=16,
        )
        latency = benchmark_autoregressive_sampling(
            model,
            seq_len=20,
            num_runs=2,
            conditioned=True,
        )
        assert latency > 0


class TestDiscriminatorBenchmark:
    def test_discriminator_benchmark(self):
        result = benchmark_discriminator(
            hidden_dim=32,
            num_layers=2,
            batch_size=2,
            seq_len=50,
        )
        assert result["parameters"] > 0
        assert result["forward_latency_ms"] > 0
        assert result["size_mb"] > 0
