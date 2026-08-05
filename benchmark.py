"""
Benchmarking suite for MDN-RNN handwriting generation models.

Measures inference latency, throughput, memory usage, and model size.
"""

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from data import CharVocab
from models import MDNRNN, MDNRNNConditioned, SequenceDiscriminator


@dataclass
class BenchmarkResult:
    model_name: str
    num_parameters: int
    trainable_parameters: int
    model_size_mb: float
    device: str
    batch_size: int
    seq_len: int
    forward_latency_ms: float
    forward_throughput_seqs_per_sec: float
    sampling_latency_ms_per_step: float
    peak_memory_mb: float
    num_mixtures: int
    num_layers: int
    hidden_dim: int


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_size_mb(model: torch.nn.Module) -> float:
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)


@torch.no_grad()
def benchmark_forward_pass(
    model: torch.nn.Module,
    batch_size: int = 8,
    seq_len: int = 200,
    num_warmup: int = 10,
    num_runs: int = 50,
    conditioned: bool = False,
    device: torch.device | None = None,
) -> tuple[float, float]:
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    x = torch.randn(batch_size, seq_len, 3, device=device)

    if conditioned:
        vocab = CharVocab()
        char_ids = torch.randint(1, len(vocab), (batch_size, 20), device=device)
        char_mask = torch.ones(batch_size, 20, dtype=torch.bool, device=device)

        for _ in range(num_warmup):
            model(x, char_ids, char_mask)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(num_runs):
            model(x, char_ids, char_mask)

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / num_runs
    else:
        for _ in range(num_warmup):
            model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(num_runs):
            model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / num_runs

    latency_ms = elapsed * 1000
    throughput = batch_size * num_runs / (elapsed * num_runs)
    return latency_ms, throughput


@torch.no_grad()
def benchmark_autoregressive_sampling(
    model: torch.nn.Module,
    seq_len: int = 100,
    num_runs: int = 10,
    conditioned: bool = False,
    device: torch.device | None = None,
) -> float:
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    latencies = []
    for _ in range(num_runs):
        hidden = model.init_hidden(1, device=device)
        x = torch.zeros(1, 1, 3, device=device)

        if conditioned:
            CharVocab()
            char_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long, device=device)
            char_mask = torch.ones(1, 5, dtype=torch.bool, device=device)

            start = time.perf_counter()
            for _ in range(seq_len):
                _, hidden, _ = model(x, char_ids, char_mask, hidden, chunk_size=1)
                x = torch.randn(1, 1, 3, device=device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
        else:
            start = time.perf_counter()
            for _ in range(seq_len):
                _, hidden = model(x, hidden)
                x = torch.randn(1, 1, 3, device=device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000

        latencies.append(elapsed / seq_len)

    return float(np.mean(latencies))


def get_peak_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    return 0.0


def run_benchmark(
    hidden_dim: int = 256,
    num_layers: int = 3,
    num_mixtures: int = 20,
    num_windows: int = 10,
    char_embed_dim: int = 32,
    batch_size: int = 8,
    seq_len: int = 200,
    sample_len: int = 100,
    device: torch.device | None = None,
) -> list[BenchmarkResult]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []
    vocab = CharVocab()

    for model_name, conditioned in [("MDNRNN (unconditional)", False), ("MDNRNNConditioned", True)]:
        print(f"\nBenchmarking {model_name}...")

        if conditioned:
            model = MDNRNNConditioned(
                input_dim=3,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_mixtures=num_mixtures,
                num_windows=num_windows,
                char_vocab_size=len(vocab),
                char_embed_dim=char_embed_dim,
                dropout=0.0,
            ).to(device)
        else:
            model = MDNRNN(
                input_dim=3,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_mixtures=num_mixtures,
                dropout=0.0,
            ).to(device)

        total_params, trainable_params = count_parameters(model)
        size = model_size_mb(model)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        fwd_latency, fwd_throughput = benchmark_forward_pass(
            model,
            batch_size=batch_size,
            seq_len=seq_len,
            conditioned=conditioned,
            device=device,
        )

        sampling_latency = benchmark_autoregressive_sampling(
            model,
            seq_len=sample_len,
            conditioned=conditioned,
            device=device,
        )

        peak_mem = get_peak_memory_mb(device)

        result = BenchmarkResult(
            model_name=model_name,
            num_parameters=total_params,
            trainable_parameters=trainable_params,
            model_size_mb=round(size, 2),
            device=str(device),
            batch_size=batch_size,
            seq_len=seq_len,
            forward_latency_ms=round(fwd_latency, 2),
            forward_throughput_seqs_per_sec=round(fwd_throughput, 2),
            sampling_latency_ms_per_step=round(sampling_latency, 3),
            peak_memory_mb=round(peak_mem, 2),
            num_mixtures=num_mixtures,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
        )
        results.append(result)

        print(f"  Parameters: {total_params:,} ({size:.2f} MB)")
        print(f"  Forward pass: {fwd_latency:.2f} ms (batch={batch_size}, seq={seq_len})")
        print(f"  Throughput: {fwd_throughput:.2f} seq/s")
        print(f"  Sampling: {sampling_latency:.3f} ms/step")
        if peak_mem > 0:
            print(f"  Peak memory: {peak_mem:.2f} MB")

    return results


def benchmark_discriminator(
    hidden_dim: int = 128,
    num_layers: int = 4,
    batch_size: int = 16,
    seq_len: int = 200,
    device: torch.device | None = None,
) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    disc = SequenceDiscriminator(
        input_dim=3,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(device)

    total, _ = count_parameters(disc)
    size = model_size_mb(disc)

    x = torch.randn(batch_size, seq_len, 3, device=device)
    disc.eval()

    for _ in range(10):
        disc(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(50):
        disc(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / 50

    return {
        "model": "SequenceDiscriminator",
        "parameters": total,
        "size_mb": round(size, 2),
        "forward_latency_ms": round(elapsed * 1000, 2),
    }


def save_results(results: list[BenchmarkResult], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(r) for r in results]
    output_path.write_text(json.dumps(data, indent=2))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark MDN-RNN models")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--num_mixtures", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=200)
    parser.add_argument("--sample_len", type=int, default=100)
    parser.add_argument("--output", type=str, default="./benchmark_results.json")
    args = parser.parse_args()

    print("=" * 60)
    print("MDN-RNN Handwriting Generation Benchmark")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results = run_benchmark(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_mixtures=args.num_mixtures,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        sample_len=args.sample_len,
        device=device,
    )

    print("\n" + "=" * 60)
    print("Discriminator Benchmark")
    print("=" * 60)
    disc_result = benchmark_discriminator(device=device)
    for k, v in disc_result.items():
        print(f"  {k}: {v}")

    save_results(results, args.output)
