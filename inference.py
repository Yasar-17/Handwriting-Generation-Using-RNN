"""
Production inference CLI for handwriting generation.

Provides a clean command-line interface for generating handwriting from
trained models, with support for batch generation, different temperatures,
and multiple output formats.

Usage:
    python inference.py \
        --ckpt ./output_conditioned/checkpoints/checkpoint_best.pt \
        --stats ./output_conditioned/stats.json \
        --text "hello world" \
        --output_dir ./generated \
        --temperature 0.5 \
        --num_samples 3
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from data import CharVocab, denormalize_deltas, render_strokes
from models import MDNRNN, MDNRNNConditioned

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_model(ckpt_path: str | Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_state = ckpt["model"]
    conditioned = ckpt.get("conditioned", False)

    hidden_dim = 256
    num_layers = 3
    num_mixtures = 20

    for key in model_state:
        if "lstm.weight_hh_l0" in key:
            hidden_dim = model_state[key].shape[0] // 4
        if key.startswith("lstm.weight_hh_l"):
            layer_idx = int(key.split("lstm.weight_hh_l")[1][0])
            num_layers = max(num_layers, layer_idx + 1)
        if "mdn_head.weight" in key:
            num_mixtures = model_state[key].shape[0] // 6

    if conditioned:
        vocab = CharVocab()
        model = MDNRNNConditioned(
            input_dim=3,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_mixtures=num_mixtures,
            num_windows=10,
            char_vocab_size=len(vocab),
            char_embed_dim=32,
            dropout=0.0,
        )
    else:
        model = MDNRNN(
            input_dim=3,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_mixtures=num_mixtures,
            dropout=0.0,
        )

    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return model, ckpt


@torch.no_grad()
def sample_from_model(
    model: torch.nn.Module,
    ckpt: dict,
    text: str,
    stats: dict,
    temperature: float = 0.5,
    max_seq_len: int = 1000,
    device: torch.device | None = None,
) -> np.ndarray:
    if device is None:
        device = next(model.parameters()).device

    conditioned = ckpt.get("conditioned", False)
    mean_x = stats["mean_x"]
    std_x = stats["std_x"]
    mean_y = stats["mean_y"]
    std_y = stats["std_y"]

    if conditioned:
        vocab = CharVocab()
        char_ids = vocab.encode(text)
        char_tensor = torch.tensor([char_ids], dtype=torch.long, device=device)
        char_mask = torch.ones(1, len(char_ids), dtype=torch.bool, device=device)

        hidden = model.init_hidden(1, device=device)
        x = torch.zeros(1, 1, 3, device=device)
        deltas = []
        consecutive_pen_up = 0

        for _ in range(max_seq_len):
            params, hidden, _ = model(x, char_tensor, char_mask, hidden, chunk_size=1)

            pi = params["pi"][0, 0] / temperature
            pi = torch.softmax(pi, dim=0)
            component = torch.multinomial(pi, 1).item()

            mx = params["mu_x"][0, 0, component].item()
            my = params["mu_y"][0, 0, component].item()
            sx = max(params["sigma_x"][0, 0, component].item() * temperature, 1e-6)
            sy = max(params["sigma_y"][0, 0, component].item() * temperature, 1e-6)
            r = params["rho"][0, 0, component].item()
            pen_prob = params["pen_up"][0, 0].item()

            z1, z2 = np.random.randn(), np.random.randn()
            dx = mx + sx * z1
            dy = my + sy * (r * z1 + np.sqrt(max(1 - r**2, 0)) * z2)
            pen_up = 1.0 if np.random.rand() < pen_prob else 0.0

            deltas.append([dx, dy, pen_up])
            if pen_up == 1.0:
                consecutive_pen_up += 1
            else:
                consecutive_pen_up = 0
            if consecutive_pen_up >= 25:
                break
            x = torch.tensor([[[dx, dy, pen_up]]], device=device, dtype=torch.float32)
    else:
        hidden = model.init_hidden(1, device=device)
        x = torch.zeros(1, 1, 3, device=device)
        deltas = []

        for _ in range(max_seq_len):
            params, hidden = model(x, hidden)

            pi = params["pi"][0, 0] / temperature
            pi = torch.softmax(pi, dim=0)
            component = torch.multinomial(pi, 1).item()

            mx = params["mu_x"][0, 0, component].item()
            my = params["mu_y"][0, 0, component].item()
            sx = max(params["sigma_x"][0, 0, component].item() * temperature, 1e-6)
            sy = max(params["sigma_y"][0, 0, component].item() * temperature, 1e-6)
            r = params["rho"][0, 0, component].item()
            pen_prob = params["pen_up"][0, 0].item()

            z1, z2 = np.random.randn(), np.random.randn()
            dx = mx + sx * z1
            dy = my + sy * (r * z1 + np.sqrt(max(1 - r**2, 0)) * z2)
            pen_up = 1.0 if np.random.rand() < pen_prob else 0.0

            deltas.append([dx, dy, pen_up])
            x = torch.tensor([[[dx, dy, pen_up]]], device=device, dtype=torch.float32)

    deltas_arr = np.array(deltas, dtype=np.float32)
    if len(deltas_arr) > 0 and deltas_arr[:, 2].sum() == 0:
        deltas_arr[-1, 2] = 1.0

    return denormalize_deltas(deltas_arr, mean_x, std_x, mean_y, std_y)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate handwriting from trained model")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--stats", type=str, required=True, help="Path to stats.json")
    parser.add_argument("--text", type=str, nargs="+", required=True, help="Text(s) to generate")
    parser.add_argument("--output_dir", type=str, default="./generated", help="Output directory")
    parser.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature")
    parser.add_argument("--num_samples", type=int, default=1, help="Samples per text")
    parser.add_argument("--max_seq_len", type=int, default=1000, help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    setup_logging(args.verbose)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model from %s", args.ckpt)
    model, ckpt = load_model(args.ckpt, device)
    stats = json.loads(Path(args.stats).read_text())
    conditioned = ckpt.get("conditioned", False)

    logger.info("Model loaded | Conditioned: %s | Device: %s", conditioned, device)

    for text in args.text:
        for i in range(args.num_samples):
            logger.info("Generating '%s' (sample %d/%d, T=%.2f)...", text, i + 1, args.num_samples, args.temperature)
            deltas = sample_from_model(
                model, ckpt, text, stats,
                temperature=args.temperature,
                max_seq_len=args.max_seq_len,
                device=device,
            )

            title = f"'{text}' (T={args.temperature}, sample {i + 1})"
            fig = render_strokes(deltas, title=title)
            safe_text = text.replace(" ", "_").replace("/", "_")[:50]
            fig_path = output_dir / f"{safe_text}_sample{i + 1}.png"
            fig.savefig(fig_path, dpi=150)
            plt.close(fig)
            logger.info("Saved %s", fig_path)

    logger.info("Done. Generated %d images in %s", len(args.text) * args.num_samples, output_dir)


if __name__ == "__main__":
    main()
