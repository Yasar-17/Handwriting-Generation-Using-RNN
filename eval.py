"""
Evaluation script for MDN-RNN handwriting generation.

Provides:
  1. Average NLL computation on a held-out validation set.
  2. Side-by-side comparison images of pre/post-adversarial model samples.
  3. generate_handwriting(text) function for demo use.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from data import (
    CharVocab,
    build_conditioned_dataloader,
    build_dataloader,
    denormalize_deltas,
    prepare_splits,
    render_strokes,
)
from losses import MDNLoss
from models import MDNRNN, MDNRNNConditioned
from sampling import sample_mixture_component

# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def load_checkpoint(
    ckpt_path: str | Path,
    device: torch.device,
    conditioned: bool | None = None,
) -> tuple[torch.nn.Module, dict]:
    """Load a model checkpoint and return (model, ckpt_dict).

    If `conditioned` is None it is inferred from the checkpoint metadata.
    """
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if conditioned is None:
        conditioned = ckpt.get("conditioned", False)

    vocab = CharVocab()

    if conditioned:
        model = MDNRNNConditioned(
            input_dim=3,
            hidden_dim=ckpt["model"].get("lstm.weight_hh_l0", None).shape[1]
            if "lstm.weight_hh_l0" in ckpt["model"]
            else 256,
            num_layers=_infer_num_layers(ckpt["model"]),
            num_mixtures=_infer_num_mixtures(ckpt["model"]),
            num_windows=10,
            char_vocab_size=len(vocab),
            char_embed_dim=32,
            dropout=0.2,
        )
    else:
        model = MDNRNN(
            input_dim=3,
            hidden_dim=_infer_hidden_dim(ckpt["model"]),
            num_layers=_infer_num_layers(ckpt["model"]),
            num_mixtures=_infer_num_mixtures(ckpt["model"]),
            dropout=0.2,
        )

    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, ckpt


def _infer_num_layers(state_dict: dict) -> int:
    count = 0
    for key in state_dict:
        if key.startswith("lstm.weight_hh_l"):
            layer_idx = int(key.split("lstm.weight_hh_l")[1][0])
            count = max(count, layer_idx + 1)
    return max(count, 3)


def _infer_hidden_dim(state_dict: dict) -> int:
    for key in state_dict:
        if "lstm.weight_hh_l0" in key:
            return state_dict[key].shape[0] // 4
    return 256


def _infer_num_mixtures(state_dict: dict) -> int:
    for key in state_dict:
        if "mdn_head.weight" in key:
            out_features = state_dict[key].shape[0]
            return out_features // 6
    return 20


# ---------------------------------------------------------------------------
# 1. Compute average NLL on validation set
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_val_nll(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    conditioned: bool = False,
) -> float:
    """Compute average MDN negative log-likelihood over the validation set."""
    model.eval()
    loss_fn = MDNLoss()
    total_loss = 0.0
    count = 0

    for batch in tqdm(loader, desc="Evaluating NLL", leave=False):
        data = batch["data"].to(device)
        mask = batch["mask"].to(device)

        if conditioned:
            char_ids = batch["char_ids"].to(device)
            char_mask = batch["char_mask"].to(device)
            params, _, _ = model(data, char_ids, char_mask)
        else:
            params, _ = model(data)

        nll = loss_fn(params, data, mask)
        batch_valid = mask.sum().item()
        total_loss += nll.item() * batch_valid
        count += batch_valid

    return total_loss / max(count, 1)


# ---------------------------------------------------------------------------
# 2. Side-by-side comparison images
# ---------------------------------------------------------------------------


@torch.no_grad()
def sample_conditioned(
    model: MDNRNNConditioned,
    text: str,
    vocab: CharVocab,
    temperature: float = 0.5,
    top_k: int = 0,
    top_p: float = 1.0,
    max_seq_len: int = 1000,
    device: torch.device | None = None,
) -> np.ndarray:
    """Generate handwriting for a given text string using windowed attention."""
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    char_ids = vocab.encode(text)
    char_tensor = torch.tensor([char_ids], dtype=torch.long, device=device)
    char_mask = torch.ones(1, len(char_ids), dtype=torch.bool, device=device)

    hidden = model.init_hidden(1, device=device)
    x = torch.zeros(1, 1, 3, device=device)

    deltas = []
    consecutive_pen_up = 0

    for _step in range(max_seq_len):
        params, hidden, _ = model(x, char_tensor, char_mask, hidden)

        component = sample_mixture_component(
            params["pi"][0, 0].cpu().numpy(),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        mx = params["mu_x"][0, 0, component].item()
        my = params["mu_y"][0, 0, component].item()
        sx = max(params["sigma_x"][0, 0, component].item() * temperature, 1e-6)
        sy = max(params["sigma_y"][0, 0, component].item() * temperature, 1e-6)
        r = params["rho"][0, 0, component].item()
        pen_prob = params["pen_up"][0, 0].item()

        z1 = np.random.randn()
        z2 = np.random.randn()
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

    deltas_arr = np.array(deltas, dtype=np.float32)
    if len(deltas_arr) > 0 and deltas_arr[:, 2].sum() == 0:
        deltas_arr[-1, 2] = 1.0

    return deltas_arr


@torch.no_grad()
def sample_unconditional(
    model: MDNRNN,
    seq_len: int = 500,
    temperature: float = 0.5,
    top_k: int = 0,
    top_p: float = 1.0,
    device: torch.device | None = None,
) -> np.ndarray:
    """Generate a stroke sequence autoregressively from the unconditional MDN-RNN."""
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    hidden = model.init_hidden(1, device=device)
    x = torch.zeros(1, 1, 3, device=device)

    deltas = []
    for _ in range(seq_len):
        params, hidden = model(x, hidden)

        component = sample_mixture_component(
            params["pi"][0, 0].cpu().numpy(),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        mx = params["mu_x"][0, 0, component].item()
        my = params["mu_y"][0, 0, component].item()
        sx = max(params["sigma_x"][0, 0, component].item() * temperature, 1e-6)
        sy = max(params["sigma_y"][0, 0, component].item() * temperature, 1e-6)
        r = params["rho"][0, 0, component].item()
        pen_prob = params["pen_up"][0, 0].item()

        z1 = np.random.randn()
        z2 = np.random.randn()
        dx = mx + sx * z1
        dy = my + sy * (r * z1 + np.sqrt(max(1 - r**2, 0)) * z2)

        pen_up = 1.0 if np.random.rand() < pen_prob else 0.0

        deltas.append([dx, dy, pen_up])
        x = torch.tensor([[[dx, dy, pen_up]]], device=device, dtype=torch.float32)

    deltas_arr = np.array(deltas, dtype=np.float32)
    if deltas_arr[:, 2].sum() == 0:
        deltas_arr[-1, 2] = 1.0

    return deltas_arr


def generate_comparison_images(
    pre_ckpt_path: str | Path,
    post_ckpt_path: str | Path,
    texts: list[str],
    stats_path: str | Path,
    output_dir: str | Path,
    temperature: float = 0.5,
    top_k: int = 0,
    top_p: float = 1.0,
    num_samples_per_text: int = 3,
    seed: int = 42,
) -> None:
    """Generate side-by-side comparison images for pre/post adversarial models.

    For each input text, generates `num_samples_per_text` samples from each model
    and saves a comparison figure.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = json.loads(Path(stats_path).read_text())
    mean_x, std_x, mean_y, std_y = stats["mean_x"], stats["std_x"], stats["mean_y"], stats["std_y"]

    vocab = CharVocab()

    pre_model, pre_ckpt = load_checkpoint(pre_ckpt_path, device)
    post_model, _post_ckpt = load_checkpoint(post_ckpt_path, device)

    conditioned = pre_ckpt.get("conditioned", False)

    for text in texts:
        for sample_idx in range(num_samples_per_text):
            if conditioned:
                pre_deltas = sample_conditioned(
                    pre_model, text, vocab, temperature, top_k=top_k, top_p=top_p, device=device
                )
                post_deltas = sample_conditioned(
                    post_model, text, vocab, temperature, top_k=top_k, top_p=top_p, device=device
                )
            else:
                pre_deltas = sample_unconditional(
                    pre_model, temperature=temperature, top_k=top_k, top_p=top_p, device=device
                )
                post_deltas = sample_unconditional(
                    post_model, temperature=temperature, top_k=top_k, top_p=top_p, device=device
                )

            pre_deltas_denorm = denormalize_deltas(pre_deltas, mean_x, std_x, mean_y, std_y)
            post_deltas_denorm = denormalize_deltas(post_deltas, mean_x, std_x, mean_y, std_y)

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            pre_fig = render_strokes(pre_deltas_denorm, title="Pre-adversarial")
            post_fig = render_strokes(post_deltas_denorm, title="Post-adversarial")

            for ax, src_fig in zip(axes, [pre_fig, post_fig], strict=False):
                for child in src_fig.get_children():
                    if isinstance(child, matplotlib.lines.Line2D):
                        ax.add_line(child.copy())
                ax.set_xlim(src_fig.axes[0].get_xlim())
                ax.set_ylim(src_fig.axes[0].get_ylim())
                ax.set_aspect("equal")
                ax.axis("off")

            axes[0].set_title("Pre-adversarial", fontsize=12)
            axes[1].set_title("Post-adversarial", fontsize=12)
            fig.suptitle(f"Text: '{text}' (sample {sample_idx + 1})", fontsize=14, y=1.02)
            fig.tight_layout()

            safe_text = text.replace(" ", "_").replace("/", "_")[:50]
            fig_path = output_dir / f"compare_{safe_text}_sample{sample_idx + 1}.png"
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {fig_path}")


# ---------------------------------------------------------------------------
# 3. generate_handwriting(text) demo function
# ---------------------------------------------------------------------------


class HandwritingGenerator:
    """Demo wrapper that loads a trained model and generates handwriting images."""

    def __init__(
        self,
        ckpt_path: str | Path,
        stats_path: str | Path,
        device: torch.device | None = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.ckpt = load_checkpoint(ckpt_path, self.device)
        self.conditioned = self.ckpt.get("conditioned", False)

        stats = json.loads(Path(stats_path).read_text())
        self.mean_x = stats["mean_x"]
        self.std_x = stats["std_x"]
        self.mean_y = stats["mean_y"]
        self.std_y = stats["std_y"]

        self.vocab = CharVocab()

    def generate_handwriting(
        self,
        text: str,
        temperature: float = 0.5,
        top_k: int = 0,
        top_p: float = 1.0,
        max_seq_len: int = 1000,
    ) -> plt.Figure:
        """Generate a rendered handwriting image for arbitrary input text.

        Args:
            text: The text to generate handwriting for.
            temperature: Sampling temperature (lower = more deterministic).
            top_k: If > 0, restrict sampling to the k most probable components.
            top_p: If in (0, 1), nucleus-filter to the top cumulative mass p.
            max_seq_len: Maximum number of timesteps to generate.

        Returns:
            A matplotlib Figure containing the rendered handwriting.
        """
        if self.conditioned:
            deltas = sample_conditioned(
                self.model,
                text,
                self.vocab,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_seq_len=max_seq_len,
                device=self.device,
            )
        else:
            seq_len = max(len(text) * 30, 200)
            deltas = sample_unconditional(
                self.model,
                seq_len=min(seq_len, max_seq_len),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                device=self.device,
            )

        deltas_denorm = denormalize_deltas(deltas, self.mean_x, self.std_x, self.mean_y, self.std_y)
        return render_strokes(deltas_denorm, title=f"'{text}' (T={temperature})")


# Convenience function for direct import
def generate_handwriting(
    text: str,
    ckpt_path: str | Path,
    stats_path: str | Path,
    temperature: float = 0.5,
    device: torch.device | None = None,
) -> plt.Figure:
    """One-call function to generate handwriting for demo purposes.

    Args:
        text: The text to generate handwriting for.
        ckpt_path: Path to the model checkpoint.
        stats_path: Path to the normalization stats JSON.
        temperature: Sampling temperature.
        device: Torch device (auto-detected if None).

    Returns:
        A matplotlib Figure with the rendered handwriting.
    """
    gen = HandwritingGenerator(ckpt_path, stats_path, device=device)
    return gen.generate_handwriting(text, temperature=temperature)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MDN-RNN handwriting models")
    parser.add_argument("--data_dir", type=str, required=True, help="Root directory of IAM XML files")
    parser.add_argument("--stats_path", type=str, required=True, help="Path to stats.json from training")
    parser.add_argument("--pre_ckpt", type=str, default=None, help="Path to pre-adversarial checkpoint")
    parser.add_argument("--post_ckpt", type=str, default=None, help="Path to post-adversarial checkpoint")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to single checkpoint (for NLL eval or demo)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=0, help="Top-k sampling (0 = disabled)")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p nucleus sampling (1.0 = disabled)")
    parser.add_argument("--comparison_texts", type=str, nargs="+", default=["the quick brown fox", "hello world"])
    parser.add_argument("--num_samples", type=int, default=3, help="Samples per text for comparison")
    parser.add_argument("--output_dir", type=str, default="./eval_output")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = json.loads(Path(args.stats_path).read_text())
    mean_x, std_x, mean_y, std_y = stats["mean_x"], stats["std_x"], stats["mean_y"], stats["std_y"]

    _, val_xml, _ = prepare_splits(args.data_dir)
    vocab = CharVocab()

    # -----------------------------------------------------------------------
    # NLL evaluation
    # -----------------------------------------------------------------------
    ckpt_to_eval = args.ckpt or args.post_ckpt
    if ckpt_to_eval:
        model, ckpt = load_checkpoint(ckpt_to_eval, device)
        conditioned = ckpt.get("conditioned", False)

        if conditioned:
            loader = build_conditioned_dataloader(
                val_xml,
                vocab=vocab,
                batch_size=args.batch_size,
                shuffle=False,
                mean_x=mean_x,
                std_x=std_x,
                mean_y=mean_y,
                std_y=std_y,
            )
        else:
            loader = build_dataloader(
                val_xml,
                batch_size=args.batch_size,
                shuffle=False,
                mean_x=mean_x,
                std_x=std_x,
                mean_y=mean_y,
                std_y=std_y,
            )

        nll = compute_val_nll(model, loader, device, conditioned=conditioned)
        print(f"\nValidation NLL: {nll:.4f}")

        results = {"val_nll": nll, "checkpoint": ckpt_to_eval, "conditioned": conditioned}
        (output_dir / "eval_results.json").write_text(json.dumps(results, indent=2))
        print(f"Results saved to {output_dir / 'eval_results.json'}")

    # -----------------------------------------------------------------------
    # Comparison images
    # -----------------------------------------------------------------------
    if args.pre_ckpt and args.post_ckpt:
        print("\nGenerating comparison images...")
        generate_comparison_images(
            pre_ckpt_path=args.pre_ckpt,
            post_ckpt_path=args.post_ckpt,
            texts=args.comparison_texts,
            stats_path=args.stats_path,
            output_dir=output_dir / "comparisons",
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_samples_per_text=args.num_samples,
            seed=args.seed,
        )

    # -----------------------------------------------------------------------
    # Demo generation
    # -----------------------------------------------------------------------
    demo_ckpt = args.ckpt or args.post_ckpt
    if demo_ckpt:
        print("\nGenerating demo samples...")
        gen = HandwritingGenerator(demo_ckpt, args.stats_path, device=device)
        demo_dir = output_dir / "demo"
        demo_dir.mkdir(exist_ok=True)

        for text in args.comparison_texts:
            fig = gen.generate_handwriting(text, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
            safe_text = text.replace(" ", "_").replace("/", "_")[:50]
            fig_path = demo_dir / f"demo_{safe_text}.png"
            fig.savefig(fig_path, dpi=150)
            plt.close(fig)
            print(f"  Saved {fig_path}")


if __name__ == "__main__":
    main()
