"""
Training loop for MDN-RNN handwriting generation.

Supports both unconditional and text-conditioned training.
In conditioned mode, uses windowed attention over character embeddings
following Graves' "Generating Sequences With Recurrent Neural Networks".

Every few epochs, generates samples (unconditional or conditioned on text),
renders them, and saves checkpoints.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from data import (
    CharVocab,
    build_conditioned_dataloader,
    build_dataloader,
    compute_dataset_stats,
    denormalize_deltas,
    prepare_splits,
    render_strokes,
)
from ema import ModelEMA
from losses import MDNLoss, adversarial_loss, gradient_penalty, mdn_mixture_mean
from models import MDNRNN, MDNRNNConditioned, SequenceDiscriminator
from sampling import sample_mixture_component

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Autoregressive sampling (unconditional)
# ---------------------------------------------------------------------------

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
        dy = my + sy * (r * z1 + np.sqrt(max(1 - r ** 2, 0)) * z2)

        pen_up = 1.0 if np.random.rand() < pen_prob else 0.0

        deltas.append([dx, dy, pen_up])
        x = torch.tensor([[[dx, dy, pen_up]]], device=device, dtype=torch.float32)

    deltas_arr = np.array(deltas, dtype=np.float32)
    if deltas_arr[:, 2].sum() == 0:
        deltas_arr[-1, 2] = 1.0

    return deltas_arr


# ---------------------------------------------------------------------------
# Autoregressive sampling (conditioned)
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
    """Generate handwriting for a given text string using windowed attention.

    Stops early if pen_up probability stays high for several consecutive steps.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    char_ids = vocab.encode(text)
    char_tensor = torch.tensor([char_ids], dtype=torch.long, device=device)  # (1, C)
    char_mask = torch.ones(1, len(char_ids), dtype=torch.bool, device=device)

    hidden = model.init_hidden(1, device=device)
    x = torch.zeros(1, 1, 3, device=device)

    deltas = []
    consecutive_pen_up = 0

    for _step in range(max_seq_len):
        params, hidden, _ = model(x, char_tensor, char_mask, hidden, chunk_size=1)

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
        dy = my + sy * (r * z1 + np.sqrt(max(1 - r ** 2, 0)) * z2)

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


# ---------------------------------------------------------------------------
# Training / evaluation (unconditional)
# ---------------------------------------------------------------------------

def train_one_epoch_uncond(
    model: MDNRNN,
    loader: DataLoader,
    loss_fn: MDNLoss,
    discriminator: SequenceDiscriminator | None,
    optimizer: optim.Optimizer,
    disc_optimizer: optim.Optimizer | None,
    device: torch.device,
    clip_grad: float = 5.0,
    adv_weight: float = 0.1,
    grad_penalty_weight: float = 0.0,
    grad_accum_steps: int = 1,
    use_amp: bool = False,
    scaler: GradScaler | None = None,
    disc_scaler: GradScaler | None = None,
    ema: ModelEMA | None = None,
) -> dict[str, float]:
    model.train()
    if discriminator is not None:
        discriminator.train()
    epoch_mdn_loss = 0.0
    epoch_adv_loss = 0.0
    epoch_disc_loss = 0.0
    count = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc="  Train", leave=False)):
        data = batch["data"].to(device)
        mask = batch["mask"].to(device)

        should_accumulate = (batch_idx + 1) % grad_accum_steps != 0

        with autocast(enabled=use_amp):
            params, _ = model(data)
            mdn = loss_fn(params, data, mask)
            loss = mdn

            if discriminator is not None:
                fake_seq = mdn_mixture_mean(
                    params["mu_x"], params["mu_y"], params["pi"], params["pen_up"]
                )
                disc_fake = discriminator(fake_seq)
                disc_real = discriminator(data).detach()

                _, gen_adv = adversarial_loss(disc_real, disc_fake, mask[:, 0])
                loss = loss + adv_weight * gen_adv

                disc_optimizer.zero_grad()
                disc_real = discriminator(data)
                disc_fake = discriminator(fake_seq.detach())
                disc_loss, _ = adversarial_loss(disc_real, disc_fake, mask[:, 0])
            else:
                gen_adv = torch.tensor(0.0)
                disc_loss = torch.tensor(0.0)

        if discriminator is not None and grad_penalty_weight > 0:
            disc_loss = disc_loss + gradient_penalty(
                discriminator,
                data.detach(),
                fake_seq.detach(),
                lambda_=grad_penalty_weight,
            )

        loss = loss / grad_accum_steps
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if not should_accumulate:
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

            if ema is not None:
                ema.update(model)

        if discriminator is not None:
            disc_loss = disc_loss / grad_accum_steps
            if use_amp:
                disc_scaler.scale(disc_loss).backward()
            else:
                disc_loss.backward()

            if not should_accumulate:
                if use_amp:
                    disc_scaler.unscale_(disc_optimizer)
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), clip_grad)
                if use_amp:
                    disc_scaler.step(disc_optimizer)
                    disc_scaler.update()
                else:
                    disc_optimizer.step()
                disc_optimizer.zero_grad()

        batch_valid = mask.sum().item()
        epoch_mdn_loss += mdn.item() * batch_valid
        epoch_adv_loss += gen_adv.item() * batch_valid
        epoch_disc_loss += disc_loss.item() * grad_accum_steps * batch_valid
        count += batch_valid

    return {
        "mdn_loss": epoch_mdn_loss / max(count, 1),
        "adv_loss": epoch_adv_loss / max(count, 1),
        "disc_loss": epoch_disc_loss / max(count, 1),
    }


@torch.no_grad()
def evaluate_uncond(
    model: MDNRNN,
    loader: DataLoader,
    loss_fn: MDNLoss,
    discriminator: SequenceDiscriminator | None,
    device: torch.device,
    use_amp: bool = False,
) -> dict[str, float]:
    model.eval()
    if discriminator is not None:
        discriminator.eval()
    val_mdn_loss = 0.0
    val_adv_loss = 0.0
    val_disc_loss = 0.0
    count = 0

    for batch in tqdm(loader, desc="  Val", leave=False):
        data = batch["data"].to(device)
        mask = batch["mask"].to(device)

        with autocast(enabled=use_amp):
            params, _ = model(data)
            mdn = loss_fn(params, data, mask)

            if discriminator is not None:
                fake_seq = mdn_mixture_mean(
                    params["mu_x"], params["mu_y"], params["pi"], params["pen_up"]
                )
                disc_fake = discriminator(fake_seq)
                disc_real = discriminator(data)

                _, gen_adv = adversarial_loss(disc_real, disc_fake, mask[:, 0])
                disc_loss, _ = adversarial_loss(disc_real, disc_fake, mask[:, 0])
            else:
                gen_adv = torch.tensor(0.0)
                disc_loss = torch.tensor(0.0)

        batch_valid = mask.sum().item()
        val_mdn_loss += mdn.item() * batch_valid
        val_adv_loss += gen_adv.item() * batch_valid
        val_disc_loss += disc_loss.item() * batch_valid
        count += batch_valid

    return {
        "mdn_loss": val_mdn_loss / max(count, 1),
        "adv_loss": val_adv_loss / max(count, 1),
        "disc_loss": val_disc_loss / max(count, 1),
    }


# ---------------------------------------------------------------------------
# Training / evaluation (conditioned)
# ---------------------------------------------------------------------------

def train_one_epoch_cond(
    model: MDNRNNConditioned,
    loader: DataLoader,
    loss_fn: MDNLoss,
    discriminator: SequenceDiscriminator | None,
    optimizer: optim.Optimizer,
    disc_optimizer: optim.Optimizer | None,
    device: torch.device,
    clip_grad: float = 5.0,
    adv_weight: float = 0.1,
    grad_penalty_weight: float = 0.0,
    grad_accum_steps: int = 1,
    use_amp: bool = False,
    scaler: GradScaler | None = None,
    disc_scaler: GradScaler | None = None,
    ema: ModelEMA | None = None,
) -> dict[str, float]:
    model.train()
    if discriminator is not None:
        discriminator.train()
    epoch_mdn_loss = 0.0
    epoch_adv_loss = 0.0
    epoch_disc_loss = 0.0
    count = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc="  Train", leave=False)):
        data = batch["data"].to(device)
        mask = batch["mask"].to(device)
        char_ids = batch["char_ids"].to(device)
        char_mask = batch["char_mask"].to(device)

        should_accumulate = (batch_idx + 1) % grad_accum_steps != 0

        with autocast(enabled=use_amp):
            params, _, _ = model(data, char_ids, char_mask)
            mdn = loss_fn(params, data, mask)
            loss = mdn

            if discriminator is not None:
                fake_seq = mdn_mixture_mean(
                    params["mu_x"], params["mu_y"], params["pi"], params["pen_up"]
                )
                disc_fake = discriminator(fake_seq)
                disc_real = discriminator(data).detach()

                _, gen_adv = adversarial_loss(disc_real, disc_fake, mask[:, 0])
                loss = loss + adv_weight * gen_adv

                disc_optimizer.zero_grad()
                disc_real = discriminator(data)
                disc_fake = discriminator(fake_seq.detach())
                disc_loss, _ = adversarial_loss(disc_real, disc_fake, mask[:, 0])
            else:
                gen_adv = torch.tensor(0.0)
                disc_loss = torch.tensor(0.0)

        if discriminator is not None and grad_penalty_weight > 0:
            disc_loss = disc_loss + gradient_penalty(
                discriminator,
                data.detach(),
                fake_seq.detach(),
                lambda_=grad_penalty_weight,
            )

        loss = loss / grad_accum_steps
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if not should_accumulate:
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

            if ema is not None:
                ema.update(model)

        if discriminator is not None:
            disc_loss = disc_loss / grad_accum_steps
            if use_amp:
                disc_scaler.scale(disc_loss).backward()
            else:
                disc_loss.backward()

            if not should_accumulate:
                if use_amp:
                    disc_scaler.unscale_(disc_optimizer)
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), clip_grad)
                if use_amp:
                    disc_scaler.step(disc_optimizer)
                    disc_scaler.update()
                else:
                    disc_optimizer.step()
                disc_optimizer.zero_grad()

        batch_valid = mask.sum().item()
        epoch_mdn_loss += mdn.item() * batch_valid
        epoch_adv_loss += gen_adv.item() * batch_valid
        epoch_disc_loss += disc_loss.item() * grad_accum_steps * batch_valid
        count += batch_valid

    return {
        "mdn_loss": epoch_mdn_loss / max(count, 1),
        "adv_loss": epoch_adv_loss / max(count, 1),
        "disc_loss": epoch_disc_loss / max(count, 1),
    }


@torch.no_grad()
def evaluate_cond(
    model: MDNRNNConditioned,
    loader: DataLoader,
    loss_fn: MDNLoss,
    discriminator: SequenceDiscriminator | None,
    device: torch.device,
    use_amp: bool = False,
) -> dict[str, float]:
    model.eval()
    if discriminator is not None:
        discriminator.eval()
    val_mdn_loss = 0.0
    val_adv_loss = 0.0
    val_disc_loss = 0.0
    count = 0

    for batch in tqdm(loader, desc="  Val", leave=False):
        data = batch["data"].to(device)
        mask = batch["mask"].to(device)
        char_ids = batch["char_ids"].to(device)
        char_mask = batch["char_mask"].to(device)

        with autocast(enabled=use_amp):
            params, _, _ = model(data, char_ids, char_mask)
            mdn = loss_fn(params, data, mask)

            if discriminator is not None:
                fake_seq = mdn_mixture_mean(
                    params["mu_x"], params["mu_y"], params["pi"], params["pen_up"]
                )
                disc_fake = discriminator(fake_seq)
                disc_real = discriminator(data)

                _, gen_adv = adversarial_loss(disc_real, disc_fake, mask[:, 0])
                disc_loss, _ = adversarial_loss(disc_real, disc_fake, mask[:, 0])
            else:
                gen_adv = torch.tensor(0.0)
                disc_loss = torch.tensor(0.0)

        batch_valid = mask.sum().item()
        val_mdn_loss += mdn.item() * batch_valid
        val_adv_loss += gen_adv.item() * batch_valid
        val_disc_loss += disc_loss.item() * batch_valid
        count += batch_valid

    return {
        "mdn_loss": val_mdn_loss / max(count, 1),
        "adv_loss": val_adv_loss / max(count, 1),
        "disc_loss": val_disc_loss / max(count, 1),
    }


# ---------------------------------------------------------------------------
# Plot training loss
# ---------------------------------------------------------------------------

def plot_training_loss(log: list[dict], output_dir: Path) -> None:
    """Generate training loss plots from the log."""
    if not log:
        return

    epochs = [entry["epoch"] for entry in log]
    train_mdn = [entry["train_mdn_loss"] for entry in log]
    val_mdn = [entry["val_mdn_loss"] for entry in log]
    train_total = [entry["train_total"] for entry in log]
    val_total = [entry["val_total"] for entry in log]
    lr = [entry["lr"] for entry in log]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, train_mdn, label="Train MDN Loss", alpha=0.7)
    axes[0].plot(epochs, val_mdn, label="Val MDN Loss", alpha=0.7)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MDN Loss")
    axes[0].set_title("MDN Negative Log-Likelihood")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_total, label="Train Total", alpha=0.7)
    axes[1].plot(epochs, val_total, label="Val Total", alpha=0.7)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Total Loss")
    axes[1].set_title("Total Loss (MDN + Adversarial)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, lr, label="Learning Rate", color="red", alpha=0.7)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("Learning Rate Schedule")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "training_loss.png", dpi=150)
    plt.close(fig)

    if any("train_adv_loss" in entry for entry in log):
        train_adv = [entry.get("train_adv_loss", 0) for entry in log]
        val_adv = [entry.get("val_adv_loss", 0) for entry in log]
        train_disc = [entry.get("train_disc_loss", 0) for entry in log]
        val_disc = [entry.get("val_disc_loss", 0) for entry in log]

        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
        axes2[0].plot(epochs, train_adv, label="Train Adv", alpha=0.7)
        axes2[0].plot(epochs, val_adv, label="Val Adv", alpha=0.7)
        axes2[0].set_xlabel("Epoch")
        axes2[0].set_ylabel("Adversarial Loss")
        axes2[0].set_title("Generator Adversarial Loss")
        axes2[0].legend()
        axes2[0].grid(True, alpha=0.3)

        axes2[1].plot(epochs, train_disc, label="Train Disc", alpha=0.7)
        axes2[1].plot(epochs, val_disc, label="Val Disc", alpha=0.7)
        axes2[1].set_xlabel("Epoch")
        axes2[1].set_ylabel("Discriminator Loss")
        axes2[1].set_title("Discriminator Loss")
        axes2[1].legend()
        axes2[1].grid(True, alpha=0.3)

        fig2.tight_layout()
        fig2.savefig(output_dir / "training_gan_loss.png", dpi=150)
        plt.close(fig2)


def log_metrics_to_tensorboard(
    writer: SummaryWriter,
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    lr: float,
) -> None:
    """Log training and validation metrics to TensorBoard."""
    writer.add_scalar("Loss/Train_MDN", train_metrics["mdn_loss"], epoch)
    writer.add_scalar("Loss/Val_MDN", val_metrics["mdn_loss"], epoch)
    writer.add_scalar("Loss/Train_Total", train_metrics["mdn_loss"] + 0.1 * train_metrics.get("adv_loss", 0), epoch)
    writer.add_scalar("Loss/Val_Total", val_metrics["mdn_loss"] + 0.1 * val_metrics.get("adv_loss", 0), epoch)
    writer.add_scalar("Metrics/Learning_Rate", lr, epoch)

    if train_metrics.get("adv_loss", 0) > 0:
        writer.add_scalar("Loss/Train_Adversarial", train_metrics["adv_loss"], epoch)
        writer.add_scalar("Loss/Val_Adversarial", val_metrics["adv_loss"], epoch)
        writer.add_scalar("Loss/Train_Discriminator", train_metrics["disc_loss"], epoch)
        writer.add_scalar("Loss/Val_Discriminator", val_metrics["disc_loss"], epoch)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_yaml_config(config_path: str | Path) -> dict:
    """Load a YAML config file into a nested dictionary.

    Raises:
        SystemExit: if PyYAML is not installed or the file is not a mapping.
    """
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(
            "PyYAML is required for --config. Install it with: pip install pyyaml"
        ) from exc

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise SystemExit(f"Config file {path} must contain a top-level mapping.")
    return config


def flatten_config(config: dict) -> dict:
    """Flatten a nested config dict into dotted ``section.key`` -> value pairs."""
    flat: dict = {}
    for key, value in config.items():
        if isinstance(value, dict):
            for leaf, v in flatten_config(value).items():
                flat[f"{key}.{leaf}"] = v
        else:
            flat[key] = value
    return flat


def apply_config(parser: argparse.ArgumentParser, config: dict) -> None:
    """Overlay a YAML config onto the parser defaults.

    Only leaf keys that match an existing argparse destination are applied,
    so unknown or section-level keys are safely ignored. Command-line flags
    still win because they are parsed after the config defaults are set.
    """
    known = {action.dest for action in parser._actions if action.dest != argparse.SUPPRESS}
    defaults: dict = {}
    for key, value in flatten_config(config).items():
        leaf = key.rsplit(".", 1)[-1]
        if leaf in known and value is not None:
            defaults[leaf] = value
    if defaults:
        parser.set_defaults(**defaults)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MDN-RNN for handwriting generation")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file (optional)")
    parser.add_argument("--data_dir", type=str, default=None, help="Root directory of IAM XML files")
    parser.add_argument("--num_workers", type=int, default=0, help="Dataloader worker processes")
    parser.add_argument("--output_dir", type=str, default="./output", help="Checkpoint and sample output dir")
    parser.add_argument("--conditioned", action="store_true", help="Enable text-conditioned training")
    parser.add_argument("--condition_text", type=str, default="the quick brown fox", help="Text to generate during sampling")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--num_mixtures", type=int, default=20)
    parser.add_argument("--num_windows", type=int, default=10, help="Number of attention windows (conditioned only)")
    parser.add_argument("--char_embed_dim", type=int, default=32, help="Character embedding dimension")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip_grad", type=float, default=5.0)
    parser.add_argument("--sample_every", type=int, default=5, help="Generate sample every N epochs")
    parser.add_argument("--sample_len", type=int, default=800, help="Timesteps per generated sample")
    parser.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k sampling: keep only the k most probable mixture components (0 = disabled)")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling: keep components up to cumulative probability p (1.0 = disabled)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--use_gan", action="store_true", help="Enable adversarial training with sequence discriminator")
    parser.add_argument("--disc_hidden_dim", type=int, default=128, help="Discriminator hidden dimension")
    parser.add_argument("--disc_num_layers", type=int, default=4, help="Number of Conv1D layers in discriminator")
    parser.add_argument("--disc_dropout", type=float, default=0.2, help="Discriminator dropout rate")
    parser.add_argument("--adv_weight", type=float, default=0.1, help="Weight for adversarial loss combined with MDN NLL")
    parser.add_argument(
        "--grad_penalty_weight", type=float, default=0.0,
        help="WGAN-GP gradient penalty weight for the discriminator "
             "(0 = disabled; a value around 10.0 is typical). Penalizing the "
             "discriminator's gradient norm keeps it Lipschitz-smooth and "
             "stabilizes adversarial training.",
    )
    parser.add_argument("--disc_lr", type=float, default=1e-4, help="Discriminator learning rate")
    parser.add_argument(
        "--chunk_size", type=int, default=1,
        help="Conditioned training speedup: number of timesteps per chunked "
             "LSTM call. 1 = exact per-step recurrence (Graves 2013). "
             "Larger values (e.g. 16) trade a small amount of attention "
             "granularity for a large reduction in LSTM launch overhead. "
             "Sampling always uses chunk_size=1 for fidelity.",
    )
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Number of steps to accumulate gradients")
    parser.add_argument("--use_amp", action="store_true", help="Enable automatic mixed precision training")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Number of warmup epochs for LR scheduler")
    parser.add_argument("--use_cosine_annealing", action="store_true", help="Use cosine annealing LR scheduler")
    parser.add_argument("--early_stopping_patience", type=int, default=0, help="Early stopping patience (0 = disabled)")
    parser.add_argument(
        "--use_ema", action="store_true",
        help="Maintain an exponential moving average of the generator weights "
             "and use the smoothed weights for sampling (higher-quality, "
             "more stable samples; standard for GAN training).",
    )
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay factor (larger = slower adaptation)")

    # A first parse only discovers --config so a config file can set defaults
    # before the final parse. Command-line flags always take precedence.
    pre_args, _ = parser.parse_known_args()
    if pre_args.config:
        config = load_yaml_config(pre_args.config)
        apply_config(parser, config)
        logger.info("Loaded configuration from %s", pre_args.config)
    args = parser.parse_args()

    if args.data_dir is None:
        parser.error("--data_dir is required (or set data.data_dir in the config file)")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    tb_dir = output_dir / "tensorboard"
    tb_dir.mkdir(exist_ok=True)

    writer = SummaryWriter(log_dir=str(tb_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = "conditioned" if args.conditioned else "unconditional"
    logger.info("Device: %s | Mode: %s", device, mode)

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    logger.info("Collecting XML files...")
    train_xml, val_xml, test_xml = prepare_splits(args.data_dir)
    logger.info("Train: %d, Val: %d, Test: %d", len(train_xml), len(val_xml), len(test_xml))

    if not train_xml:
        logger.error("No training data found. Exiting.")
        sys.exit(1)

    mean_x, std_x, mean_y, std_y = compute_dataset_stats(train_xml)
    logger.info("Stats: mean_x=%.4f, std_x=%.4f, mean_y=%.4f, std_y=%.4f", mean_x, std_x, mean_y, std_y)

    stats_path = output_dir / "stats.json"
    stats_path.write_text(json.dumps({
        "mean_x": mean_x, "std_x": std_x,
        "mean_y": mean_y, "std_y": std_y,
    }, indent=2))

    vocab = CharVocab()

    if args.conditioned:
        train_loader = build_conditioned_dataloader(
            train_xml, vocab=vocab, batch_size=args.batch_size, shuffle=True,
            mean_x=mean_x, std_x=std_x, mean_y=mean_y, std_y=std_y,
            num_workers=args.num_workers,
        )
        val_loader = build_conditioned_dataloader(
            val_xml, vocab=vocab, batch_size=args.batch_size, shuffle=False,
            mean_x=mean_x, std_x=std_x, mean_y=mean_y, std_y=std_y,
            num_workers=args.num_workers,
        )
    else:
        train_loader = build_dataloader(
            train_xml, batch_size=args.batch_size, shuffle=True,
            mean_x=mean_x, std_x=std_x, mean_y=mean_y, std_y=std_y,
            num_workers=args.num_workers,
        )
        val_loader = build_dataloader(
            val_xml, batch_size=args.batch_size, shuffle=False,
            mean_x=mean_x, std_x=std_x, mean_y=mean_y, std_y=std_y,
            num_workers=args.num_workers,
        )

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    if args.conditioned:
        model = MDNRNNConditioned(
            input_dim=3,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_mixtures=args.num_mixtures,
            num_windows=args.num_windows,
            char_vocab_size=len(vocab),
            char_embed_dim=args.char_embed_dim,
            dropout=args.dropout,
            chunk_size=args.chunk_size,
        ).to(device)
    else:
        model = MDNRNN(
            input_dim=3,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_mixtures=args.num_mixtures,
            dropout=args.dropout,
        ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    cosine_scheduler = None
    if args.use_cosine_annealing:
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )
    loss_fn = MDNLoss()

    discriminator = None
    disc_optimizer = None
    if args.use_gan:
        discriminator = SequenceDiscriminator(
            input_dim=3,
            hidden_dim=args.disc_hidden_dim,
            num_layers=args.disc_num_layers,
            dropout=args.disc_dropout,
        ).to(device)
        disc_optimizer = optim.Adam(discriminator.parameters(), lr=args.disc_lr)
        logger.info("GAN training enabled | adv_weight=%.2f | disc_lr=%.2e", args.adv_weight, args.disc_lr)
        if args.grad_penalty_weight > 0:
            logger.info("  Gradient penalty (WGAN-GP) enabled | weight=%.2f", args.grad_penalty_weight)

    scaler = GradScaler(enabled=args.use_amp)
    disc_scaler = GradScaler(enabled=args.use_amp) if args.use_gan else None

    ema = None
    if args.use_ema:
        ema = ModelEMA(model, decay=args.ema_decay, device=device)
        logger.info("EMA of generator weights enabled | decay=%.4f", args.ema_decay)

    start_epoch = 0
    log = []

    if args.resume:
        logger.info("Resuming from %s", args.resume)
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if args.use_cosine_annealing and "cosine_scheduler" in ckpt:
            cosine_scheduler.load_state_dict(ckpt["cosine_scheduler"])
        if args.use_gan and "discriminator" in ckpt:
            discriminator.load_state_dict(ckpt["discriminator"])
            disc_optimizer.load_state_dict(ckpt["disc_optimizer"])
        if ema is not None and "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        start_epoch = ckpt["epoch"] + 1
        log = ckpt.get("log", [])

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    logger.info("\nTraining for %d epochs (resuming from epoch %d)...", args.epochs, start_epoch)
    if args.use_amp:
        logger.info("  Mixed precision (AMP) enabled")
    if args.grad_accum_steps > 1:
        logger.info("  Gradient accumulation: %d steps (effective batch size: %d)", args.grad_accum_steps, args.batch_size * args.grad_accum_steps)
    if args.warmup_epochs > 0:
        logger.info("  LR warmup: %d epochs", args.warmup_epochs)
    if args.use_cosine_annealing:
        logger.info("  Cosine annealing LR scheduler enabled")
    if args.early_stopping_patience > 0:
        logger.info("  Early stopping enabled (patience: %d)", args.early_stopping_patience)

    best_val = float("inf")
    patience_counter = 0
    best_val_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        if args.conditioned:
            train_metrics = train_one_epoch_cond(
                model, train_loader, loss_fn, discriminator,
                optimizer, disc_optimizer, device, args.clip_grad, args.adv_weight,
                grad_penalty_weight=args.grad_penalty_weight,
                grad_accum_steps=args.grad_accum_steps,
                use_amp=args.use_amp,
                scaler=scaler,
                disc_scaler=disc_scaler,
                ema=ema,
            )
            val_metrics = evaluate_cond(model, val_loader, loss_fn, discriminator, device, use_amp=args.use_amp)
        else:
            train_metrics = train_one_epoch_uncond(
                model, train_loader, loss_fn, discriminator,
                optimizer, disc_optimizer, device, args.clip_grad, args.adv_weight,
                grad_penalty_weight=args.grad_penalty_weight,
                grad_accum_steps=args.grad_accum_steps,
                use_amp=args.use_amp,
                scaler=scaler,
                disc_scaler=disc_scaler,
                ema=ema,
            )
            val_metrics = evaluate_uncond(model, val_loader, loss_fn, discriminator, device, use_amp=args.use_amp)

        if args.use_cosine_annealing:
            if epoch < args.warmup_epochs:
                warmup_factor = (epoch + 1) / args.warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group["lr"] = args.lr * warmup_factor
            else:
                cosine_scheduler.step()
        else:
            if epoch >= args.warmup_epochs:
                scheduler.step(val_metrics["mdn_loss"])
            elif epoch < args.warmup_epochs:
                warmup_factor = (epoch + 1) / args.warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group["lr"] = args.lr * warmup_factor

        lr = optimizer.param_groups[0]["lr"]
        train_total = train_metrics["mdn_loss"] + args.adv_weight * train_metrics["adv_loss"]
        val_total = val_metrics["mdn_loss"] + args.adv_weight * val_metrics["adv_loss"]

        log_metrics_to_tensorboard(writer, epoch, train_metrics, val_metrics, lr)

        if args.use_gan:
            logger.info(
                "Epoch %3d | "
                "train_mdn=%.4f train_adv=%.4f train_disc=%.4f train_total=%.4f | "
                "val_mdn=%.4f val_adv=%.4f val_disc=%.4f val_total=%.4f | "
                "lr=%.6f",
                epoch,
                train_metrics["mdn_loss"], train_metrics["adv_loss"], train_metrics["disc_loss"], train_total,
                val_metrics["mdn_loss"], val_metrics["adv_loss"], val_metrics["disc_loss"], val_total,
                lr,
            )
        else:
            logger.info(
                "Epoch %3d | train_loss=%.4f | val_loss=%.4f | lr=%.6f",
                epoch, train_metrics["mdn_loss"], val_metrics["mdn_loss"], lr,
            )

        log_entry = {
            "epoch": epoch,
            "train_mdn_loss": train_metrics["mdn_loss"],
            "val_mdn_loss": val_metrics["mdn_loss"],
            "train_total": train_total,
            "val_total": val_total,
            "lr": lr,
        }
        if args.use_gan:
            log_entry["train_adv_loss"] = train_metrics["adv_loss"]
            log_entry["val_adv_loss"] = val_metrics["adv_loss"]
            log_entry["train_disc_loss"] = train_metrics["disc_loss"]
            log_entry["val_disc_loss"] = val_metrics["disc_loss"]

        log.append(log_entry)
        (output_dir / "log.json").write_text(json.dumps(log, indent=2))

        # Generate sample (from EMA weights when enabled, for smoother samples)
        if (epoch + 1) % args.sample_every == 0 or epoch == start_epoch:
            logger.info("  Generating sample at epoch %d...", epoch)
            sample_model = ema.ema_model() if ema is not None else model

            if args.conditioned:
                text = args.condition_text
                deltas = sample_conditioned(
                    sample_model, text, vocab,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    max_seq_len=args.sample_len,
                    device=device,
                )
                title = f"Epoch {epoch} | '{text}' (T={args.temperature})"
            else:
                deltas = sample_unconditional(
                    sample_model,
                    seq_len=args.sample_len,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    device=device,
                )
                title = f"Epoch {epoch} (T={args.temperature})"

            deltas_denorm = denormalize_deltas(deltas, mean_x, std_x, mean_y, std_y)
            fig = render_strokes(deltas_denorm, title=title)
            fig_path = samples_dir / f"epoch_{epoch:04d}.png"
            fig.savefig(fig_path, dpi=150)
            plt.close(fig)
            writer.add_figure("Samples/Handwriting", fig, epoch)
            logger.info("  Saved %s", fig_path)

        # Checkpoint
        is_best = val_total < best_val
        if is_best:
            best_val = val_total
            patience_counter = 0
            best_val_epoch = epoch
        else:
            patience_counter += 1

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val": best_val,
            "log": log,
            "conditioned": args.conditioned,
        }
        if args.use_cosine_annealing:
            ckpt["cosine_scheduler"] = cosine_scheduler.state_dict()
        if args.use_gan:
            ckpt["discriminator"] = discriminator.state_dict()
            ckpt["disc_optimizer"] = disc_optimizer.state_dict()
            ckpt["adv_weight"] = args.adv_weight
            ckpt["grad_penalty_weight"] = args.grad_penalty_weight
        if ema is not None:
            ckpt["ema"] = ema.state_dict()
            ckpt["ema_decay"] = args.ema_decay
        torch.save(ckpt, ckpt_dir / f"checkpoint_epoch_{epoch:04d}.pt")
        if is_best:
            torch.save(ckpt, ckpt_dir / "checkpoint_best.pt")
            if ema is not None:
                # Drop-in EMA checkpoint: eval/inference can load it directly.
                ema_ckpt = dict(ckpt)
                ema_ckpt["model"] = ema.state_dict()
                torch.save(ema_ckpt, ckpt_dir / "checkpoint_best_ema.pt")

        # Generate training loss plots
        plot_training_loss(log, output_dir)

        # Early stopping check
        if args.early_stopping_patience > 0 and patience_counter >= args.early_stopping_patience:
            logger.info("\nEarly stopping triggered at epoch %d (best val at epoch %d)", epoch, best_val_epoch)
            break

    writer.close()
    logger.info("\nTraining complete. Best val total loss: %.4f at epoch %d", best_val, best_val_epoch)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the training script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


if __name__ == "__main__":
    setup_logging()
    main()
