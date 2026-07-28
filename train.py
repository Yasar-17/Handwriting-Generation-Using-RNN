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
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from data import (
    CharVocab,
    IAMStrokeDataset,
    IAMConditionedDataset,
    build_dataloader,
    build_conditioned_dataloader,
    collect_xml_files,
    compute_dataset_stats,
    denormalize_deltas,
    prepare_splits,
    render_strokes,
)
from losses import MDNLoss, mdn_mixture_mean, adversarial_loss
from models import MDNRNN, MDNRNNConditioned, SequenceDiscriminator


# ---------------------------------------------------------------------------
# Autoregressive sampling (unconditional)
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_unconditional(
    model: MDNRNN,
    seq_len: int = 500,
    temperature: float = 0.5,
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

        pi = params["pi"][0, 0]
        mu_x = params["mu_x"][0, 0]
        mu_y = params["mu_y"][0, 0]
        sigma_x = params["sigma_x"][0, 0]
        sigma_y = params["sigma_y"][0, 0]
        rho = params["rho"][0, 0]
        pen_prob = params["pen_up"][0, 0].item()

        pi = pi / temperature
        pi = torch.softmax(pi, dim=0)
        component = torch.multinomial(pi, 1).item()

        mx = mu_x[component].item()
        my = mu_y[component].item()
        sx = max(sigma_x[component].item() * temperature, 1e-6)
        sy = max(sigma_y[component].item() * temperature, 1e-6)
        r = rho[component].item()

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

    for step in range(max_seq_len):
        params, hidden, _ = model(x, char_tensor, char_mask, hidden, chunk_size=1)

        pi = params["pi"][0, 0]
        mu_x = params["mu_x"][0, 0]
        mu_y = params["mu_y"][0, 0]
        sigma_x = params["sigma_x"][0, 0]
        sigma_y = params["sigma_y"][0, 0]
        rho = params["rho"][0, 0]
        pen_prob = params["pen_up"][0, 0].item()

        pi = pi / temperature
        pi = torch.softmax(pi, dim=0)
        component = torch.multinomial(pi, 1).item()

        mx = mu_x[component].item()
        my = mu_y[component].item()
        sx = max(sigma_x[component].item() * temperature, 1e-6)
        sy = max(sigma_y[component].item() * temperature, 1e-6)
        r = rho[component].item()

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
) -> dict[str, float]:
    model.train()
    if discriminator is not None:
        discriminator.train()
    epoch_mdn_loss = 0.0
    epoch_adv_loss = 0.0
    epoch_disc_loss = 0.0
    count = 0

    for batch in tqdm(loader, desc="  Train", leave=False):
        data = batch["data"].to(device)
        mask = batch["mask"].to(device)

        optimizer.zero_grad()
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

            disc_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), clip_grad)
            disc_optimizer.step()
        else:
            gen_adv = torch.tensor(0.0)
            disc_loss = torch.tensor(0.0)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        batch_valid = mask.sum().item()
        epoch_mdn_loss += mdn.item() * batch_valid
        epoch_adv_loss += gen_adv.item() * batch_valid
        epoch_disc_loss += disc_loss.item() * batch_valid
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
) -> dict[str, float]:
    model.train()
    if discriminator is not None:
        discriminator.train()
    epoch_mdn_loss = 0.0
    epoch_adv_loss = 0.0
    epoch_disc_loss = 0.0
    count = 0

    for batch in tqdm(loader, desc="  Train", leave=False):
        data = batch["data"].to(device)
        mask = batch["mask"].to(device)
        char_ids = batch["char_ids"].to(device)
        char_mask = batch["char_mask"].to(device)

        optimizer.zero_grad()
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

            disc_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), clip_grad)
            disc_optimizer.step()
        else:
            gen_adv = torch.tensor(0.0)
            disc_loss = torch.tensor(0.0)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        batch_valid = mask.sum().item()
        epoch_mdn_loss += mdn.item() * batch_valid
        epoch_adv_loss += gen_adv.item() * batch_valid
        epoch_disc_loss += disc_loss.item() * batch_valid
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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train MDN-RNN for handwriting generation")
    parser.add_argument("--data_dir", type=str, required=True, help="Root directory of IAM XML files")
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--use_gan", action="store_true", help="Enable adversarial training with sequence discriminator")
    parser.add_argument("--disc_hidden_dim", type=int, default=128, help="Discriminator hidden dimension")
    parser.add_argument("--disc_num_layers", type=int, default=4, help="Number of Conv1D layers in discriminator")
    parser.add_argument("--disc_dropout", type=float, default=0.2, help="Discriminator dropout rate")
    parser.add_argument("--adv_weight", type=float, default=0.1, help="Weight for adversarial loss combined with MDN NLL")
    parser.add_argument("--disc_lr", type=float, default=1e-4, help="Discriminator learning rate")
    parser.add_argument(
        "--chunk_size", type=int, default=1,
        help="Conditioned training speedup: number of timesteps per chunked "
             "LSTM call. 1 = exact per-step recurrence (Graves 2013). "
             "Larger values (e.g. 16) trade a small amount of attention "
             "granularity for a large reduction in LSTM launch overhead. "
             "Sampling always uses chunk_size=1 for fidelity.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = "conditioned" if args.conditioned else "unconditional"
    print(f"Device: {device} | Mode: {mode}")

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    print("Collecting XML files...")
    train_xml, val_xml, test_xml = prepare_splits(args.data_dir)
    print(f"  Train: {len(train_xml)}, Val: {len(val_xml)}, Test: {len(test_xml)}")

    if not train_xml:
        print("No training data found. Exiting.")
        sys.exit(1)

    mean_x, std_x, mean_y, std_y = compute_dataset_stats(train_xml)
    print(f"  Stats: mean_x={mean_x:.4f}, std_x={std_x:.4f}, mean_y={mean_y:.4f}, std_y={std_y:.4f}")

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
        )
        val_loader = build_conditioned_dataloader(
            val_xml, vocab=vocab, batch_size=args.batch_size, shuffle=False,
            mean_x=mean_x, std_x=std_x, mean_y=mean_y, std_y=std_y,
        )
    else:
        train_loader = build_dataloader(
            train_xml, batch_size=args.batch_size, shuffle=True,
            mean_x=mean_x, std_x=std_x, mean_y=mean_y, std_y=std_y,
        )
        val_loader = build_dataloader(
            val_xml, batch_size=args.batch_size, shuffle=False,
            mean_x=mean_x, std_x=std_x, mean_y=mean_y, std_y=std_y,
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
        print(f"GAN training enabled | adv_weight={args.adv_weight} | disc_lr={args.disc_lr}")

    start_epoch = 0
    log = []

    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if args.use_gan and "discriminator" in ckpt:
            discriminator.load_state_dict(ckpt["discriminator"])
            disc_optimizer.load_state_dict(ckpt["disc_optimizer"])
        start_epoch = ckpt["epoch"] + 1
        log = ckpt.get("log", [])

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    print(f"\nTraining for {args.epochs} epochs (resuming from epoch {start_epoch})...")
    best_val = float("inf")

    for epoch in range(start_epoch, args.epochs):
        if args.conditioned:
            train_metrics = train_one_epoch_cond(
                model, train_loader, loss_fn, discriminator,
                optimizer, disc_optimizer, device, args.clip_grad, args.adv_weight,
            )
            val_metrics = evaluate_cond(model, val_loader, loss_fn, discriminator, device)
        else:
            train_metrics = train_one_epoch_uncond(
                model, train_loader, loss_fn, discriminator,
                optimizer, disc_optimizer, device, args.clip_grad, args.adv_weight,
            )
            val_metrics = evaluate_uncond(model, val_loader, loss_fn, discriminator, device)

        scheduler.step(val_metrics["mdn_loss"])

        lr = optimizer.param_groups[0]["lr"]
        train_total = train_metrics["mdn_loss"] + args.adv_weight * train_metrics["adv_loss"]
        val_total = val_metrics["mdn_loss"] + args.adv_weight * val_metrics["adv_loss"]

        if args.use_gan:
            print(
                f"Epoch {epoch:3d} | "
                f"train_mdn={train_metrics['mdn_loss']:.4f} train_adv={train_metrics['adv_loss']:.4f} "
                f"train_disc={train_metrics['disc_loss']:.4f} train_total={train_total:.4f} | "
                f"val_mdn={val_metrics['mdn_loss']:.4f} val_adv={val_metrics['adv_loss']:.4f} "
                f"val_disc={val_metrics['disc_loss']:.4f} val_total={val_total:.4f} | "
                f"lr={lr:.6f}"
            )
        else:
            print(
                f"Epoch {epoch:3d} | "
                f"train_loss={train_metrics['mdn_loss']:.4f} | "
                f"val_loss={val_metrics['mdn_loss']:.4f} | "
                f"lr={lr:.6f}"
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

        # Generate sample
        if (epoch + 1) % args.sample_every == 0 or epoch == start_epoch:
            print(f"  Generating sample at epoch {epoch}...")

            if args.conditioned:
                text = args.condition_text
                deltas = sample_conditioned(
                    model, text, vocab,
                    temperature=args.temperature,
                    max_seq_len=args.sample_len,
                    device=device,
                )
                title = f"Epoch {epoch} | '{text}' (T={args.temperature})"
            else:
                deltas = sample_unconditional(
                    model,
                    seq_len=args.sample_len,
                    temperature=args.temperature,
                    device=device,
                )
                title = f"Epoch {epoch} (T={args.temperature})"

            deltas_denorm = denormalize_deltas(deltas, mean_x, std_x, mean_y, std_y)
            fig = render_strokes(deltas_denorm, title=title)
            fig_path = samples_dir / f"epoch_{epoch:04d}.png"
            fig.savefig(fig_path, dpi=150)
            plt.close(fig)
            print(f"  Saved {fig_path}")

        # Checkpoint
        is_best = val_total < best_val
        if is_best:
            best_val = val_total

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val": best_val,
            "log": log,
            "conditioned": args.conditioned,
        }
        if args.use_gan:
            ckpt["discriminator"] = discriminator.state_dict()
            ckpt["disc_optimizer"] = disc_optimizer.state_dict()
            ckpt["adv_weight"] = args.adv_weight
        torch.save(ckpt, ckpt_dir / f"checkpoint_epoch_{epoch:04d}.pt")
        if is_best:
            torch.save(ckpt, ckpt_dir / "checkpoint_best.pt")

    print(f"\nTraining complete. Best val total loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
