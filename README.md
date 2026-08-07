# Handwriting Generation Using RNN + GAN

![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.0-ee4c2c.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)

This project writes text in **handwriting**. It uses a special neural network called
a **Mixture-Density RNN** (MDN-RNN) to predict the pen's movement, one small step at a
time. It can also use a **GAN** to make the strokes look more real.

The method follows the famous paper *"Generating Sequences With Recurrent Neural
Networks"* by Alex Graves (2013).

> Handwriting is stored as a **sequence of points** (x, y moves + pen up/down), not as
> an image. The model predicts these points, and we draw them later to make a picture.

---

## What's New

This project has been improved with production-ready features:

- **Mixed Precision Training** (`--use_amp`) — faster training on GPU, less memory.
- **Gradient Accumulation** (`--grad_accum_steps`) — simulate a bigger batch size.
- **Learning Rate Warmup** — smooth training start.
- **Cosine Annealing** — better learning-rate schedule.
- **Early Stopping** — stop when validation loss stops improving.
- **GAN gradient penalty (WGAN-GP)** (`--grad_penalty_weight`) — keeps the
  discriminator stable so adversarial training does not collapse.
- **Exponential Moving Average (EMA)** (`--use_ema`) — keeps a smoothed copy of the
  generator weights for higher-quality, more stable samples.
- **Top-k / Top-p (nucleus) sampling** (`--top_k`, `--top_p`) — modern decoding
  control for cleaner or more varied strokes.
- **TensorBoard** — watch training live in your browser.
- **Inference CLI** — generate handwriting from the terminal.
- **SVG export + themes** — save strokes as SVG in multiple styles.
- **Test suite** — full pytest coverage.

---

## Results Gallery

The sample images below are saved in this repository (under `output/samples/` and
`output_conditioned/samples/`). They show how strokes improve during training.

### Unconditional model (free-form strokes)

| Epoch 0 (random) | Epoch 24 (coherent) |
|---|---|
| ![uncond-epoch0](output/samples/epoch_0000.png) | ![uncond-epoch24](output/samples/epoch_0024.png) |

### Text-conditioned model (writes the given text)

| Epoch 0 (wobbly) | Epoch 9 (structured) |
|---|---|
| ![cond-epoch0](output_conditioned/samples/epoch_0000.png) | ![cond-epoch9](output_conditioned/samples/epoch_0009.png) |

> For clear, human-readable handwriting, train on the **real IAM-OnDB** dataset. The
> bundled synthetic data is only used to prove that the whole pipeline learns.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Project Layout](#project-layout)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data](#data)
- [Training](#training)
- [Sampling / Inference](#sampling--inference)
- [Evaluation](#evaluation)
- [Results](#results)
- [Training Flags Reference](#training-flags-reference)
- [Running Tests](#running-tests)
- [License](#license)
- [References](#references)

---

## How It Works

The model reads one step of pen movement `(dx, dy, pen_up)` and predicts the next one.
Instead of a single answer, it outputs a **mixture** of many possible next moves. This
captures the fact that handwriting is not fixed — the same word can be written in many
ways.

The model outputs:

| Output | Meaning |
|---|---|
| `mu_x`, `mu_y` | the center of each possible move |
| `sigma_x`, `sigma_y` | how spread out each move is |
| `rho` | how x and y moves relate |
| `pi` | how likely each possible move is |
| `pen_up` | probability the pen lifts |

The loss is the **negative log-likelihood** of the true next move under this mixture,
plus the loss for the pen-lift prediction. All math is done in log space, so it is
numerically stable.

### Two training modes

- **Unconditional** — learns the general shape of strokes and draws free-form.
- **Conditioned** — learns to write specific text using windowed attention. The model
  looks at the characters while it writes and moves its attention forward one word at a
  time (monotonic attention).

### The GAN part (optional)

A small 1D-CNN called a **discriminator** tries to tell real strokes from generated
strokes. The generator (the MDN-RNN) is pushed to fool it. This makes the strokes
sharper. The whole GAN is optional — just add `--use_gan`.

---

## Project Layout

```
Handwriting Generation using RNN + GAN/
├── data.py                       # Read IAM XML, normalize, batch, render
├── models.py                     # MDNRNN, MDNRNNConditioned, attention, discriminator
├── losses.py                     # MDN loss, adversarial loss, gradient penalty
├── ema.py                        # Exponential moving average of model weights
├── sampling.py                   # Temperature, top-k, top-p sampling helpers
├── train.py                      # Training loop (both modes) + config file support
├── eval.py                       # NLL evaluation, comparisons, demo API
├── inference.py                  # Inference CLI
├── generate_synthetic_data.py    # Make fake IAM-style XML data
├── render.py                     # SVG + multi-theme rendering
├── requirements.txt
├── pyproject.toml
├── LICENSE
├── README.md
├── tests/                        # pytest tests
├── output/                       # Unconditional run: samples, logs, stats
└── output_conditioned/           # Conditioned run: same structure
```

---

## Installation

You need **Python 3.10 or newer** and a CPU or CUDA GPU.

```bash
git clone https://github.com/Yasar-17/Handwriting-Generation-Using-RNN.git
cd "Handwriting-Generation-Using-RNN"

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

To check everything is wired correctly:

```bash
pytest tests/
```

---

## Quick Start

The fastest way to see handwriting without downloading anything:

```bash
# 1. Generate synthetic IAM-style XML (once)
python generate_synthetic_data.py --output_dir ./synthetic_data --num_samples 500

# 2. Train the conditioned model for a few epochs (CPU is fine)
python train.py --conditioned \
    --data_dir ./synthetic_data \
    --output_dir ./output_conditioned \
    --epochs 10 --batch_size 16 --hidden_dim 256 --num_layers 3 \
    --num_mixtures 20 --num_windows 10 --char_embed_dim 32 \
    --sample_every 2 --temperature 0.5 \
    --condition_text "hello world"

# 3. Generate handwriting for any text
python eval.py \
    --data_dir ./synthetic_data \
    --stats_path ./output_conditioned/stats.json \
    --ckpt ./output_conditioned/checkpoints/checkpoint_best.pt \
    --comparison_texts "hello world" "the quick brown fox" \
    --temperature 0.5 --output_dir ./eval_output
```

Images appear in `output_conditioned/samples/` (during training) and
`eval_output/demo/` (after training).

---

## Data

The project reads IAM **On-Line Handwriting** XML files (lines of points with a text
transcript). You have two options:

1. **Real IAM data** — put the line-level XML files in a folder and pass it with
   `--data_dir`. The script splits it 80/10/10 (train/val/test).
2. **Synthetic data** — run `generate_synthetic_data.py` to create fake IAM-style XML.
   No download needed.

```bash
python generate_synthetic_data.py --output_dir ./synthetic_data --num_samples 1000 --seed 42
```

### Preprocessing

- Convert absolute `(x, y, pen_up)` to relative `(dx, dy, pen_up)`.
- Z-score normalize using training statistics (saved to `stats.json`).
- Pad variable-length sequences and mask the padded parts in the loss.
- Encode characters with a simple `CharVocab` (index 0 = padding).

---

## Training

`train.py` supports both modes, with or without the GAN.

### Unconditional

```bash
python train.py \
    --data_dir ./synthetic_data \
    --output_dir ./output \
    --epochs 50 --batch_size 64 --hidden_dim 256 --num_layers 3 \
    --num_mixtures 20 --dropout 0.2 \
    --lr 1e-3 --clip_grad 5.0 \
    --sample_every 5 --sample_len 800 --temperature 0.5 \
    --seed 42
```

### Conditioned (write specific text)

```bash
python train.py --conditioned \
    --data_dir ./synthetic_data \
    --output_dir ./output_conditioned \
    --epochs 50 --batch_size 64 \
    --hidden_dim 256 --num_layers 3 --num_mixtures 20 \
    --num_windows 10 --char_embed_dim 32 \
    --condition_text "the quick brown fox jumps over the lazy dog" \
    --sample_every 5 --sample_len 1000 --temperature 0.5
```

### With the GAN

```bash
python train.py --conditioned --use_gan \
    --data_dir ./synthetic_data --output_dir ./output_conditioned_gan \
    --adv_weight 0.1 --disc_lr 1e-4 \
    --disc_hidden_dim 128 --disc_num_layers 4 --disc_dropout 0.2
```

### With a stable GAN (gradient penalty)

Add a WGAN-GP gradient penalty to stop the discriminator from overpowering the
generator:

```bash
python train.py --conditioned --use_gan --grad_penalty_weight 10.0 \
    --data_dir ./synthetic_data --output_dir ./output_conditioned_gp
```

A value around **10.0** is typical. `0` disables the penalty.

### With EMA (smoother, higher-quality samples)

EMA keeps a running average of the generator weights and uses them for sampling:

```bash
python train.py --conditioned --use_ema --ema_decay 0.999 \
    --data_dir ./synthetic_data --output_dir ./output_conditioned_ema
```

When EMA is enabled, the best EMA checkpoint is saved as
`checkpoint_best_ema.pt`. You can load it with `eval.py` / `inference.py` exactly
like a normal checkpoint.

### Mixed precision

```bash
python train.py --conditioned --use_amp \
    --data_dir ./synthetic_data --output_dir ./output_conditioned_amp
```

### Gradient accumulation

```bash
# Effective batch size = 16 * 4 = 64
python train.py --conditioned \
    --data_dir ./synthetic_data --output_dir ./output_conditioned_accum \
    --batch_size 16 --grad_accum_steps 4
```

### Warmup + cosine annealing

```bash
python train.py --conditioned \
    --data_dir ./synthetic_data --output_dir ./output_conditioned_warmup \
    --warmup_epochs 5 --use_cosine_annealing --epochs 50
```

### Early stopping

```bash
python train.py --conditioned \
    --data_dir ./synthetic_data --output_dir ./output_conditioned_es \
    --early_stopping_patience 10 --epochs 100
```

### Full production example

```bash
python train.py --conditioned --use_gan --use_amp \
    --data_dir ./synthetic_data --output_dir ./output_production \
    --epochs 100 --batch_size 32 --grad_accum_steps 2 \
    --warmup_epochs 5 --use_cosine_annealing \
    --early_stopping_patience 15 \
    --grad_penalty_weight 10.0 --use_ema --ema_decay 0.999 \
    --hidden_dim 256 --num_layers 3 --num_mixtures 20 \
    --num_windows 10 --char_embed_dim 32 \
    --condition_text "the quick brown fox jumps over the lazy dog" \
    --sample_every 5 --sample_len 1000 --temperature 0.5 \
    --top_k 5 --top_p 0.95 \
    --adv_weight 0.1 --disc_lr 1e-4
```

### Speed up conditioned training (`--chunk_size`)

The conditioned model must be unrolled one step at a time (because of the attention
feedback). `--chunk_size N` holds the attention context fixed inside each chunk, so
the LSTM runs `ceil(T / N)` times instead of `T` times. This is a big speedup with
almost no quality loss:

| `--chunk_size` | forward time (CPU) | speedup |
|---:|---:|---:|
| 1 (exact) | ~1950 ms | 1.0× |
| 16 | ~377 ms | ~5.2× |
| 32 | ~326 ms | ~6.0× |

Sampling always uses `chunk_size=1` for the best quality.

```bash
python train.py --conditioned --chunk_size 16 --data_dir ./synthetic_data ...
```

### Outputs

- `output/checkpoints/checkpoint_epoch_XXXX.pt` — per-epoch checkpoints.
- `output/checkpoints/checkpoint_best.pt` — best validation model.
- `output/checkpoints/checkpoint_best_ema.pt` — best EMA model (if `--use_ema`).
- `output/samples/epoch_XXXX.png` — rendered samples.
- `output/log.json` — per-epoch losses and learning rate.
- `output/stats.json` — normalization stats (needed for inference).

### Resume training

```bash
python train.py --conditioned --data_dir ./synthetic_data \
    --output_dir ./output_conditioned --resume ./output_conditioned/checkpoints/checkpoint_best.pt
```

---

## Sampling / Inference

### Python API

```python
from eval import generate_handwriting

fig = generate_handwriting(
    text="hello world",
    ckpt_path="./output_conditioned/checkpoints/checkpoint_best.pt",
    stats_path="./output_conditioned/stats.json",
    temperature=0.5,
)
fig.savefig("hello_world.png")
```

Or use the `HandwritingGenerator` class for repeated use:

```python
from eval import HandwritingGenerator

gen = HandwritingGenerator(
    ckpt_path="./output_conditioned/checkpoints/checkpoint_best.pt",
    stats_path="./output_conditioned/stats.json",
)
fig = gen.generate_handwriting("the quick brown fox", temperature=0.5)
```

### About temperature, top-k and top-p

During sampling the model picks one "next move" from its mixture:

- **Temperature** — lower (< 1) means crisp and repetitive; higher (> 1) means varied
  and noisier.
- **Top-k** (`--top_k N`) — only consider the `N` most likely moves.
- **Top-p** (`--top_p 0.95`) — only keep moves until their combined chance reaches
  `p` (nucleus sampling). This removes unlikely "tail" moves and often looks cleaner.

For conditioned generation the loop stops early when the pen stays lifted for many
steps (the text is finished).

### Inference CLI

```bash
python inference.py \
    --ckpt ./output_conditioned/checkpoints/checkpoint_best.pt \
    --stats ./output_conditioned/stats.json \
    --text "hello world" "machine learning" \
    --output_dir ./generated \
    --temperature 0.5 --num_samples 3 \
    --top_k 5 --top_p 0.95
```

---

## Evaluation

```bash
python eval.py \
    --data_dir ./synthetic_data \
    --stats_path ./output_conditioned/stats.json \
    --pre_ckpt ./output_conditioned/checkpoints/checkpoint_best.pt \
    --post_ckpt ./output_conditioned_gan/checkpoints/checkpoint_best.pt \
    --comparison_texts "hello world" "the quick brown fox" \
    --num_samples 3 --temperature 0.5 \
    --output_dir ./eval_output
```

This produces:

1. **Validation NLL** — average MDN negative log-likelihood on held-out data
   (saved to `eval_output/eval_results.json`).
2. **Pre/post-GAN comparison images** in `eval_output/comparisons/`.
3. **Demo samples** in `eval_output/demo/`.

---

## Results

Training was run for a small number of epochs on synthetic data to validate the full
pipeline. The negative log-likelihood decreases steadily for both models.

### Unconditional val NLL

| Epoch | 0    | 5     | 10    | 15    | 20    | 24    |
|-------|------|-------|-------|-------|-------|-------|
| Val NLL | 2.32 | 0.33 | −0.40 | −0.76 | −0.86 | −0.88 |

### Conditioned val NLL

| Epoch | 0    | 1    | 2    | 3    | 4    | 9    |
|-------|------|------|------|------|------|------|
| Val NLL | 2.01 | 0.47 | −0.13 | −0.33 | −0.44 | −0.84 |

> NLL is reported per timestep, so a **lower** number means the model fits better.

---

## Training Flags Reference

Quick reference of the most useful flags:

| Argument | Default | What it does |
|----------|---------|---------------|
| `--conditioned` | off | Text-conditioned + attention mode |
| `--hidden_dim` | 256 | LSTM hidden size |
| `--num_layers` | 3 | LSTM layers |
| `--num_mixtures` | 20 | Number of Gaussian components |
| `--num_windows` | 10 | Attention windows (conditioned only) |
| `--batch_size` | 64 | Minibatch size |
| `--epochs` | 50 | Number of epochs |
| `--lr` | 1e-3 | Learning rate |
| `--temperature` | 0.5 | Sampling temperature |
| `--top_k` | 0 | Top-k sampling (0 = off) |
| `--top_p` | 1.0 | Top-p nucleus sampling (1.0 = off) |
| `--use_gan` | off | Enable adversarial training |
| `--adv_weight` | 0.1 | Adversarial loss weight |
| `--grad_penalty_weight` | 0.0 | WGAN-GP gradient penalty (0 = off) |
| `--use_ema` | off | Use EMA of generator weights |
| `--ema_decay` | 0.999 | EMA decay factor |
| `--use_amp` | off | Mixed precision training |
| `--grad_accum_steps` | 1 | Gradient accumulation steps |
| `--warmup_epochs` | 5 | LR warmup epochs |
| `--use_cosine_annealing` | off | Cosine annealing schedule |
| `--early_stopping_patience` | 0 | Early stopping (0 = off) |
| `--chunk_size` | 1 | Conditioned training speedup |
| `--config` | None | YAML config file |
| `--seed` | 42 | Random seed |

---

## Running Tests

```bash
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=.
```

---

## License

This project is released under the [MIT License](./LICENSE).

---

## References

- Alex Graves. **"Generating Sequences With Recurrent Neural Networks."** *Neural
  Computation*, 2013. — the original MDN-RNN + windowed attention handwriting model.
- IAM On-Line Handwriting Database: A. Graves & J. Schmidhuber, *IAM-OnDB*, University
  of Bern.
- Bishop, C. M. **"Mixture Density Networks."** Technical Report NCRG/94/004, Aston
  University, 1994.
- Goodfellow, I. et al. **"Generative Adversarial Nets."** NeurIPS 2014.
