"""
Gradio web interface for interactive handwriting generation.

Provides a browser-based UI for generating handwriting from text with
adjustable parameters.

Usage:
    python app.py
"""

import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data import CharVocab, denormalize_deltas, render_strokes
from models import MDNRNN, MDNRNNConditioned

try:
    import gradio as gr
except ImportError:
    raise ImportError("Install gradio: pip install gradio")


def _load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path("./output_conditioned/checkpoints/checkpoint_best.pt")
    stats_path = Path("./output_conditioned/stats.json")

    if not ckpt_path.exists():
        ckpt_path = Path("./output/checkpoints/checkpoint_best.pt")
        stats_path = Path("./output/stats.json")

    if not ckpt_path.exists():
        return None, None, None, None, device

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
            input_dim=3, hidden_dim=hidden_dim, num_layers=num_layers,
            num_mixtures=num_mixtures, num_windows=10,
            char_vocab_size=len(vocab), char_embed_dim=32, dropout=0.0,
        )
    else:
        vocab = None
        model = MDNRNN(
            input_dim=3, hidden_dim=hidden_dim, num_layers=num_layers,
            num_mixtures=num_mixtures, dropout=0.0,
        )

    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {
        "mean_x": 0.0, "std_x": 1.0, "mean_y": 0.0, "std_y": 1.0,
    }

    return model, ckpt, stats, vocab, device


@torch.no_grad()
def generate_handwriting(text, temperature, seed, max_len):
    if model is None:
        raise gr.Error("No model found. Train a model first.")

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    conditioned = ckpt.get("conditioned", False)
    device_ = device

    if conditioned:
        char_ids = vocab.encode(text)
        char_tensor = torch.tensor([char_ids], dtype=torch.long, device=device_)
        char_mask = torch.ones(1, len(char_ids), dtype=torch.bool, device=device_)

        hidden = model.init_hidden(1, device=device_)
        x = torch.zeros(1, 1, 3, device=device_)
        deltas = []
        consecutive_pen_up = 0

        for _ in range(int(max_len)):
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
            x = torch.tensor([[[dx, dy, pen_up]]], device=device_, dtype=torch.float32)
    else:
        hidden = model.init_hidden(1, device=device_)
        x = torch.zeros(1, 1, 3, device=device_)
        deltas = []

        for _ in range(int(max_len)):
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
            x = torch.tensor([[[dx, dy, pen_up]]], device=device_, dtype=torch.float32)

    deltas_arr = np.array(deltas, dtype=np.float32)
    if len(deltas_arr) > 0 and deltas_arr[:, 2].sum() == 0:
        deltas_arr[-1, 2] = 1.0

    deltas_denorm = denormalize_deltas(
        deltas_arr, stats["mean_x"], stats["std_x"], stats["mean_y"], stats["std_y"]
    )

    fig = render_strokes(deltas_denorm, title=text, figsize=(10, 4))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)
    buf.seek(0)

    from PIL import Image
    img = Image.open(buf)

    return img, f"Generated {len(deltas_arr)} stroke points | Seed: {int(seed)}"


model, ckpt, stats, vocab, device = _load_model()

demo = gr.Interface(
    fn=generate_handwriting,
    inputs=[
        gr.Textbox(
            label="Enter Text",
            placeholder="Type something here...",
            value="Hello World",
            lines=2,
        ),
        gr.Slider(
            minimum=0.1, maximum=2.0, value=0.5, step=0.1,
            label="Temperature (lower = more deterministic)",
        ),
        gr.Number(value=42, label="Random Seed", precision=0),
        gr.Slider(
            minimum=100, maximum=3000, value=1000, step=100,
            label="Max Sequence Length",
        ),
    ],
    outputs=[
        gr.Image(label="Generated Handwriting", type="pil"),
        gr.Textbox(label="Info"),
    ],
    title="Handwriting Generation with MDN-RNN + GAN",
    description=(
        "Generate realistic handwriting from text using a Mixture-Density Network "
        "with LSTM and optional GAN refinement, based on Alex Graves' seminal work. "
        "Adjust temperature to control creativity vs. determinism."
    ),
    article=(
        "Built with PyTorch. "
        "Model: 3-layer LSTM (hidden_dim=256) with 20-component Mixture Density Network output. "
        "Conditioned variant uses monotonic windowed attention over character embeddings."
    ),
    examples=[
        ["Hello World", 0.5, 42, 1000],
        ["the quick brown fox", 0.3, 7, 1500],
        ["PyTorch is amazing", 0.7, 123, 1000],
        ["Deep Learning", 0.4, 99, 800],
    ],
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
