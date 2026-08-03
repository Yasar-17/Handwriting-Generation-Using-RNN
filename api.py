"""
FastAPI REST API for handwriting generation.

Provides endpoints for generating handwriting images from text input.
Supports both JSON and multipart form data, with PNG and SVG output formats.

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000
"""

import base64
import io
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from data import CharVocab, denormalize_deltas, render_strokes
from models import MDNRNN, MDNRNNConditioned

logger = logging.getLogger(__name__)

_model = None
_ckpt = None
_stats = None
_vocab = None
_device = None


def _load_resources():
    global _model, _ckpt, _stats, _vocab, _device

    ckpt_path = Path("./output_conditioned/checkpoints/checkpoint_best.pt")
    stats_path = Path("./output_conditioned/stats.json")

    if not ckpt_path.exists():
        ckpt_path = Path("./output/checkpoints/checkpoint_best.pt")
        stats_path = Path("./output/stats.json")

    if not ckpt_path.exists():
        logger.warning("No checkpoint found. API will return 503 until a model is trained.")
        return

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ckpt = torch.load(ckpt_path, map_location=_device, weights_only=False)
    model_state = _ckpt["model"]
    conditioned = _ckpt.get("conditioned", False)

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
        _vocab = CharVocab()
        _model = MDNRNNConditioned(
            input_dim=3, hidden_dim=hidden_dim, num_layers=num_layers,
            num_mixtures=num_mixtures, num_windows=10,
            char_vocab_size=len(_vocab), char_embed_dim=32, dropout=0.0,
        )
    else:
        _model = MDNRNN(
            input_dim=3, hidden_dim=hidden_dim, num_layers=num_layers,
            num_mixtures=num_mixtures, dropout=0.0,
        )

    _model.load_state_dict(model_state)
    _model.to(_device)
    _model.eval()

    if stats_path.exists():
        _stats = json.loads(stats_path.read_text())
    else:
        _stats = {"mean_x": 0.0, "std_x": 1.0, "mean_y": 0.0, "std_y": 1.0}

    logger.info("Model loaded | Conditioned: %s | Device: %s", conditioned, _device)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_resources()
    yield


app = FastAPI(
    title="Handwriting Generation API",
    description="Generate handwriting images from text using MDN-RNN + GAN",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500, description="Text to generate handwriting for")
    temperature: float = Field(0.5, ge=0.1, le=2.0, description="Sampling temperature")
    max_seq_len: int = Field(1000, ge=100, le=5000, description="Maximum sequence length")
    seed: int = Field(42, ge=0, description="Random seed for reproducibility")


class GenerateResponse(BaseModel):
    text: str
    image_base64: str
    width: int
    height: int
    sequence_length: int
    generation_time_ms: float


@torch.no_grad()
def _generate_strokes(text: str, temperature: float, max_seq_len: int, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)

    conditioned = _ckpt.get("conditioned", False)
    mean_x = _stats["mean_x"]
    std_x = _stats["std_x"]
    mean_y = _stats["mean_y"]
    std_y = _stats["std_y"]

    if conditioned:
        char_ids = _vocab.encode(text)
        char_tensor = torch.tensor([char_ids], dtype=torch.long, device=_device)
        char_mask = torch.ones(1, len(char_ids), dtype=torch.bool, device=_device)

        hidden = _model.init_hidden(1, device=_device)
        x = torch.zeros(1, 1, 3, device=_device)
        deltas = []
        consecutive_pen_up = 0

        for _ in range(max_seq_len):
            params, hidden, _ = _model(x, char_tensor, char_mask, hidden, chunk_size=1)

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
            x = torch.tensor([[[dx, dy, pen_up]]], device=_device, dtype=torch.float32)
    else:
        hidden = _model.init_hidden(1, device=_device)
        x = torch.zeros(1, 1, 3, device=_device)
        deltas = []

        for _ in range(max_seq_len):
            params, hidden = _model(x, hidden)

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
            x = torch.tensor([[[dx, dy, pen_up]]], device=_device, dtype=torch.float32)

    deltas_arr = np.array(deltas, dtype=np.float32)
    if len(deltas_arr) > 0 and deltas_arr[:, 2].sum() == 0:
        deltas_arr[-1, 2] = 1.0

    return denormalize_deltas(deltas_arr, mean_x, std_x, mean_y, std_y)


@app.get("/health")
async def health():
    return {
        "status": "healthy" if _model is not None else "no_model",
        "device": str(_device) if _device else "N/A",
        "conditioned": _ckpt.get("conditioned", False) if _ckpt else False,
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="No model loaded. Train a model first.")

    start = time.time()
    deltas = _generate_strokes(req.text, req.temperature, req.max_seq_len, req.seed)
    fig = render_strokes(deltas, title=req.text)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)
    buf.seek(0)

    img_bytes = buf.getvalue()
    elapsed = (time.time() - start) * 1000

    from PIL import Image
    img = Image.open(io.BytesIO(img_bytes))
    w, h = img.size

    return GenerateResponse(
        text=req.text,
        image_base64=base64.b64encode(img_bytes).decode(),
        width=w,
        height=h,
        sequence_length=len(deltas),
        generation_time_ms=round(elapsed, 2),
    )


@app.get("/generate/png")
async def generate_png(
    text: str = Query(..., min_length=1, max_length=500),
    temperature: float = Query(0.5, ge=0.1, le=2.0),
    seed: int = Query(42, ge=0),
):
    if _model is None:
        raise HTTPException(status_code=503, detail="No model loaded.")

    deltas = _generate_strokes(text, temperature, 1000, seed)
    fig = render_strokes(deltas, title=text)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")


@app.get("/")
async def root():
    return {
        "name": "Handwriting Generation API",
        "version": "2.0.0",
        "endpoints": {
            "POST /generate": "Generate handwriting (JSON body, returns base64 PNG)",
            "GET /generate/png?text=...": "Generate handwriting (query params, returns PNG)",
            "GET /health": "Health check",
            "GET /docs": "Interactive API documentation",
        },
    }
