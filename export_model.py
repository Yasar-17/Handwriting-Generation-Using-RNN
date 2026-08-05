"""
Model export utilities for TorchScript and ONNX.

Provides functions to export trained MDN-RNN models to TorchScript (.pt)
and ONNX (.onnx) formats for deployment and inference optimization.
"""

import json
from pathlib import Path

import torch


def export_to_torchscript(
    ckpt_path: str | Path,
    output_path: str | Path,
    conditioned: bool = False,
) -> Path:
    """Export a trained model checkpoint to TorchScript format.

    Args:
        ckpt_path: Path to the model checkpoint (.pt)
        output_path: Path to save the TorchScript model
        conditioned: Whether the model is text-conditioned

    Returns:
        Path to the exported TorchScript model
    """
    ckpt_path = Path(ckpt_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_state = ckpt["model"]

    if conditioned:
        from data import CharVocab
        from models import MDNRNNConditioned

        vocab = CharVocab()
        hidden_dim = model_state.get("lstm.weight_hh_l0", None)
        hidden_dim = hidden_dim.shape[1] if hidden_dim is not None else 256

        num_layers = 0
        for key in model_state:
            if key.startswith("lstm.weight_hh_l"):
                layer_idx = int(key.split("lstm.weight_hh_l")[1][0])
                num_layers = max(num_layers, layer_idx + 1)

        num_mixtures = 20
        for key in model_state:
            if "mdn_head.weight" in key:
                num_mixtures = model_state[key].shape[0] // 6
                break

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
        from models import MDNRNN

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

        model = MDNRNN(
            input_dim=3,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_mixtures=num_mixtures,
            dropout=0.0,
        )

    model.load_state_dict(model_state)
    model.eval()

    scripted_model = torch.jit.script(model)
    scripted_model.save(output_path)
    print(f"TorchScript model saved to {output_path}")
    return output_path


def export_to_onnx(
    ckpt_path: str | Path,
    output_path: str | Path,
    conditioned: bool = False,
    seq_len: int = 100,
    opset_version: int = 17,
) -> Path:
    """Export a trained model checkpoint to ONNX format.

    Args:
        ckpt_path: Path to the model checkpoint (.pt)
        output_path: Path to save the ONNX model
        conditioned: Whether the model is text-conditioned
        seq_len: Sequence length for the dummy input
        opset_version: ONNX opset version

    Returns:
        Path to the exported ONNX model
    """
    ckpt_path = Path(ckpt_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_state = ckpt["model"]

    if conditioned:
        from data import CharVocab
        from models import MDNRNNConditioned

        vocab = CharVocab()
        hidden_dim = model_state.get("lstm.weight_hh_l0", None)
        hidden_dim = hidden_dim.shape[1] if hidden_dim is not None else 256

        num_layers = 0
        for key in model_state:
            if key.startswith("lstm.weight_hh_l"):
                layer_idx = int(key.split("lstm.weight_hh_l")[1][0])
                num_layers = max(num_layers, layer_idx + 1)

        num_mixtures = 20
        for key in model_state:
            if "mdn_head.weight" in key:
                num_mixtures = model_state[key].shape[0] // 6
                break

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
        from models import MDNRNN

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

        model = MDNRNN(
            input_dim=3,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_mixtures=num_mixtures,
            dropout=0.0,
        )

    model.load_state_dict(model_state)
    model.eval()

    batch_size = 1
    dummy_input = torch.randn(batch_size, seq_len, 3)

    if conditioned:
        dummy_char_ids = torch.randint(0, 80, (batch_size, 20), dtype=torch.long)
        dummy_char_mask = torch.ones(batch_size, 20, dtype=torch.bool)
        dummy_hidden = model.init_hidden(batch_size)

        torch.onnx.export(
            model,
            (dummy_input, dummy_char_ids, dummy_char_mask, dummy_hidden),
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["strokes", "char_ids", "char_mask", "hidden_h", "hidden_c"],
            output_names=[
                "mu_x",
                "mu_y",
                "sigma_x",
                "sigma_y",
                "rho",
                "pi",
                "pen_up",
                "hidden_h_out",
                "hidden_c_out",
                "attention_weights",
            ],
            dynamic_axes={
                "strokes": {0: "batch", 1: "sequence"},
                "char_ids": {0: "batch", 1: "chars"},
                "char_mask": {0: "batch", 1: "chars"},
                "hidden_h": {1: "batch"},
                "hidden_c": {1: "batch"},
                "mu_x": {0: "batch", 1: "sequence"},
                "mu_y": {0: "batch", 1: "sequence"},
                "sigma_x": {0: "batch", 1: "sequence"},
                "sigma_y": {0: "batch", 1: "sequence"},
                "rho": {0: "batch", 1: "sequence"},
                "pi": {0: "batch", 1: "sequence"},
                "pen_up": {0: "batch", 1: "sequence"},
                "hidden_h_out": {1: "batch"},
                "hidden_c_out": {1: "batch"},
                "attention_weights": {0: "batch", 1: "sequence"},
            },
        )
    else:
        dummy_hidden = model.init_hidden(batch_size)

        torch.onnx.export(
            model,
            (dummy_input, dummy_hidden),
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["strokes", "hidden_h", "hidden_c"],
            output_names=["mu_x", "mu_y", "sigma_x", "sigma_y", "rho", "pi", "pen_up", "hidden_h_out", "hidden_c_out"],
            dynamic_axes={
                "strokes": {0: "batch", 1: "sequence"},
                "hidden_h": {1: "batch"},
                "hidden_c": {1: "batch"},
                "mu_x": {0: "batch", 1: "sequence"},
                "mu_y": {0: "batch", 1: "sequence"},
                "sigma_x": {0: "batch", 1: "sequence"},
                "sigma_y": {0: "batch", 1: "sequence"},
                "rho": {0: "batch", 1: "sequence"},
                "pi": {0: "batch", 1: "sequence"},
                "pen_up": {0: "batch", 1: "sequence"},
                "hidden_h_out": {1: "batch"},
                "hidden_c_out": {1: "batch"},
            },
        )

    print(f"ONNX model saved to {output_path}")
    return output_path


def export_model(
    ckpt_path: str | Path,
    output_dir: str | Path,
    stats_path: str | Path | None = None,
    fmt: str = "torchscript",
) -> dict[str, Path]:
    """Export a model checkpoint to the specified format.

    Args:
        ckpt_path: Path to the model checkpoint
        output_dir: Directory to save exported models
        stats_path: Optional path to stats.json to include in export
        fmt: Export format ("torchscript", "onnx", or "both")

    Returns:
        Dictionary with paths to exported files
    """
    ckpt_path = Path(ckpt_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    conditioned = ckpt.get("conditioned", False)

    exported = {}

    if fmt in ("torchscript", "both"):
        ts_path = output_dir / "model.torchscript.pt"
        exported["torchscript"] = export_to_torchscript(ckpt_path, ts_path, conditioned)

    if fmt in ("onnx", "both"):
        onnx_path = output_dir / "model.onnx"
        exported["onnx"] = export_to_onnx(ckpt_path, onnx_path, conditioned)

    if stats_path is not None:
        import shutil

        stats_dest = output_dir / "stats.json"
        shutil.copy(stats_path, stats_dest)
        exported["stats"] = stats_dest

    export_info = {
        "conditioned": conditioned,
        "format": fmt,
        "exported_files": {k: str(v) for k, v in exported.items()},
    }
    (output_dir / "export_info.json").write_text(json.dumps(export_info, indent=2))

    return exported


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export trained MDN-RNN models")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./exported_model", help="Output directory")
    parser.add_argument("--stats_path", type=str, default=None, help="Path to stats.json")
    parser.add_argument("--format", type=str, default="torchscript", choices=["torchscript", "onnx", "both"])
    args = parser.parse_args()

    export_model(args.ckpt_path, args.output_dir, args.stats_path, args.format)
