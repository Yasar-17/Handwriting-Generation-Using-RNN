"""
Sanity-check script for the IAM handwriting data pipeline.

Usage:
    python sanity_check.py --data_dir /path/to/iam/xml  [--num_samples 5] [--output_dir ./sanity_output]

Downloads a couple of sample XML files if no data directory is provided,
then exercises the full pipeline: parse -> normalize -> render -> denormalize -> render.
"""

import argparse
import sys
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Ensure the project root is on sys.path so we can import data
sys.path.insert(0, str(Path(__file__).parent))

from data import (
    absolute_to_relative,
    build_dataloader,
    collect_xml_files,
    compute_dataset_stats,
    denormalize_deltas,
    normalize_deltas,
    parse_iam_xml,
    render_strokes,
)

# ---------------------------------------------------------------------------
# Minimal sample XML for offline testing (no download required)
# ---------------------------------------------------------------------------

SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<StrokeSet>
  <Stroke>
    <Point x="100" y="200" time="0"/>
    <Point x="110" y="195" time="10"/>
    <Point x="125" y="190" time="20"/>
    <Point x="140" y="192" time="30"/>
    <Point x="155" y="200" time="40"/>
    <Point x="160" y="210" time="50"/>
  </Stroke>
  <Stroke>
    <Point x="180" y="195" time="100"/>
    <Point x="190" y="190" time="110"/>
    <Point x="200" y="188" time="120"/>
    <Point x="210" y="192" time="130"/>
    <Point x="215" y="200" time="140"/>
  </Stroke>
  <Stroke>
    <Point x="105" y="220" time="200"/>
    <Point x="115" y="225" time="210"/>
    <Point x="130" y="230" time="220"/>
    <Point x="145" y="228" time="230"/>
    <Point x="155" y="220" time="240"/>
  </Stroke>
</StrokeSet>
"""


def create_sample_xml(output_dir: Path) -> list[Path]:
    """Write a few synthetic XML files for testing without IAM data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(3):
        path = output_dir / f"sample_{i:03d}.xml"
        path.write_text(SAMPLE_XML)
        paths.append(path)
    return paths


def download_sample_xml(output_dir: Path, urls: list[str]) -> list[Path]:
    """Download real IAM XML samples from the web."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for url in urls:
        fname = url.split("/")[-1]
        path = output_dir / fname
        if not path.exists():
            print(f"  Downloading {fname} ...")
            urllib.request.urlretrieve(url, path)
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity-check IAM data pipeline")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing IAM XML files")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to render")
    parser.add_argument("--output_dir", type=str, default="./sanity_output", help="Where to save output images")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Get XML files -------------------------------------------------------
    # -----------------------------------------------------------------------
    if args.data_dir and Path(args.data_dir).exists():
        xml_files = collect_xml_files(args.data_dir)
        if not xml_files:
            print(f"No XML files found in {args.data_dir}. Falling back to synthetic samples.")
            xml_files = create_sample_xml(output_dir / "synthetic")
    else:
        print("No --data_dir provided (or path invalid). Using synthetic XML samples.")
        xml_files = create_sample_xml(output_dir / "synthetic")

    xml_files = xml_files[: args.num_samples]
    print(f"\nUsing {len(xml_files)} XML files:")
    for f in xml_files:
        print(f"  {f}")

    # -----------------------------------------------------------------------
    # 2. Parse & render raw strokes ------------------------------------------
    # -----------------------------------------------------------------------
    print("\n--- Step 1: Parse XML and render raw strokes ---")
    all_sequences = []
    for xml_path in xml_files:
        pts, _ = parse_iam_xml(xml_path)
        print(f"  {xml_path.name}: {len(pts)} points")
        all_sequences.append(pts)

        fig = render_strokes(absolute_to_relative(pts), title=f"Raw: {xml_path.name}")
        fig.savefig(output_dir / f"raw_{xml_path.stem}.png")
        plt.close(fig)

    # -----------------------------------------------------------------------
    # 3. Compute stats, normalize, denormalize, verify round-trip -----------
    # -----------------------------------------------------------------------
    print("\n--- Step 2: Compute normalization stats ---")
    mean_x, std_x, mean_y, std_y = compute_dataset_stats(xml_files)
    print(f"  mean_x={mean_x:.4f}, std_x={std_x:.4f}")
    print(f"  mean_y={mean_y:.4f}, std_y={std_y:.4f}")

    print("\n--- Step 3: Normalize -> Denormalize round-trip ---")
    for i, pts in enumerate(all_sequences):
        deltas = absolute_to_relative(pts)
        normed = normalize_deltas(deltas, mean_x, std_x, mean_y, std_y)
        denormed = denormalize_deltas(normed, mean_x, std_x, mean_y, std_y)
        err = np.abs(deltas - denormed).max()
        print(f"  {xml_files[i].name}: max round-trip error = {err:.2e}")
        assert err < 1e-5, f"Round-trip error too large: {err}"

        fig = render_strokes(normed, title=f"Normalized: {xml_files[i].name}")
        fig.savefig(output_dir / f"normalized_{xml_files[i].stem}.png")
        plt.close(fig)

    # -----------------------------------------------------------------------
    # 4. Dataset + DataLoader ------------------------------------------------
    # -----------------------------------------------------------------------
    print("\n--- Step 4: Build Dataset + DataLoader ---")
    dl = build_dataloader(
        xml_files,
        batch_size=2,
        shuffle=False,
        mean_x=mean_x,
        std_x=std_x,
        mean_y=mean_y,
        std_y=std_y,
    )
    print(f"  Dataset size: {len(dl.dataset)}")
    print(f"  Number of batches: {len(dl)}")

    for batch_idx, batch in enumerate(dl):
        data = batch["data"]
        batch["mask"]
        lengths = batch["lengths"]
        print(f"  Batch {batch_idx}: data shape={data.shape}, lengths={lengths.tolist()}")

        # Render first sample from batch (denormalize first)
        seq = data[0].numpy()
        valid_len = lengths[0].item()
        seq = seq[:valid_len]
        denormed = denormalize_deltas(seq, mean_x, std_x, mean_y, std_y)

        fig = render_strokes(denormed, title=f"From DataLoader batch {batch_idx}")
        fig.savefig(output_dir / f"dataloader_batch{batch_idx}.png")
        plt.close(fig)

    # -----------------------------------------------------------------------
    # 5. Summary -------------------------------------------------------------
    # -----------------------------------------------------------------------
    print(f"\nAll checks passed. Output images saved to: {output_dir}")
    print("Files produced:")
    for p in sorted(output_dir.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
