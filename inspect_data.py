"""
Phase 1 Data Sanity Check: visualize X-ray projection frames with timestamps.

Usage:
    python -m simple_fusion.inspect_data \\
        --source-path data/patient.pkl \\
        --output-dir outputs/data_check \\
        [--n-frames 16]

Outputs (written to --output-dir):
    train_grid.png      grid of equally-spaced training projection frames
    test_grid.png       grid of test projection frames
    time_angle.png      scatter: projection angle vs normalised time (colour = phase)
    phase_dist.png      histogram of respiratory-phase labels
    scanner_cfg.txt     human-readable scanner configuration
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from simple_fusion import _bootstrap  # noqa: F401
from simple_fusion.data import load_naf_scene


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualise NAF .pickle dataset contents."
    )
    p.add_argument("--source-path", required=True, help="Path to .pickle file.")
    p.add_argument("--output-dir", default="outputs/data_check")
    p.add_argument(
        "--n-frames", type=int, default=16,
        help="How many evenly-spaced frames to show in each split grid.",
    )
    p.add_argument(
        "--grid-cols", type=int, default=8,
        help="Number of columns in the projection grid.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _projection_grid(
    cameras: list,
    n_frames: int,
    n_cols: int,
    title: str,
    save_path: Path,
) -> None:
    """Save a grid of X-ray projections sampled evenly across `cameras`."""
    if not cameras:
        return

    # Sample evenly
    indices = np.linspace(0, len(cameras) - 1, min(n_frames, len(cameras)), dtype=int)
    selected = [cameras[i] for i in indices]

    n_rows = int(np.ceil(len(selected) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.8))
    axes = np.array(axes).reshape(n_rows, n_cols)

    global_max = max(
        cam.original_image.float().max().item() for cam in selected
    )
    global_max = max(global_max, 1e-6)

    for idx, cam in enumerate(selected):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]
        img = cam.original_image.squeeze().cpu().float().numpy()
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=global_max)
        phase = int(getattr(cam, "phase", -1))
        angle = float(getattr(cam, "angle", float("nan")))
        ax.set_title(
            f"t={cam.time:.3f}\nph={phase}  θ={angle:.1f}°",
            fontsize=7,
        )
        ax.axis("off")

    # Hide unused panels
    for idx in range(len(selected), n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].axis("off")

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_path}")


def _time_angle_scatter(
    train_cameras: list,
    test_cameras: list,
    save_path: Path,
) -> None:
    """Scatter: projection angle vs normalised time, coloured by phase."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for ax, cameras, label in [
        (axes[0], train_cameras, "Train"),
        (axes[1], test_cameras, "Test"),
    ]:
        if not cameras:
            ax.set_title(f"{label}: no cameras")
            continue
        times = [cam.time for cam in cameras]
        angles = [float(getattr(cam, "angle", 0.0)) for cam in cameras]
        phases = [int(getattr(cam, "phase", 0)) for cam in cameras]

        sc = ax.scatter(times, angles, c=phases, cmap="tab10", s=12, alpha=0.8)
        plt.colorbar(sc, ax=ax, label="Phase")
        ax.set_xlabel("Normalised time t")
        ax.set_ylabel("Projection angle (°)")
        ax.set_title(f"{label} split  ({len(cameras)} views)")
        ax.grid(True, linewidth=0.3)

    fig.suptitle("Projection angle vs normalised time (colour = respiratory phase)", fontsize=11)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_path}")


def _phase_histogram(
    train_cameras: list,
    test_cameras: list,
    save_path: Path,
) -> None:
    """Histogram of phase labels for each split."""
    all_phases = sorted({int(getattr(c, "phase", 0)) for c in train_cameras + test_cameras})
    if len(all_phases) <= 1:
        return  # nothing interesting

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=False)
    for ax, cameras, label in [
        (axes[0], train_cameras, "Train"),
        (axes[1], test_cameras, "Test"),
    ]:
        phases = [int(getattr(c, "phase", 0)) for c in cameras]
        counts = [phases.count(p) for p in all_phases]
        ax.bar(all_phases, counts, color="steelblue", alpha=0.85)
        ax.set_xticks(all_phases)
        ax.set_xlabel("Phase label")
        ax.set_ylabel("Count")
        ax.set_title(f"{label} split")
        ax.grid(True, linewidth=0.3, axis="y")

    fig.suptitle("Respiratory phase distribution", fontsize=11)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_path}")


def _save_scanner_cfg(scanner_cfg: dict, scene_scale: float, save_path: Path) -> None:
    lines = ["Scanner Configuration (scene-scale normalised)\n", "=" * 48]
    lines.append(f"scene_scale : {scene_scale:.6f}  (2 / max_sVoxel in mm)")
    for k, v in scanner_cfg.items():
        lines.append(f"  {k:<16}: {v}")
    text = "\n".join(lines) + "\n"
    save_path.write_text(text)
    print(f"  Saved → {save_path}")
    print(text)


def _projection_stats(cameras: list, label: str) -> None:
    if not cameras:
        return
    all_pixels = torch.cat(
        [cam.original_image.float().flatten() for cam in cameras]
    )
    print(
        f"  {label:6s} projections:  "
        f"min={all_pixels.min():.4f}  max={all_pixels.max():.4f}  "
        f"mean={all_pixels.mean():.4f}  std={all_pixels.std():.4f}  "
        f"shape={cameras[0].original_image.shape}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading scene …")
    scene = load_naf_scene(
        args.source_path,
        device="cpu",   # keep on CPU for visualisation; no rasterizer needed
        eval_split=True,
        include_volume=False,
    )

    print(
        f"  Train cameras : {len(scene.train_cameras)}\n"
        f"  Test  cameras : {len(scene.test_cameras)}"
    )

    # --- Projection value statistics ---
    print("\nProjection statistics:")
    _projection_stats(scene.train_cameras, "train")
    _projection_stats(scene.test_cameras, "test")

    # --- Scanner config ---
    print("\nScanner configuration:")
    _save_scanner_cfg(
        scene.scanner_cfg,
        scene.scene_scale,
        output_dir / "scanner_cfg.txt",
    )

    # --- Bounding box ---
    bbox = scene.bbox
    print(f"\nScene bbox (normalised):\n  min={bbox[0].tolist()}\n  max={bbox[1].tolist()}")

    # --- Projection grids ---
    print("\nGenerating projection grids …")
    _projection_grid(
        scene.train_cameras,
        n_frames=args.n_frames,
        n_cols=args.grid_cols,
        title=f"Train projections  ({len(scene.train_cameras)} views total, {args.n_frames} shown)",
        save_path=output_dir / "train_grid.png",
    )
    _projection_grid(
        scene.test_cameras,
        n_frames=args.n_frames,
        n_cols=args.grid_cols,
        title=f"Test projections  ({len(scene.test_cameras)} views total, {args.n_frames} shown)",
        save_path=output_dir / "test_grid.png",
    )

    # --- Time / angle scatter ---
    print("Generating time-angle scatter …")
    _time_angle_scatter(
        scene.train_cameras,
        scene.test_cameras,
        save_path=output_dir / "time_angle.png",
    )

    # --- Phase histogram ---
    print("Generating phase histogram …")
    _phase_histogram(
        scene.train_cameras,
        scene.test_cameras,
        save_path=output_dir / "phase_dist.png",
    )

    print(f"\nAll outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
