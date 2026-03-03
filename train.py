from __future__ import annotations

import argparse
import math
from pathlib import Path
import random

import torch
import numpy as np

from simple_fusion.data import (
    load_init_point_cloud,
    load_naf_scene,
    load_optional_latents,
)
from simple_fusion.model import (
    SimpleFusionModel,
    initialize_canonical_gaussian_params,
    initialize_canonical_latent_from_encoder,
)
from simple_fusion.rendering import render_projection
from simple_fusion.loss_utils import l1_loss, ssim, tv_3d_loss
from xray_gaussian_rasterization_voxelization import GaussianVoxelizer, GaussianVoxelizationSettings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple SF-VAE + X2-Gaussian fusion trainer")
    parser.add_argument(
        "--preset",
        type=str,
        default="default",
        choices=["default", "paper2503"],
        help="Parameter preset. 'paper2503' applies common 3DGS-style schedule used in arXiv:2503.21779.",
    )
    parser.add_argument("--source-path", type=str, required=True, help="Path to the NAF .pickle dataset.")
    parser.add_argument(
        "--decoder-checkpoint",
        type=str,
        default="gs-embedding/checkpoints/checkpoint_sfvae_sh0_144.pth",
        help="Path to the pretrained SF-VAE checkpoint.",
    )
    parser.add_argument(
        "--init-path",
        type=str,
        default=None,
        help="Optional init_*.npy point cloud path. Defaults to the X2-Gaussian naming convention.",
    )
    parser.add_argument(
        "--latent-path",
        type=str,
        default=None,
        help="Optional canonical latent file (.npy or .npz with key 'emb').",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/simple_fusion")
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--grid-dim", type=int, default=12)
    parser.add_argument("--max-gaussians", type=int, default=8192)
    parser.add_argument(
        "--decoder-mode",
        type=str,
        default="direct",
        choices=["direct", "point"],
        help="Use direct Gaussian parameter decoding or the older point-cloud fallback.",
    )
    parser.add_argument(
        "--latent-init",
        type=str,
        default="encoder",
        choices=["encoder", "random", "file"],
        help="How to initialize canonical latent codes when training starts.",
    )
    parser.add_argument(
        "--bootstrap-num-points",
        type=int,
        default=144,
        help="Number of synthetic local points used for encoder-based latent initialization.",
    )
    parser.add_argument(
        "--encode-chunk-size",
        type=int,
        default=1024,
        help="Batch size for encoder-based latent initialization.",
    )
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-final", type=float, default=1e-5, help="Final LR for cosine annealing.")
    parser.add_argument("--lambda-ssim", type=float, default=0.25)
    parser.add_argument("--lambda-latent", type=float, default=1e-4)
    parser.add_argument("--lambda-delta-z", type=float, default=1e-4,
                        help="L2 penalty on per-frame latent offsets (limits deformation size).")
    parser.add_argument("--lambda-plane", type=float, default=1e-4)
    parser.add_argument("--lambda-period", type=float, default=1e-5)
    parser.add_argument("--lambda-tv3d", type=float, default=1e-4,
                        help="Weight for 3D TV loss on voxelized Gaussian volume.")
    parser.add_argument("--tv3d-every", type=int, default=10,
                        help="Apply 3D TV loss every N steps (0 to disable).")
    parser.add_argument("--tv3d-res", type=int, default=64,
                        help="Voxel resolution for 3D TV regularization.")
    parser.add_argument("--decoder-hidden-dim", type=int, default=128)
    parser.add_argument("--direct-position-scale", type=float, default=0.05)
    parser.add_argument("--max-scale", type=float, default=0.01,
                        help="Hard upper bound for Gaussian scale after decoding.")
    parser.add_argument("--freeze-decoder", action="store_true")
    parser.add_argument("--decode-chunk-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=500,
                        help="Evaluate on test cameras every N steps (0 = never).")
    parser.add_argument("--log-every", type=int, default=20,
                        help="Print per-component losses every N steps.")
    parser.add_argument("--densify-every", type=int, default=100,
                        help="Run 3DGS-style densify/prune every N steps (0 to disable).")
    parser.add_argument("--densify-from", type=int, default=200,
                        help="Start densification from this step.")
    parser.add_argument("--densify-until", type=int, default=4000,
                        help="Stop densification after this step.")
    parser.add_argument("--densify-grad-thresh", type=float, default=2e-4,
                        help="Gradient threshold used to clone/split Gaussians.")
    parser.add_argument("--prune-min-opacity", type=float, default=0.01,
                        help="Prune Gaussians whose opacity falls below this value.")
    parser.add_argument("--split-scale-threshold", type=float, default=0.02,
                        help="Split (instead of clone) when mean Gaussian scale is above this threshold.")
    parser.add_argument("--opacity-reset-every", type=int, default=0,
                        help="Reset Gaussian opacity every N steps (0 to disable).")
    parser.add_argument("--opacity-reset-value", type=float, default=0.01,
                        help="Target opacity used by periodic opacity reset.")
    return parser.parse_args()


def apply_preset(args: argparse.Namespace) -> None:
    if args.preset != "paper2503":
        return

    args.densify_every = 100
    args.densify_from = 500
    args.densify_until = min(args.iterations, 15000)
    args.densify_grad_thresh = 2e-4
    args.prune_min_opacity = 0.005
    args.split_scale_threshold = 0.01
    args.opacity_reset_every = 3000
    args.opacity_reset_value = 0.01
    args.lambda_plane = max(args.lambda_plane, 1e-3)


def voxelize_frame(frame, scanner_cfg: dict, res: int) -> torch.Tensor:
    voxel_size = np.array(scanner_cfg["sVoxel"], dtype=np.float32)
    voxel_center = np.array(scanner_cfg["offOrigin"], dtype=np.float32)
    settings = GaussianVoxelizationSettings(
        scale_modifier=1.0,
        nVoxel_x=int(res),
        nVoxel_y=int(res),
        nVoxel_z=int(res),
        sVoxel_x=float(voxel_size[0]),
        sVoxel_y=float(voxel_size[1]),
        sVoxel_z=float(voxel_size[2]),
        center_x=float(voxel_center[0]),
        center_y=float(voxel_center[1]),
        center_z=float(voxel_center[2]),
        prefiltered=False,
        debug=False,
    )
    voxelizer = GaussianVoxelizer(voxel_settings=settings)
    volume, _ = voxelizer(
        means3D=frame.xyz.contiguous(),
        opacities=frame.density.contiguous(),
        scales=torch.exp(frame.scaling_logits).contiguous(),
        rotations=frame.rotations.contiguous(),
    )
    return volume


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((prediction - target) ** 2).item()
    if mse < 1e-12:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def save_checkpoint(model: SimpleFusionModel, optimizer: torch.optim.Optimizer, step: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"checkpoint_{step:06d}.pt"
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        checkpoint_path,
    )


def build_optimizer_scheduler(
    model: SimpleFusionModel,
    *,
    lr: float,
    lr_final: float,
    total_steps: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.CosineAnnealingLR]:
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=lr,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps),
        eta_min=lr_final,
    )
    return optimizer, scheduler


@torch.no_grad()
def evaluate(model: SimpleFusionModel, test_cameras: list) -> dict[str, float]:
    model.eval()
    psnr_vals = []
    for camera in test_cameras:
        frame = model(camera.time)
        render_pkg = render_projection(
            camera,
            xyz=frame.xyz,
            scaling_logits=frame.scaling_logits,
            rotations=frame.rotations,
            density=frame.density,
        )
        pred = render_pkg["render"]
        target = camera.original_image
        psnr_vals.append(psnr(pred, target))
    model.train()
    return {"psnr": sum(psnr_vals) / len(psnr_vals) if psnr_vals else float("nan")}


def main() -> None:
    args = parse_args()
    apply_preset(args)
    if not torch.cuda.is_available():
        raise RuntimeError("This simple trainer requires CUDA because the X-ray rasterizer is CUDA-only.")

    set_seed(args.seed)
    scene = load_naf_scene(args.source_path, device="cuda", eval_split=True, include_volume=False)
    xyz_np, density_np, keep_indices = load_init_point_cloud(
        args.source_path,
        args.init_path,
        max_gaussians=args.max_gaussians,
        seed=args.seed,
    )
    latent_np = load_optional_latents(args.latent_path, keep_indices=keep_indices)

    canonical_xyz = torch.from_numpy(xyz_np).cuda()
    canonical_density = torch.from_numpy(density_np).cuda()
    canonical_latent = None if latent_np is None else torch.from_numpy(latent_np).cuda()
    canonical_scaling_logits, canonical_rotations, canonical_density_logits = (
        initialize_canonical_gaussian_params(
            xyz=canonical_xyz,
            density=canonical_density,
        )
    )

    if args.latent_init == "file":
        if canonical_latent is None:
            raise ValueError("--latent-init file requires --latent-path.")
        print(f"Initialized canonical latents from file: {args.latent_path}")
    elif canonical_latent is None and args.latent_init == "encoder":
        print("Initializing canonical latents with the pretrained SF-VAE encoder.")
        canonical_latent = initialize_canonical_latent_from_encoder(
            xyz=canonical_xyz,
            density=canonical_density,
            checkpoint_path=args.decoder_checkpoint,
            embedding_dim=args.embedding_dim,
            grid_dim=args.grid_dim,
            device=canonical_xyz.device,
            num_points=args.bootstrap_num_points,
            encode_chunk_size=args.encode_chunk_size,
        )
    elif canonical_latent is not None:
        print("Initialized canonical latents from the provided latent file.")
    else:
        print("Initialized canonical latents randomly.")

    model = SimpleFusionModel(
        canonical_xyz=canonical_xyz,
        canonical_scaling_logits=canonical_scaling_logits,
        canonical_rotations=canonical_rotations,
        canonical_density_logits=canonical_density_logits,
        canonical_latent=canonical_latent,
        embedding_dim=args.embedding_dim,
        grid_dim=args.grid_dim,
        decoder_checkpoint=args.decoder_checkpoint,
        decoder_mode=args.decoder_mode,
        freeze_decoder=args.freeze_decoder,
        decode_chunk_size=args.decode_chunk_size,
        decoder_hidden_dim=args.decoder_hidden_dim,
        direct_position_scale=args.direct_position_scale,
        max_scale=args.max_scale,
    ).cuda()

    optimizer, scheduler = build_optimizer_scheduler(
        model,
        lr=args.lr,
        lr_final=args.lr_final,
        total_steps=args.iterations,
    )

    output_dir = Path(args.output_dir)
    print(
        f"Loaded {len(scene.train_cameras)} train views, "
        f"{len(scene.test_cameras)} test views, "
        f"{model.canonical_xyz.shape[0]} canonical Gaussians, "
        f"latent_init={args.latent_init}, "
        f"decoder_mode={args.decoder_mode}, "
        f"preset={args.preset}."
    )

    grad_accum = torch.zeros(model.canonical_xyz.shape[0], device=model.canonical_xyz.device)
    grad_count = torch.zeros_like(grad_accum)

    for step in range(1, args.iterations + 1):
        camera = random.choice(scene.train_cameras)
        frame = model(camera.time)
        render_pkg = render_projection(
            camera,
            xyz=frame.xyz,
            scaling_logits=frame.scaling_logits,
            rotations=frame.rotations,
            density=frame.density,
        )

        prediction = render_pkg["render"]
        target = camera.original_image

        losses: dict[str, torch.Tensor] = {}
        losses["l1"] = l1_loss(prediction, target)
        if args.lambda_ssim > 0:
            losses["ssim"] = args.lambda_ssim * (1.0 - ssim(prediction, target))
        if args.lambda_latent > 0:
            losses["latent"] = args.lambda_latent * model.latent_regularization()
        if args.lambda_delta_z > 0:
            losses["delta_z"] = args.lambda_delta_z * model.delta_z_regularization(frame)
        if args.lambda_plane > 0:
            losses["plane"] = args.lambda_plane * model.deformation_regularization()
        if args.lambda_period > 0:
            losses["period"] = args.lambda_period * model.period_regularization()
        if args.lambda_tv3d > 0 and args.tv3d_every > 0 and step % args.tv3d_every == 0:
            vol_pred = voxelize_frame(frame, scene.scanner_cfg, res=args.tv3d_res)
            losses["tv_3d"] = args.lambda_tv3d * tv_3d_loss(vol_pred)

        total_loss = sum(losses.values())

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()

        with torch.no_grad():
            viewspace_points = render_pkg["viewspace_points"]
            viewspace_grads = viewspace_points.grad
            visibility = render_pkg["visibility_filter"]
            if viewspace_grads is not None and visibility.numel() == grad_accum.numel():
                grad_norm = viewspace_grads.norm(dim=-1)
                visible_idx = torch.where(visibility)[0]
                if visible_idx.numel() > 0:
                    grad_accum[visible_idx] += grad_norm[visible_idx]
                    grad_count[visible_idx] += 1

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        scheduler.step()

        should_densify = (
            args.decoder_mode == "direct"
            and args.densify_every > 0
            and args.densify_from <= step <= args.densify_until
            and step % args.densify_every == 0
        )
        if should_densify:
            densify_stats = model.densify_and_prune(
                grad_accum=grad_accum,
                grad_count=grad_count,
                min_opacity=args.prune_min_opacity,
                grad_threshold=args.densify_grad_thresh,
                max_gaussians=args.max_gaussians,
                split_scale_threshold=args.split_scale_threshold,
            )
            if densify_stats["added"] > 0 or densify_stats["pruned"] > 0:
                current_lr = optimizer.param_groups[0]["lr"]
                remaining_steps = max(1, args.iterations - step)
                optimizer, scheduler = build_optimizer_scheduler(
                    model,
                    lr=current_lr,
                    lr_final=args.lr_final,
                    total_steps=remaining_steps,
                )
                print(
                    f"  [densify] step={step:06d} added={densify_stats['added']} "
                    f"pruned={densify_stats['pruned']} total={densify_stats['final']}"
                )

            grad_accum = torch.zeros(model.canonical_xyz.shape[0], device=model.canonical_xyz.device)
            grad_count = torch.zeros_like(grad_accum)

        should_reset_opacity = (
            args.decoder_mode == "direct"
            and args.opacity_reset_every > 0
            and step % args.opacity_reset_every == 0
        )
        if should_reset_opacity:
            model.reset_opacity(opacity_value=args.opacity_reset_value)
            print(f"  [opacity_reset] step={step:06d} value={args.opacity_reset_value:.4f}")

        if step == 1 or step % args.log_every == 0:
            parts = [f"step={step:06d}", f"loss={total_loss.item():.6f}"]
            parts += [f"{k}={v.item():.2e}" for k, v in losses.items()]
            parts.append(f"period={model.time_encoder.period.item():.4f}")
            parts.append(f"n_gauss={model.canonical_xyz.shape[0]}")
            print("  ".join(parts))

        if args.eval_every > 0 and (step % args.eval_every == 0 or step == args.iterations):
            if scene.test_cameras:
                metrics = evaluate(model, scene.test_cameras)
                print(f"  [eval] step={step:06d}  " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
            else:
                print(f"  [eval] step={step:06d}  no test cameras available")

        if step % args.save_every == 0 or step == args.iterations:
            save_checkpoint(model, optimizer, step, output_dir)


if __name__ == "__main__":
    main()
