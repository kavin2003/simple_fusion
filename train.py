from __future__ import annotations

import argparse
import math
from pathlib import Path
import random

import torch

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
from simple_fusion.loss_utils import l1_loss, ssim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple SF-VAE + X2-Gaussian fusion trainer")
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
    parser.add_argument("--decoder-hidden-dim", type=int, default=128)
    parser.add_argument("--direct-position-scale", type=float, default=0.05)
    parser.add_argument("--freeze-decoder", action="store_true")
    parser.add_argument("--decode-chunk-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=500,
                        help="Evaluate on test cameras every N steps (0 = never).")
    parser.add_argument("--log-every", type=int, default=20,
                        help="Print per-component losses every N steps.")
    return parser.parse_args()


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
    ).cuda()

    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
    )
    # Cosine annealing from lr to lr_final over all iterations.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.iterations, eta_min=args.lr_final
    )

    output_dir = Path(args.output_dir)
    print(
        f"Loaded {len(scene.train_cameras)} train views, "
        f"{len(scene.test_cameras)} test views, "
        f"{model.canonical_xyz.shape[0]} canonical Gaussians, "
        f"latent_init={args.latent_init}, "
        f"decoder_mode={args.decoder_mode}."
    )

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

        total_loss = sum(losses.values())

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if step == 1 or step % args.log_every == 0:
            parts = [f"step={step:06d}", f"loss={total_loss.item():.6f}"]
            parts += [f"{k}={v.item():.2e}" for k, v in losses.items()]
            parts.append(f"period={model.time_encoder.period.item():.4f}")
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
