import argparse
import json
import math
import sys
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import imageio
from tqdm import tqdm

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

from xray_gaussian_rasterization_voxelization import GaussianVoxelizer, GaussianVoxelizationSettings

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="X2GS 3D PSNR & 4D Orbital Visualization")
    p.add_argument("--source-path", required=True, help="Path to .pickle dataset")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--output-dir", default="outputs/eval_case1")
    p.add_argument("--max-gaussians", type=int, default=50000)
    p.add_argument("--embedding-dim", type=int, default=32)
    p.add_argument("--grid-dim", type=int, default=12)
    p.add_argument("--video-frames", type=int, default=50)
    p.add_argument("--voxel-res", type=int, default=128, help="3D PSNR 体素分辨率")
    p.add_argument("--decoder-checkpoint", default="gs-embedding/checkpoints/checkpoint_sfvae_sh0_144.pth")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# 3D PSNR 计算逻辑
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_3d_psnr(model: SimpleFusionModel, scene, res: int = 128):
    if not hasattr(scene, 'volume') or scene.volume is None:
        print("[Warning] 数据集中未发现 3D Volume。")
        return 0.0

    print(f"正在计算 {res}^3 分辨率下的 3D PSNR...")
    device = model.canonical_xyz.device
    frame = model(0.0)

    try:
        from xray_gaussian_rasterization_voxelization import _C

        # --- 新增：自动计算包围盒 (Bounding Box) ---
        # 找出当前时刻所有高斯球的中心点范围
        mins = frame.xyz.min(dim=0)[0]
        maxs = frame.xyz.max(dim=0)[0]
        center = (mins + maxs) / 2.0
        # 计算采样步长 (sVoxel)，让网格恰好包裹住所有高斯球
        # 假设物体是均匀的，我们取三个轴向最大的跨度
        span = (maxs - mins).max().item()
        sVoxel = span / res

        print(f"[Debug] Auto-BBox Center: {center.cpu().numpy()}")
        print(f"[Debug] Auto-sVoxel: {sVoxel:.6f}")

        viewmatrix = torch.eye(4, device=device)

        # 调用底层算子，传入动态计算的中心和步长
        _, pred_volume, _, _, _, _ = _C.voxelize_gaussians(
            frame.xyz.contiguous(),
            frame.density.contiguous(),
            frame.scaling_logits.contiguous(),
            frame.rotations.contiguous(),
            1.0,
            viewmatrix.contiguous(),
            int(res), int(res), int(res),
            1.0, 1.0,  # tan_fov
            float(center[0]), float(center[1]), float(center[2]),  # 自动对齐中心
            float(sVoxel),  # 自动对齐尺度
            False,
            False
        )

        # 检查采样后的范围
        if pred_volume.max() == 0:
            print("[Warning] 采样结果依然全为0，请检查 xyz 坐标量级是否与 scene.volume 匹配")

    except Exception as e:
        print(f"[Error] 调用 Voxelizer 算子失败: {e}")
        return 0.0

    # 后续对比逻辑（修复 Tensor 类型问题）
    gt_volume = scene.volume  # 假设形状是 [T, D, H, W] 或 [D, H, W]

    # 1. 处理维度：如果是 4D (T, D, H, W)，取第一帧 t=0
    if gt_volume.ndim == 4:
        gt_volume = gt_volume[0]

    # 确保在 GPU 上且是 float
    gt_volume = gt_volume.float().to(device)

    # 2. 准备插值 (F.interpolate 要求输入是 5D: N, C, D, H, W)
    # 把 [D, H, W] 变成 [1, 1, D, H, W]
    gt_volume_5d = gt_volume.unsqueeze(0).unsqueeze(0)

    # 3. 对齐分辨率到 (res, res, res)
    gt_volume_rescaled = torch.nn.functional.interpolate(
        gt_volume_5d,
        size=(int(res), int(res), int(res)),
        mode='trilinear',
        align_corners=False
    ).squeeze()  # 变回 [res, res, res]

    # 4. 计算 PSNR
    # 确保 pred_volume 也是 [res, res, res]
    if pred_volume.ndim > 3:
        pred_volume = pred_volume.squeeze()

    mse_3d = torch.mean((pred_volume - gt_volume_rescaled) ** 2).item()

    # 打印一下范围，方便调试
    print(f"[Debug] Pred range: {pred_volume.min():.4f}-{pred_volume.max():.4f}")
    print(f"[Debug] GT range: {gt_volume_rescaled.min():.4f}-{gt_volume_rescaled.max():.4f}")

    if mse_3d < 1e-10:
        return 100.0
    psnr_3d = -10 * math.log10(mse_3d)
    return psnr_3d


# ---------------------------------------------------------------------------
# 视频渲染 (修复 NumPy .clone Bug)
# ---------------------------------------------------------------------------

@torch.no_grad()
def render_orbital_video(model, ref_cam, output_path, n_frames=50):
    print(f"渲染 4D 轨道视频: {output_path}")
    frames = []

    # 备份原始矩阵
    orig_R = np.copy(ref_cam.R)

    for i in tqdm(range(n_frames)):
        t = i / n_frames
        angle = (i / n_frames) * 2 * math.pi

        # 绕 Y 轴旋转
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        rot_y = np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ])

        # 【关键修复】：手动更新旋转并重置相机的所有派生矩阵
        ref_cam.R = orig_R @ rot_y

        # 如果你的相机类有 update 方法，必须调用
        if hasattr(ref_cam, 'update'):
            ref_cam.update()
            # 如果是 SimpleGS 框架，可能需要手动更新这个：
        if hasattr(ref_cam, 'world_view_transform'):
            from simple_fusion.rendering import get_view_matrix  # 假设有这个函数
            # 此处逻辑取决于你的相机类实现，通常修改 R 后需要重新生成 transform

        frame_data = model(t)
        render_pkg = render_projection(
            ref_cam,
            xyz=frame_data.xyz,
            scaling_logits=frame_data.scaling_logits,
            rotations=frame_data.rotations,
            density=frame_data.density
        )

        img = render_pkg["render"].squeeze().cpu().numpy()
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        frames.append(img)

    imageio.mimsave(output_path, frames, fps=20)
    ref_cam.R = orig_R


# ---------------------------------------------------------------------------
# Main 流程
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1: 加载场景 (包含 3D Volume)...")
    scene = load_naf_scene(args.source_path, device="cuda", eval_split=True, include_volume=True)

    print(f"Step 2: 构建模型 (点数: {args.max_gaussians})...")
    xyz_np, density_np, keep_indices = load_init_point_cloud(
        args.source_path, None, max_gaussians=args.max_gaussians, seed=args.seed
    )

    # 初始化模型，确保结构与训练时完全一致
    model = SimpleFusionModel(
        canonical_xyz=torch.from_numpy(xyz_np).cuda(),
        canonical_scaling_logits=torch.zeros((len(xyz_np), 3)).cuda(),
        canonical_rotations=torch.zeros((len(xyz_np), 4)).cuda(),
        canonical_density_logits=torch.from_numpy(density_np).cuda(),
        canonical_latent=torch.randn((len(xyz_np), args.embedding_dim)).cuda(),
        embedding_dim=args.embedding_dim,
        grid_dim=args.grid_dim,
        decoder_checkpoint=args.decoder_checkpoint,
        freeze_decoder=True
    ).cuda()

    print(f"加载 Checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cuda")
    model.load_state_dict(ckpt["model"])
    model.eval()

    print("\nStep 3: 正在评估测试集 2D 指标...")
    mse_list = []
    for cam in tqdm(scene.test_cameras):
        frame = model(cam.time)
        res = render_projection(cam, xyz=frame.xyz, scaling_logits=frame.scaling_logits,
                                rotations=frame.rotations, density=frame.density)["render"]
        mse_list.append(torch.mean((res - cam.original_image) ** 2).item())

    avg_2d_psnr = -10 * math.log10(np.mean(mse_list))
    print(f"  Test Set 平均 PSNR: {avg_2d_psnr:.4f}")

    print("\nStep 4: 正在评估 3D 空间指标...")
    psnr_3d = compute_3d_psnr(model, scene, res=args.voxel_res)
    print(f"  3D Volumetric PSNR: {psnr_3d:.4f}")

    print("\nStep 5: 正在渲染多视角呼吸动图...")
    if scene.test_cameras:
        render_orbital_video(model, scene.test_cameras[0], output_dir / "lung_4d_orbital.mp4",
                             n_frames=args.video_frames)

    # 汇总保存
    results = {
        "2d_psnr": avg_2d_psnr,
        "3d_psnr": psnr_3d,
        "max_gaussians": args.max_gaussians,
        "learned_period": float(model.time_encoder.period.item())
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n[Done] 所有评估结果已保存在: {output_dir}")


if __name__ == "__main__":
    main()