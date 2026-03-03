from __future__ import annotations

import math

import torch

from simple_fusion import _bootstrap  # noqa: F401


def _load_rasterizer():
    try:
        from xray_gaussian_rasterization_voxelization import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )
    except ImportError as exc:
        raise ImportError(
            "Failed to import xray_gaussian_rasterization_voxelization. "
            "Install/build x2-gaussian/x2_gaussian/submodules/xray-gaussian-rasterization-voxelization first."
        ) from exc
    return GaussianRasterizationSettings, GaussianRasterizer


def render_projection(
    camera,
    *,
    xyz: torch.Tensor,
    scaling_logits: torch.Tensor,
    rotations: torch.Tensor,
    density: torch.Tensor,
    scale_modifier: float = 1.0,
    debug: bool = False,
) -> dict[str, torch.Tensor]:
    GaussianRasterizationSettings, GaussianRasterizer = _load_rasterizer()
    screenspace_points = torch.zeros_like(xyz, dtype=xyz.dtype, requires_grad=True, device=xyz.device)
    try:
        screenspace_points.retain_grad()
    except RuntimeError:
        pass

    if camera.mode == 0:
        tanfovx = 1.0
        tanfovy = 1.0
    elif camera.mode == 1:
        tanfovx = math.tan(camera.FoVx * 0.5)
        tanfovy = math.tan(camera.FoVy * 0.5)
    else:
        raise ValueError(f"Unsupported scanner mode: {camera.mode}")

    raster_settings = GaussianRasterizationSettings(
        image_height=int(camera.image_height),
        image_width=int(camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        scale_modifier=scale_modifier,
        viewmatrix=camera.world_view_transform,
        projmatrix=camera.full_proj_transform,
        campos=camera.camera_center,
        prefiltered=False,
        mode=camera.mode,
        debug=debug,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    rendered_image, radii = rasterizer(
        means3D=xyz,
        means2D=screenspace_points,
        opacities=density,
        scales=torch.exp(scaling_logits),
        rotations=rotations,
        cov3D_precomp=None,
    )
    return {
        "render": rendered_image,
        "radii": radii,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
    }
