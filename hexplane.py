"""HexPlane spatiotemporal feature-plane module – inlined from X²-Gaussian."""
from __future__ import annotations

import itertools
from typing import Collection, Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_aabb(pts: torch.Tensor, aabb: torch.Tensor) -> torch.Tensor:
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0


def grid_sample_wrapper(
    grid: torch.Tensor,
    coords: torch.Tensor,
    align_corners: bool = True,
) -> torch.Tensor:
    grid_dim = coords.shape[-1]
    if grid.dim() == grid_dim + 1:
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)
    if grid_dim not in (2, 3):
        raise NotImplementedError(f"grid_sample_wrapper: unsupported dim {grid_dim}")
    coords = coords.view(
        [coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:])
    )
    B, feature_dim = grid.shape[:2]
    n = coords.shape[-2]
    interp = F.grid_sample(
        grid, coords, align_corners=align_corners, mode="bilinear", padding_mode="border"
    )
    interp = interp.view(B, feature_dim, n).transpose(-1, -2)
    return interp.squeeze()


def init_grid_param(
    grid_nd: int,
    in_dim: int,
    out_dim: int,
    reso: Sequence[int],
    a: float = 0.1,
    b: float = 0.5,
) -> nn.ParameterList:
    assert in_dim == len(reso)
    has_time_planes = in_dim == 4
    coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
    grid_coefs = nn.ParameterList()
    for coo_comb in coo_combs:
        coef = nn.Parameter(
            torch.empty([1, out_dim] + [reso[cc] for cc in coo_comb[::-1]])
        )
        if has_time_planes and 3 in coo_comb:
            nn.init.ones_(coef)
        else:
            nn.init.uniform_(coef, a=a, b=b)
        grid_coefs.append(coef)
    return grid_coefs


def interpolate_ms_features(
    pts: torch.Tensor,
    ms_grids: Collection[Iterable[nn.Module]],
    grid_dimensions: int,
    concat_features: bool,
    num_levels: Optional[int],
) -> torch.Tensor:
    coo_combs = list(itertools.combinations(range(pts.shape[-1]), grid_dimensions))
    if num_levels is None:
        num_levels = len(ms_grids)
    multi_scale_interp = [] if concat_features else 0.0
    for scale_id, grid in enumerate(list(ms_grids)[:num_levels]):
        interp_space = 1.0
        for ci, coo_comb in enumerate(coo_combs):
            feature_dim = grid[ci].shape[1]
            interp_out_plane = (
                grid_sample_wrapper(grid[ci], pts[..., coo_comb]).view(-1, feature_dim)
            )
            interp_space = interp_space * interp_out_plane
        if concat_features:
            multi_scale_interp.append(interp_space)
        else:
            multi_scale_interp = multi_scale_interp + interp_space
    if concat_features:
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
    return multi_scale_interp


class HexPlaneField(nn.Module):
    def __init__(self, bounds: float, planeconfig: dict, multires: list) -> None:
        super().__init__()
        aabb = torch.tensor([[bounds, bounds, bounds], [-bounds, -bounds, -bounds]])
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.grid_config = [planeconfig]
        self.multiscale_res_multipliers = multires
        self.concat_features = True

        self.grids = nn.ModuleList()
        self.feat_dim = 0
        for res in self.multiscale_res_multipliers:
            config = self.grid_config[0].copy()
            config["resolution"] = (
                [r * res for r in config["resolution"][:3]] + config["resolution"][3:]
            )
            gp = init_grid_param(
                grid_nd=config["grid_dimensions"],
                in_dim=config["input_coordinate_dim"],
                out_dim=config["output_coordinate_dim"],
                reso=config["resolution"],
            )
            if self.concat_features:
                self.feat_dim += gp[-1].shape[1]
            else:
                self.feat_dim = gp[-1].shape[1]
            self.grids.append(gp)

    @property
    def get_aabb(self):
        return self.aabb[0], self.aabb[1]

    def get_density(
        self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        pts = normalize_aabb(pts, self.aabb)
        pts = torch.cat((pts, timestamps), dim=-1)
        pts = pts.reshape(-1, pts.shape[-1])
        features = interpolate_ms_features(
            pts,
            ms_grids=self.grids,
            grid_dimensions=self.grid_config[0]["grid_dimensions"],
            concat_features=self.concat_features,
            num_levels=None,
        )
        if len(features) < 1:
            features = torch.zeros((0, 1), device=features.device)
        return features

    def forward(
        self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.get_density(pts, timestamps)
