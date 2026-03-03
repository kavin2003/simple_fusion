from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from simple_fusion.sfvae import SFVAE
from simple_fusion.gs_utils import point2gaussian_torch_batched
from simple_fusion.hexplane import HexPlaneField
from simple_fusion.regulation import compute_plane_smoothness, compute_plane_tv


@dataclass
class GaussianFrame:
    xyz: torch.Tensor
    scaling_logits: torch.Tensor
    rotations: torch.Tensor
    density: torch.Tensor
    latent: torch.Tensor
    delta_z: torch.Tensor
    raw_gaussian_params: torch.Tensor


class PeriodicTimeEncoder(nn.Module):
    def __init__(
        self,
        initial_period: float = 1.0,
        initial_warp: float = 16.0,
        learnable_period: bool = True,
    ) -> None:
        super().__init__()
        period_value = torch.tensor(float(initial_period)).log()
        warp_value = torch.tensor(float(initial_warp)).log()
        if learnable_period:
            self.log_period = nn.Parameter(period_value)
        else:
            self.register_buffer("log_period", period_value)
        self.register_buffer("log_warp", warp_value)

    @property
    def period(self) -> torch.Tensor:
        return self.log_period.exp().clamp_min(1e-4)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        time = time.float()
        if time.ndim == 0:
            time = time[None]
        if time.ndim == 1:
            time = time[:, None]

        period = self.period.to(time.device)
        cycle_pos = torch.remainder(time, period) / period
        warp = self.log_warp.exp().to(time.device)
        tau = torch.log1p(cycle_pos * warp) / torch.log1p(warp)
        return tau

    def regularization(self, target_period: float = 1.0) -> torch.Tensor:
        target = torch.tensor(float(target_period), device=self.log_period.device)
        return (self.period - target).pow(2).mean()


class LatentDeformationField(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        bounds: float,
        kplanes_config: dict,
        multires: list[int],
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.grid = HexPlaneField(bounds, kplanes_config, multires)
        self.delta_mlp = nn.Sequential(
            nn.Linear(self.grid.feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, xyz: torch.Tensor, time_feature: torch.Tensor) -> torch.Tensor:
        grid_feature = self.grid(xyz, time_feature)
        return self.delta_mlp(grid_feature)

    def regularization(self) -> torch.Tensor:
        # Spatial planes (xy=0, xz=1, yz=3): total-variation encourages spatial smoothness.
        # Temporal planes (xt=2, yt=4, zt=5): second-derivative smoothness + L1 identity
        # (initialized to 1) discourages spurious temporal drift.
        device = self.delta_mlp[0].weight.device
        spatial_tv = torch.tensor(0.0, device=device)
        temporal_smoothness = torch.tensor(0.0, device=device)
        temporal_identity = torch.tensor(0.0, device=device)
        for grids in self.grid.grids:
            if len(grids) < 6:
                continue
            for grid_id in (0, 1, 3):  # spatial planes
                spatial_tv = spatial_tv + compute_plane_tv(grids[grid_id])
            for grid_id in (2, 4, 5):  # temporal planes
                temporal_smoothness = temporal_smoothness + compute_plane_smoothness(grids[grid_id])
                temporal_identity = temporal_identity + torch.abs(1 - grids[grid_id]).mean()
        return spatial_tv + temporal_smoothness + temporal_identity


class PointDecoderAdapter(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        grid_dim: int,
        checkpoint_path: Optional[str] = None,
        freeze_decoder: bool = True,
        decode_chunk_size: int = 2048,
    ) -> None:
        super().__init__()
        self.decode_chunk_size = decode_chunk_size
        self.sfvae = SFVAE(
            embedding_dim=embedding_dim,
            grid_dim=grid_dim,
            deterministic=True,
        )

        if checkpoint_path is not None:
            loaded = _load_svae_checkpoint(
                checkpoint_path,
                embedding_dim=embedding_dim,
                grid_dim=grid_dim,
                device="cpu",
            )
            self.sfvae.load_state_dict(loaded.state_dict(), strict=True)

        if freeze_decoder:
            for parameter in self.sfvae.parameters():
                parameter.requires_grad_(False)
            self.sfvae.eval()

    def forward(self, latent: torch.Tensor, xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = []
        for start in range(0, latent.shape[0], self.decode_chunk_size):
            end = start + self.decode_chunk_size
            latent_chunk = latent[start:end]
            points = self.sfvae.decode(latent_chunk)
            gaussian_params = point2gaussian_torch_batched(points)
            outputs.append(gaussian_params)

        gaussian_params = torch.cat(outputs, dim=0)
        gaussian_params[:, :3] = xyz

        scaling_logits = gaussian_params[:, 3:6]
        rotations = F.normalize(gaussian_params[:, 6:10], dim=-1)
        density = torch.sigmoid(gaussian_params[:, 10:11])
        return gaussian_params, torch.cat([scaling_logits, rotations, density], dim=-1)


class DirectGaussianDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 128,
        position_scale: float = 0.05,
    ) -> None:
        super().__init__()
        self.position_scale = float(position_scale)
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.xyz_head = nn.Linear(hidden_dim, 3)
        self.scaling_head = nn.Linear(hidden_dim, 3)
        self.rotation_head = nn.Linear(hidden_dim, 4)
        self.density_head = nn.Linear(hidden_dim, 1)
        self._zero_init_heads()

    def _zero_init_heads(self) -> None:
        for layer in (self.xyz_head, self.scaling_head, self.rotation_head, self.density_head):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(
        self,
        latent: torch.Tensor,
        *,
        canonical_xyz: torch.Tensor,
        canonical_scaling_logits: torch.Tensor,
        canonical_rotations: torch.Tensor,
        canonical_density_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(latent)
        delta_xyz = torch.tanh(self.xyz_head(hidden)) * self.position_scale
        delta_scaling = self.scaling_head(hidden)
        delta_rotation = self.rotation_head(hidden)
        delta_density = self.density_head(hidden)

        xyz = canonical_xyz + delta_xyz
        scaling_logits = canonical_scaling_logits + delta_scaling
        rotations = F.normalize(canonical_rotations + delta_rotation, dim=-1)
        density_logits = canonical_density_logits + delta_density
        density = torch.sigmoid(density_logits)

        raw_gaussian_params = torch.cat(
            [xyz, scaling_logits, rotations, density_logits],
            dim=-1,
        )
        compact = torch.cat([scaling_logits, rotations, density], dim=-1)
        return raw_gaussian_params, compact


def _load_svae_checkpoint(
    checkpoint_path: str,
    *,
    embedding_dim: int,
    grid_dim: int,
    device: str | torch.device,
) -> SFVAE:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            cleaned_state_dict[key[len("_orig_mod."):]] = value
        else:
            cleaned_state_dict[key] = value

    sfvae = SFVAE(
        embedding_dim=embedding_dim,
        grid_dim=grid_dim,
        deterministic=True,
    )
    sfvae.load_state_dict(cleaned_state_dict, strict=True)
    sfvae.eval()
    sfvae.to(device)
    for parameter in sfvae.parameters():
        parameter.requires_grad_(False)
    return sfvae


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.clamp(min=eps, max=1.0 - eps)
    return torch.log(x / (1.0 - x))


@torch.no_grad()
def estimate_isotropic_scales(
    xyz: torch.Tensor,
    *,
    k: int = 4,
    chunk_size: int = 1024,
    min_scale: float = 1e-3,
    max_scale: float = 0.25,
) -> torch.Tensor:
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape [N, 3].")
    if xyz.shape[0] < 2:
        return torch.full((xyz.shape[0], 3), min_scale, dtype=xyz.dtype, device=xyz.device)

    num_points = xyz.shape[0]
    k_eff = min(k + 1, num_points)
    scales = []
    for start in range(0, num_points, chunk_size):
        end = min(start + chunk_size, num_points)
        dist = torch.cdist(xyz[start:end], xyz)
        knn_dist, _ = torch.topk(dist, k=k_eff, dim=1, largest=False)
        knn_dist = knn_dist[:, 1:] if k_eff > 1 else knn_dist
        local_scale = knn_dist.mean(dim=1, keepdim=True)
        scales.append(local_scale)

    scales = torch.cat(scales, dim=0).clamp(min=min_scale, max=max_scale)
    return scales.repeat(1, 3)


def build_bootstrap_point_clouds(
    scales: torch.Tensor,
    density: torch.Tensor,
    *,
    num_points: int = 144,
) -> torch.Tensor:
    if scales.ndim != 2 or scales.shape[1] != 3:
        raise ValueError("scales must have shape [N, 3].")
    if density.ndim != 2 or density.shape[1] != 1:
        raise ValueError("density must have shape [N, 1].")

    grid_side = int(round(num_points**0.5))
    if grid_side * grid_side != num_points:
        raise ValueError("num_points must be a perfect square.")

    device = scales.device
    dtype = scales.dtype
    u, v = torch.meshgrid(
        torch.linspace(0, 2 * torch.pi, grid_side, device=device, dtype=dtype),
        torch.linspace(0, torch.pi, grid_side, device=device, dtype=dtype),
        indexing="ij",
    )
    sphere = torch.stack(
        [
            torch.sin(v) * torch.cos(u),
            torch.sin(v) * torch.sin(u),
            torch.cos(v),
        ],
        dim=-1,
    ).reshape(1, num_points, 3)

    points = sphere * scales[:, None, :]
    colors = torch.zeros(points.shape[0], num_points, 3, device=device, dtype=dtype)
    opacity = density.clamp(1e-4, 1.0).expand(-1, num_points).unsqueeze(-1)
    return torch.cat([points, colors, opacity], dim=-1)


@torch.no_grad()
def initialize_canonical_gaussian_params(
    *,
    xyz: torch.Tensor,
    density: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scales = estimate_isotropic_scales(xyz)
    scaling_logits = torch.log(scales.clamp_min(1e-6))

    rotations = torch.zeros(xyz.shape[0], 4, dtype=xyz.dtype, device=xyz.device)
    rotations[:, 0] = 1.0

    density_logits = inverse_sigmoid(density)
    return scaling_logits, rotations, density_logits


@torch.no_grad()
def initialize_canonical_latent_from_encoder(
    *,
    xyz: torch.Tensor,
    density: torch.Tensor,
    checkpoint_path: str,
    embedding_dim: int,
    grid_dim: int,
    device: str | torch.device,
    num_points: int = 144,
    encode_chunk_size: int = 1024,
) -> torch.Tensor:
    sfvae = _load_svae_checkpoint(
        checkpoint_path,
        embedding_dim=embedding_dim,
        grid_dim=grid_dim,
        device=device,
    )
    scales = estimate_isotropic_scales(xyz)
    bootstrap_points = build_bootstrap_point_clouds(scales, density, num_points=num_points)

    latents = []
    for start in range(0, bootstrap_points.shape[0], encode_chunk_size):
        end = min(start + encode_chunk_size, bootstrap_points.shape[0])
        latents.append(sfvae.encode(bootstrap_points[start:end]))
    return torch.cat(latents, dim=0)


class SimpleFusionModel(nn.Module):
    def __init__(
        self,
        canonical_xyz: torch.Tensor,
        *,
        canonical_scaling_logits: Optional[torch.Tensor] = None,
        canonical_rotations: Optional[torch.Tensor] = None,
        canonical_density_logits: Optional[torch.Tensor] = None,
        embedding_dim: int = 32,
        grid_dim: int = 12,
        canonical_latent: Optional[torch.Tensor] = None,
        decoder_checkpoint: Optional[str] = None,
        decoder_mode: str = "direct",
        freeze_decoder: bool = True,
        decode_chunk_size: int = 2048,
        bounds: float = 1.6,
        kplanes_config: Optional[dict] = None,
        multires: Optional[list[int]] = None,
        deformation_hidden_dim: int = 64,
        decoder_hidden_dim: int = 128,
        direct_position_scale: float = 0.05,
        initial_period: float = 1.0,
    ) -> None:
        super().__init__()
        if canonical_xyz.ndim != 2 or canonical_xyz.shape[1] != 3:
            raise ValueError("canonical_xyz must have shape [N, 3].")
        if decoder_mode not in {"direct", "point"}:
            raise ValueError("decoder_mode must be either 'direct' or 'point'.")

        num_gaussians = canonical_xyz.shape[0]
        if canonical_latent is None:
            canonical_latent = torch.randn(num_gaussians, embedding_dim) * 0.01
        if canonical_latent.shape != (num_gaussians, embedding_dim):
            raise ValueError(
                f"canonical_latent must have shape [{num_gaussians}, {embedding_dim}]"
            )

        if kplanes_config is None:
            kplanes_config = {
                "grid_dimensions": 2,
                "input_coordinate_dim": 4,
                "output_coordinate_dim": 32,
                "resolution": [64, 64, 64, 150],
            }
        if multires is None:
            multires = [1, 2, 4, 8]

        self.decoder_mode = decoder_mode
        self.embedding_dim = embedding_dim
        self.canonical_xyz = nn.Parameter(canonical_xyz.float())
        self.canonical_latent = nn.Parameter(canonical_latent.float())

        if self.decoder_mode == "direct":
            if canonical_scaling_logits is None or canonical_rotations is None or canonical_density_logits is None:
                raise ValueError(
                    "Direct decoder mode requires canonical scaling, rotation, and density initialization."
                )
            self.canonical_scaling_logits = nn.Parameter(canonical_scaling_logits.float())
            self.canonical_rotations = nn.Parameter(canonical_rotations.float())
            self.canonical_density_logits = nn.Parameter(canonical_density_logits.float())
        else:
            self.register_parameter("canonical_scaling_logits", None)
            self.register_parameter("canonical_rotations", None)
            self.register_parameter("canonical_density_logits", None)

        self.time_encoder = PeriodicTimeEncoder(initial_period=initial_period)
        self.deformation_field = LatentDeformationField(
            latent_dim=embedding_dim,
            bounds=bounds,
            kplanes_config=kplanes_config,
            multires=multires,
            hidden_dim=deformation_hidden_dim,
        )
        if self.decoder_mode == "direct":
            self.decoder = DirectGaussianDecoder(
                latent_dim=embedding_dim,
                hidden_dim=decoder_hidden_dim,
                position_scale=direct_position_scale,
            )
        else:
            self.decoder = PointDecoderAdapter(
                embedding_dim=embedding_dim,
                grid_dim=grid_dim,
                checkpoint_path=decoder_checkpoint,
                freeze_decoder=freeze_decoder,
                decode_chunk_size=decode_chunk_size,
            )

    def _expand_time(self, time: torch.Tensor | float, num_gaussians: int) -> torch.Tensor:
        if not torch.is_tensor(time):
            time = torch.tensor(float(time), device=self.canonical_xyz.device)
        time = time.to(self.canonical_xyz.device)
        if time.ndim == 0:
            time = time.repeat(num_gaussians)
        elif time.ndim == 1 and time.numel() == 1:
            time = time.repeat(num_gaussians)
        elif time.ndim == 1 and time.numel() != num_gaussians:
            raise ValueError("Time tensor must be scalar or have one entry per Gaussian.")
        elif time.ndim > 1:
            raise ValueError("Time tensor must be scalar or 1D.")
        return time[:, None]

    def forward(self, time: torch.Tensor | float) -> GaussianFrame:
        num_gaussians = self.canonical_xyz.shape[0]
        time_values = self._expand_time(time, num_gaussians)
        time_feature = self.time_encoder(time_values)
        delta_z = self.deformation_field(self.canonical_xyz, time_feature)
        latent = self.canonical_latent + delta_z
        if self.decoder_mode == "direct":
            raw_gaussian_params, compact_params = self.decoder(
                latent,
                canonical_xyz=self.canonical_xyz,
                canonical_scaling_logits=self.canonical_scaling_logits,
                canonical_rotations=F.normalize(self.canonical_rotations, dim=-1),
                canonical_density_logits=self.canonical_density_logits,
            )
            xyz = raw_gaussian_params[:, :3]
        else:
            raw_gaussian_params, compact_params = self.decoder(latent, self.canonical_xyz)
            xyz = self.canonical_xyz

        return GaussianFrame(
            xyz=xyz,
            scaling_logits=compact_params[:, :3],
            rotations=compact_params[:, 3:7],
            density=compact_params[:, 7:8],
            latent=latent,
            delta_z=delta_z,
            raw_gaussian_params=raw_gaussian_params,
        )

    def latent_regularization(self) -> torch.Tensor:
        """L2 penalty on the canonical latent codes to keep them on the VAE manifold."""
        return self.canonical_latent.pow(2).mean()

    def delta_z_regularization(self, frame: GaussianFrame) -> torch.Tensor:
        """L2 penalty on per-frame latent offsets to limit deformation magnitude."""
        return frame.delta_z.pow(2).mean()

    def deformation_regularization(self) -> torch.Tensor:
        return self.deformation_field.regularization()

    def period_regularization(self, target_period: float = 1.0) -> torch.Tensor:
        return self.time_encoder.regularization(target_period=target_period)

    @torch.no_grad()
    def reset_opacity(self, opacity_value: float = 0.01) -> None:
        if self.decoder_mode != "direct":
            return
        target = torch.full_like(self.canonical_density_logits, float(opacity_value))
        self.canonical_density_logits.copy_(inverse_sigmoid(target))

    @torch.no_grad()
    def densify_and_prune(
        self,
        *,
        grad_accum: torch.Tensor,
        grad_count: torch.Tensor,
        min_opacity: float,
        grad_threshold: float,
        max_gaussians: int,
        split_scale_threshold: float = 0.02,
    ) -> dict[str, int]:
        if self.decoder_mode != "direct":
            return {"added": 0, "pruned": 0, "final": int(self.canonical_xyz.shape[0])}

        device = self.canonical_xyz.device
        n = self.canonical_xyz.shape[0]
        if n == 0:
            return {"added": 0, "pruned": 0, "final": 0}

        grad_mean = grad_accum / grad_count.clamp_min(1.0)
        opacity = torch.sigmoid(self.canonical_density_logits).squeeze(-1)
        scale_world = torch.exp(self.canonical_scaling_logits).mean(dim=-1)

        grad_trigger = grad_mean >= grad_threshold
        split_mask = grad_trigger & (scale_world >= split_scale_threshold)
        clone_mask = grad_trigger & (~split_mask)

        # Keep growth bounded by max_gaussians.
        free_slots = max(0, int(max_gaussians) - int(n))
        split_idx = torch.where(split_mask)[0]
        clone_idx = torch.where(clone_mask)[0]

        # A split creates one extra Gaussian when parent is replaced by two children.
        num_split = min(int(split_idx.numel()), free_slots)
        free_slots -= num_split
        num_clone = min(int(clone_idx.numel()), free_slots)
        split_idx = split_idx[:num_split]
        clone_idx = clone_idx[:num_clone]

        new_xyz = []
        new_scaling = []
        new_rotation = []
        new_density = []
        new_latent = []

        if num_split > 0:
            base_xyz = self.canonical_xyz[split_idx]
            base_scale = torch.exp(self.canonical_scaling_logits[split_idx])
            jitter = torch.randn_like(base_xyz) * (0.5 * base_scale)

            child_xyz_1 = base_xyz + jitter
            child_xyz_2 = base_xyz - jitter
            child_scale = (base_scale / 1.6).clamp_min(1e-4)
            child_scaling_logits = torch.log(child_scale)
            child_rotation = self.canonical_rotations[split_idx]
            child_density = self.canonical_density_logits[split_idx] - 0.15
            child_latent = self.canonical_latent[split_idx]

            new_xyz += [child_xyz_1, child_xyz_2]
            new_scaling += [child_scaling_logits, child_scaling_logits]
            new_rotation += [child_rotation, child_rotation]
            new_density += [child_density, child_density]
            new_latent += [child_latent, child_latent]

        if num_clone > 0:
            clone_xyz = self.canonical_xyz[clone_idx]
            clone_scaling = self.canonical_scaling_logits[clone_idx]
            clone_rotation = self.canonical_rotations[clone_idx]
            clone_density = self.canonical_density_logits[clone_idx] - 0.05
            clone_latent = self.canonical_latent[clone_idx]

            new_xyz.append(clone_xyz)
            new_scaling.append(clone_scaling)
            new_rotation.append(clone_rotation)
            new_density.append(clone_density)
            new_latent.append(clone_latent)

        keep_mask = opacity >= float(min_opacity)
        if num_split > 0:
            keep_mask[split_idx] = False

        xyz = self.canonical_xyz[keep_mask]
        scaling_logits = self.canonical_scaling_logits[keep_mask]
        rotations = self.canonical_rotations[keep_mask]
        density_logits = self.canonical_density_logits[keep_mask]
        latent = self.canonical_latent[keep_mask]

        if new_xyz:
            xyz = torch.cat([xyz] + new_xyz, dim=0)
            scaling_logits = torch.cat([scaling_logits] + new_scaling, dim=0)
            rotations = torch.cat([rotations] + new_rotation, dim=0)
            density_logits = torch.cat([density_logits] + new_density, dim=0)
            latent = torch.cat([latent] + new_latent, dim=0)

        self.canonical_xyz = nn.Parameter(xyz.to(device))
        self.canonical_scaling_logits = nn.Parameter(scaling_logits.to(device))
        self.canonical_rotations = nn.Parameter(rotations.to(device))
        self.canonical_density_logits = nn.Parameter(density_logits.to(device))
        self.canonical_latent = nn.Parameter(latent.to(device))

        final_n = int(self.canonical_xyz.shape[0])
        added = int(final_n - keep_mask.sum().item())
        pruned = int((~keep_mask).sum().item())
        return {"added": added, "pruned": pruned, "final": final_n}
