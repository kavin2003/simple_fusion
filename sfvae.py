"""SF-VAE (Scaffold-Free VAE) – inlined from gs-embedding repo.

Combines:
  gs-embedding/model/sf_model.py        → ChamferLoss, SFEncoder, SFDecoder, SFModel
  gs-embedding/embedding_model/embedding_model.py → SFVAE
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Chamfer loss
# ---------------------------------------------------------------------------

class ChamferLoss(nn.Module):
    def __init__(self, geo_weight: float = 0.5):
        super().__init__()
        self.geo_weight = geo_weight

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return self._chamfer_with_color(x, y, alpha=self.geo_weight)

    def _chamfer_with_color(self, x: torch.Tensor, y: torch.Tensor, alpha: float = 0.5):
        coords_x, color_x = x[:, :, :3], x[:, :, 3:]
        coords_y, color_y = y[:, :, :3], y[:, :, 3:]

        norm = torch.norm(coords_x, dim=-1).max(dim=-1, keepdim=True)[0].unsqueeze(2)

        cx = coords_x.unsqueeze(2)
        cy = coords_y.unsqueeze(1)
        geo_dist = torch.sqrt(torch.sum((cx - cy) ** 2, dim=-1)) / norm

        min_geo_x2y, idx_x2y = torch.min(geo_dist, dim=2)
        min_geo_y2x, idx_y2x = torch.min(geo_dist, dim=1)

        color_y_near = torch.gather(color_y, 1, idx_x2y.unsqueeze(-1).expand(-1, -1, color_y.size(-1)))
        color_x_near = torch.gather(color_x, 1, idx_y2x.unsqueeze(-1).expand(-1, -1, color_x.size(-1)))
        color_dist = (
            torch.sqrt(torch.sum((color_x - color_y_near) ** 2, dim=2)).mean()
            + torch.sqrt(torch.sum((color_y - color_x_near) ** 2, dim=2)).mean()
        )
        loss_color = color_dist

        loss_geo = alpha * min_geo_x2y.mean() + alpha * min_geo_y2x.mean()
        total = loss_geo + (1 - alpha) * loss_color
        return total, loss_geo, (1 - alpha) * loss_color


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class SFEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 7,
        feature_dim: int = 32,
        vae: bool = True,
        deterministic: bool = False,
    ):
        super().__init__()
        self.vae = vae
        self.deterministic = deterministic

        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, feature_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(feature_dim)

        if vae:
            self.fc_mu = nn.Linear(feature_dim, feature_dim)
            self.fc_logvar = nn.Linear(feature_dim, feature_dim)
        else:
            self.fc1 = nn.Linear(feature_dim, feature_dim)
            self.fc2 = nn.Linear(feature_dim, feature_dim)

    def forward(self, x: torch.Tensor):
        # x: [B, N, input_dim]
        x = x.transpose(2, 1)  # [B, input_dim, N]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = torch.max(x, 2)[0]  # [B, feature_dim]

        if self.vae:
            mu = self.fc_mu(x)
            logvar = torch.clamp(self.fc_logvar(x), min=-6, max=3)
            if self.deterministic:
                z = mu
            else:
                std = torch.exp(0.5 * logvar) + 1e-8
                z = mu + std * torch.randn_like(std)
            return z, mu, logvar
        else:
            x = F.relu(self.fc1(x))
            return self.fc2(x)


# ---------------------------------------------------------------------------
# Decoder (folding network)
# ---------------------------------------------------------------------------

class SFDecoder(nn.Module):
    def __init__(self, grid_dim: int = 12, out_dim: int = 7, feature_dim: int = 32):
        super().__init__()
        self.grid_dim = grid_dim

        u, v = torch.meshgrid(
            torch.linspace(0, 2 * torch.pi, grid_dim),
            torch.linspace(0, torch.pi, grid_dim),
            indexing="ij",
        )
        x = torch.sin(v) * torch.cos(u) * 0.1
        y = torch.sin(v) * torch.sin(u) * 0.1
        z = torch.cos(v) * 0.1
        grid = torch.stack([x, y, z], dim=-1).view(-1, 3)
        self.register_buffer("grid", grid)

        self.fc1 = nn.Linear(3 + feature_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 3)

        self.fc1_c = nn.Linear(3 + feature_dim, 512)
        self.fc2_c = nn.Linear(512, 512)
        self.fc3_c = nn.Linear(512, out_dim - 3)

    def forward(self, global_feature: torch.Tensor) -> torch.Tensor:
        B = global_feature.size(0)
        grid = self.grid.unsqueeze(0).expand(B, -1, -1)  # [B, grid_dim², 3]
        gf = global_feature.unsqueeze(1).expand(-1, self.grid_dim**2, -1)
        x = torch.cat([grid, gf], dim=2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # [B, grid_dim², 3]

        color = torch.cat([x.detach(), gf], dim=2)
        color = F.relu(self.fc1_c(color))
        color = F.relu(self.fc2_c(color))
        color = self.fc3_c(color)

        return torch.cat([x, color], dim=2)  # [B, grid_dim², out_dim]


# ---------------------------------------------------------------------------
# SFModel
# ---------------------------------------------------------------------------

class SFModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 7,
        feature_dim: int = 32,
        grid_dim: int = 12,
        vae: bool = True,
        deterministic: bool = False,
    ):
        super().__init__()
        self.vae = vae
        self.encoder = SFEncoder(input_dim=input_dim, feature_dim=feature_dim, vae=vae, deterministic=deterministic)
        self.decoder = SFDecoder(grid_dim=grid_dim, out_dim=input_dim, feature_dim=feature_dim)

    def forward(self, x: torch.Tensor):
        if self.vae:
            z, mu, logvar = self.encoder(x)
            return self.decoder(z), z, mu, logvar
        else:
            feat = self.encoder(x)
            return self.decoder(feat), feat


# ---------------------------------------------------------------------------
# SFVAE (public API used by model.py)
# ---------------------------------------------------------------------------

class SFVAE(nn.Module):
    """Scaffold-Free VAE with encode() / decode() interface."""

    def __init__(
        self,
        embedding_dim: int = 32,
        grid_dim: int = 12,
        vae: bool = True,
        norm_weight: float = 0.0,
        deterministic: bool = False,
    ):
        super().__init__()
        self.is_vae = vae
        self.model = SFModel(
            input_dim=7,
            feature_dim=embedding_dim,
            grid_dim=grid_dim,
            vae=vae,
            deterministic=deterministic,
        )
        self.criterion = ChamferLoss(geo_weight=0.5)
        self.norm_weight = norm_weight

    def forward(self, data: torch.Tensor):
        if self.is_vae:
            output, z, mu, logvar = self.model(data)
            chamfer_loss, _, _ = self.criterion(data, output)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / data.size(0)
            return chamfer_loss + self.norm_weight * kl_loss, output, z
        else:
            output, emb = self.model(data)
            loss_geo, loss_color, _ = self.criterion(data, output)
            return loss_geo, loss_color, output, emb

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_vae:
            z, _, _ = self.model.encoder(x)
        else:
            z = self.model.encoder(x)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.model.decoder(z)
