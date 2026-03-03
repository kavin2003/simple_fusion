"""HexPlane regularization terms – inlined from X²-Gaussian."""
from __future__ import annotations

import torch


def compute_plane_tv(t: torch.Tensor) -> torch.Tensor:
    """Total variation on a 2-D feature plane [B, C, H, W]."""
    batch_size, c, h, w = t.shape
    count_h = batch_size * c * (h - 1) * w
    count_w = batch_size * c * h * (w - 1)
    h_tv = torch.square(t[..., 1:, :] - t[..., : h - 1, :]).sum()
    w_tv = torch.square(t[..., :, 1:] - t[..., :, : w - 1]).sum()
    return 2 * (h_tv / count_h + w_tv / count_w)


def compute_plane_smoothness(t: torch.Tensor) -> torch.Tensor:
    """Second-derivative smoothness along the first spatial axis (time) [B, C, H, W]."""
    batch_size, c, h, w = t.shape
    first_diff = t[..., 1:, :] - t[..., : h - 1, :]
    second_diff = first_diff[..., 1:, :] - first_diff[..., : h - 2, :]
    return torch.square(second_diff).mean()
