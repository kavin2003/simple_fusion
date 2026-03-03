"""Loss functions and image metrics – inlined from X²-Gaussian."""
from __future__ import annotations

from math import exp

import torch
import torch.nn.functional as F
from torch.autograd import Variable


# ---------------------------------------------------------------------------
# Basic losses
# ---------------------------------------------------------------------------

def l1_loss(network_output: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return torch.abs(network_output - gt).mean()


def l2_loss(network_output: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return ((network_output - gt) ** 2).mean()


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------

def _gaussian_window(window_size: int, sigma: float = 1.5) -> torch.Tensor:
    gauss = torch.tensor(
        [exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2)) for x in range(window_size)]
    )
    return gauss / gauss.sum()


def _create_window(window_size: int, channel: int) -> torch.Tensor:
    _1d = _gaussian_window(window_size).unsqueeze(1)
    _2d = _1d.mm(_1d.t()).float().unsqueeze(0).unsqueeze(0)
    return Variable(_2d.expand(channel, 1, window_size, window_size).contiguous())


def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    size_average: bool = True,
) -> torch.Tensor:
    channel = img1.size(-3)
    window = _create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window: torch.Tensor,
    window_size: int,
    channel: int,
    size_average: bool = True,
) -> torch.Tensor:
    pad = window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channel) - mu1_mu2
    C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)


# ---------------------------------------------------------------------------
# Pixel metrics
# ---------------------------------------------------------------------------

def mse(
    img1: torch.Tensor,
    img2: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """MSE per image in a batch.  img1/img2: [B, C, H, W]"""
    if mask is not None:
        n_channel = img1.shape[1]
        img1 = img1.flatten(1)
        img2 = img2.flatten(1)
        mask = mask.flatten(1).repeat(1, n_channel)
        mask = mask != 0
        return torch.stack(
            [((img1[i, mask[i]] - img2[i, mask[i]]) ** 2).mean(0, keepdim=True) for i in range(img1.shape[0])],
            dim=0,
        )
    return ((img1 - img2) ** 2).reshape(img1.shape[0], -1).mean(1, keepdim=True)


@torch.no_grad()
def psnr(
    img1: torch.Tensor,
    img2: torch.Tensor,
    mask: torch.Tensor | None = None,
    pixel_max: float = 1.0,
) -> torch.Tensor:
    """PSNR per image in a batch.  img1/img2: [B, C, H, W]"""
    mse_val = mse(img1, img2, mask)
    return 10 * torch.log10(pixel_max**2 / mse_val.float())
