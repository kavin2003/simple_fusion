"""Gaussian utilities – inlined from gs-embedding/utils/gs_utils.py.

Only the subset required by point2gaussian_torch_batched is kept.
"""
from __future__ import annotations

import torch

# Spherical harmonic constants
C0 = 0.28209479177387814
C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
      -0.4570457994644658, 1.445305721320277, -0.5900435899266435]


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def canonicalize_rotation_batch(R: torch.Tensor) -> torch.Tensor:
    """Make PCA rotation matrices deterministic by fixing column signs."""
    B = R.shape[0]
    for col in range(3):
        col_vec = R[:, :, col]
        idx = torch.argmax(col_vec.abs(), dim=1)
        signs = torch.sign(col_vec[torch.arange(B), idx])
        signs[signs == 0] = 1.0
        R[:, :, col] = col_vec * signs.view(-1, 1)
    det = torch.det(R)
    flip = det < 0
    if flip.any():
        R[flip, :, 2] *= -1
    return R


def stable_rotmat2qvec_batch(R_batch: torch.Tensor) -> torch.Tensor:
    """Numerically stable batched rotation-matrix → quaternion (w,x,y,z)."""
    B = R_batch.shape[0]
    device, dtype = R_batch.device, R_batch.dtype
    q = torch.empty((B, 4), device=device, dtype=dtype)
    trace = R_batch[:, 0, 0] + R_batch[:, 1, 1] + R_batch[:, 2, 2]
    m0 = trace > 0
    m1 = (~m0) & (R_batch[:, 0, 0] >= R_batch[:, 1, 1]) & (R_batch[:, 0, 0] >= R_batch[:, 2, 2])
    m2 = (~m0) & (~m1) & (R_batch[:, 1, 1] >= R_batch[:, 2, 2])
    m3 = (~m0) & (~m1) & (~m2)
    for mask, diag in ((m0, None), (m1, 0), (m2, 1), (m3, 2)):
        if not mask.any():
            continue
        R = R_batch[mask]
        if diag is None:
            t = torch.sqrt(trace[mask] + 1.0) * 2
            q[mask, 0] = 0.25 * t
            q[mask, 1] = (R[:, 2, 1] - R[:, 1, 2]) / t
            q[mask, 2] = (R[:, 0, 2] - R[:, 2, 0]) / t
            q[mask, 3] = (R[:, 1, 0] - R[:, 0, 1]) / t
        elif diag == 0:
            t = torch.sqrt(1.0 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2]) * 2
            q[mask, 0] = (R[:, 2, 1] - R[:, 1, 2]) / t
            q[mask, 1] = 0.25 * t
            q[mask, 2] = (R[:, 0, 1] + R[:, 1, 0]) / t
            q[mask, 3] = (R[:, 0, 2] + R[:, 2, 0]) / t
        elif diag == 1:
            t = torch.sqrt(1.0 - R[:, 0, 0] + R[:, 1, 1] - R[:, 2, 2]) * 2
            q[mask, 0] = (R[:, 0, 2] - R[:, 2, 0]) / t
            q[mask, 1] = (R[:, 0, 1] + R[:, 1, 0]) / t
            q[mask, 2] = 0.25 * t
            q[mask, 3] = (R[:, 1, 2] + R[:, 2, 1]) / t
        else:
            t = torch.sqrt(1.0 - R[:, 0, 0] - R[:, 1, 1] + R[:, 2, 2]) * 2
            q[mask, 0] = (R[:, 1, 0] - R[:, 0, 1]) / t
            q[mask, 1] = (R[:, 0, 2] + R[:, 2, 0]) / t
            q[mask, 2] = (R[:, 1, 2] + R[:, 2, 1]) / t
            q[mask, 3] = 0.25 * t
    neg = q[:, 0] < 0
    q[neg] = -q[neg]
    return q / q.norm(dim=1, keepdim=True).clamp_min(1e-12)


# ---------------------------------------------------------------------------
# PCA ellipsoid fit
# ---------------------------------------------------------------------------

def fit_ellipsoid_pca_torch_batched(
    points_batch: torch.Tensor,
    eps: float = 1e-6,
    sort_descending: bool = False,
    canonicalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, N, _ = points_batch.shape
    mean = points_batch.mean(dim=1, keepdim=True)
    Xc = points_batch - mean
    C = torch.bmm(Xc.transpose(-2, -1), Xc) / max(N - 1, 1)
    eye = torch.eye(3, device=points_batch.device, dtype=points_batch.dtype)
    C = C + eps * eye.unsqueeze(0)
    evals, evecs = torch.linalg.eigh(C)
    if sort_descending:
        idx = torch.argsort(evals, dim=1, descending=True)
        evecs = torch.gather(evecs, 2, idx.unsqueeze(1).expand(-1, 3, -1))
    R = evecs
    if canonicalize:
        R = canonicalize_rotation_batch(R)
    Xp = torch.bmm(Xc, R)
    a = Xp[:, :, 0].abs().max(dim=1).values
    b = Xp[:, :, 1].abs().max(dim=1).values
    c = Xp[:, :, 2].abs().max(dim=1).values
    abc = torch.stack([a, b, c], dim=1) + eps
    return abc, R, mean.squeeze(1)


def ellipsoid_xyz2dirs_torch_batched(
    points: torch.Tensor,
    center: torch.Tensor,
    R: torch.Tensor,
    abc: torch.Tensor,
) -> torch.Tensor:
    Xc = points - center.unsqueeze(1)
    Xr = torch.bmm(Xc, R)
    return Xr / abc.unsqueeze(1)


# ---------------------------------------------------------------------------
# Spherical harmonics
# ---------------------------------------------------------------------------

def build_sh_basis_torch_batched(dirs: torch.Tensor, deg: int = 3) -> torch.Tensor:
    """dirs: [B, N, 3] → basis: [B, N, (deg+1)²]"""
    B, N, _ = dirs.shape
    x, y, z = dirs[:, :, 0], dirs[:, :, 1], dirs[:, :, 2]
    parts = [torch.full((B, N, 1), C0, device=dirs.device, dtype=dirs.dtype)]
    if deg >= 1:
        C1 = 0.4886025119029199
        parts += [(-C1 * y).unsqueeze(-1), (C1 * z).unsqueeze(-1), (-C1 * x).unsqueeze(-1)]
    if deg >= 2:
        xx, yy, zz, xy, yz, xz = x*x, y*y, z*z, x*y, y*z, x*z
        parts += [
            (C2[0] * xy).unsqueeze(-1), (C2[1] * yz).unsqueeze(-1),
            (C2[2] * (2*zz - xx - yy)).unsqueeze(-1),
            (C2[3] * xz).unsqueeze(-1), (C2[4] * (xx - yy)).unsqueeze(-1),
        ]
    if deg >= 3:
        xx, yy, zz, xy, yz, xz = x*x, y*y, z*z, x*y, y*z, x*z
        parts += [
            (C3[0] * y * (3*xx - yy)).unsqueeze(-1),
            (C3[1] * xy * z).unsqueeze(-1),
            (C3[2] * y * (4*zz - xx - yy)).unsqueeze(-1),
            (C3[3] * z * (2*zz - 3*xx - 3*yy)).unsqueeze(-1),
            (C3[4] * x * (4*zz - xx - yy)).unsqueeze(-1),
            (C3[5] * z * (xx - yy)).unsqueeze(-1),
            (C3[6] * x * (xx - 3*yy)).unsqueeze(-1),
        ]
    return torch.cat(parts, dim=-1)


def fit_sh_torch_batched(
    dirs: torch.Tensor,
    vals: torch.Tensor,
    deg: int = 3,
    eps: float = 1e-4,
) -> torch.Tensor:
    """dirs: [B,N,3], vals: [B,N,C] → sh: [B,K,C]"""
    B_mat = build_sh_basis_torch_batched(dirs, deg=deg)  # [B,N,K]
    BT = B_mat.transpose(-2, -1)                          # [B,K,N]
    normal = torch.bmm(BT, B_mat)
    K = normal.shape[-1]
    eye = torch.eye(K, device=dirs.device, dtype=dirs.dtype)
    normal = normal + eps * eye.unsqueeze(0)
    rhs = torch.bmm(BT, vals)
    try:
        L = torch.linalg.cholesky(normal)
        return torch.cholesky_solve(rhs, L)
    except RuntimeError:
        return torch.linalg.solve(normal, rhs)


# ---------------------------------------------------------------------------
# Main entry point used by model.py
# ---------------------------------------------------------------------------

def point2gaussian_torch_batched(
    data_batch: torch.Tensor,
    deg: int = 3,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Convert a decoded point cloud to Gaussian parameters.

    Args:
        data_batch: [B, N, 7] or [N, 7] — (x,y,z, r,g,b, opacity)
    Returns:
        [B, D] Gaussian parameter vector per batch item.
    """
    if data_batch.dim() == 2:
        data_batch = data_batch.unsqueeze(0)
    B, N, _ = data_batch.shape
    points = data_batch[:, :, :3]
    colors = data_batch[:, :, 3:6]
    opac = data_batch[:, :, 6]

    abc, R, center = fit_ellipsoid_pca_torch_batched(
        points, eps=eps, sort_descending=True, canonicalize=True
    )
    q = stable_rotmat2qvec_batch(R)

    dirs = ellipsoid_xyz2dirs_torch_batched(points, center, R, abc)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    sh = fit_sh_torch_batched(dirs, colors, deg=deg)  # [B,K,3]

    feat_dc = sh[:, 0, :]            # [B, 3]
    if sh.shape[1] > 1:
        rest = sh[:, 1:, :]          # [B, K-1, 3]
        feat_extra = torch.cat([rest[:, :, 0], rest[:, :, 1], rest[:, :, 2]], dim=1)
    else:
        feat_extra = torch.zeros((B, 0), device=sh.device, dtype=sh.dtype)

    scale = torch.log(abc.clamp_min(1e-6))
    m = opac.mean(dim=1).clamp(1e-6, 1 - 1e-6)
    opacity = torch.log(m / (1 - m))

    return torch.cat([center, scale, q, opacity.unsqueeze(1), feat_dc, feat_extra], dim=1)
