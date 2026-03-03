"""Camera geometry utilities – inlined from X²-Gaussian (no external dependency)."""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


# ---------------------------------------------------------------------------
# Beam-mode lookup
# ---------------------------------------------------------------------------

mode_id = {
    "parallel": 0,
    "cone": 1,
}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def getWorld2View2(
    R: np.ndarray,
    t: np.ndarray,
    translate: np.ndarray = np.array([0.0, 0.0, 0.0]),
    scale: float = 1.0,
) -> np.ndarray:
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    return np.float32(np.linalg.inv(C2W))


def getProjectionMatrix(fovX: float, fovY: float, mode: int, scanner_cfg: dict) -> torch.Tensor:
    if mode == 0:  # Parallel beam – identity projection
        return torch.eye(4)
    elif mode == 1:  # Cone beam – perspective
        znear, zfar = 0.01, 100.0
        tanHalfFovY = math.tan(fovY / 2)
        tanHalfFovX = math.tan(fovX / 2)
        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right
        z_sign = 1.0
        P = torch.zeros(4, 4)
        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        return P
    else:
        raise ValueError(f"Unsupported scanner mode: {mode}")


def angle2pose(DSO: float, angle: float) -> np.ndarray:
    """Convert gantry angle (radians) to camera-to-world matrix.

    Applies:
      1. Rotate -90° around x-axis
      2. Rotate +90° around z-axis
      3. Rotate by *angle* around z-axis
    """
    phi1 = -np.pi / 2
    R1 = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(phi1), -np.sin(phi1)],
        [0.0, np.sin(phi1),  np.cos(phi1)],
    ])
    phi2 = np.pi / 2
    R2 = np.array([
        [np.cos(phi2), -np.sin(phi2), 0.0],
        [np.sin(phi2),  np.cos(phi2), 0.0],
        [0.0, 0.0, 1.0],
    ])
    R3 = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle),  np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    rot = R3 @ R2 @ R1
    trans = np.array([DSO * np.cos(angle), DSO * np.sin(angle), 0.0])
    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = trans
    return transform


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

class Camera(nn.Module):
    def __init__(
        self,
        colmap_id: int,
        scanner_cfg: dict,
        R: np.ndarray,
        T: np.ndarray,
        angle: float,
        mode: int,
        FoVx: float,
        FoVy: float,
        image: torch.Tensor,
        image_name: str,
        uid: int,
        time: float,
        phase: int,
        trans: np.ndarray = np.array([0.0, 0.0, 0.0]),
        scale: float = 1.0,
        data_device: str = "cuda",
    ):
        super().__init__()
        self.uid = uid
        self.colmap_id = colmap_id
        self.scanner_cfg = scanner_cfg
        self.R = R
        self.T = T
        self.angle = angle
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.mode = mode
        self.image_name = image_name
        self.time = float(time)
        self.phase = phase

        try:
            self.data_device = torch.device(data_device)
        except Exception:
            self.data_device = torch.device("cuda")

        self.original_image = image.to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        self.trans = trans
        self.scale = scale

        self.world_view_transform = (
            torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        )
        self.projection_matrix = (
            getProjectionMatrix(
                fovX=self.FoVx,
                fovY=self.FoVy,
                mode=mode,
                scanner_cfg=scanner_cfg,
            )
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
