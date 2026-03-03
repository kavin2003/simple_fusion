from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import pickle

import numpy as np
import torch

from simple_fusion.camera import Camera, angle2pose, mode_id


@dataclass
class FusionScene:
    source_path: str
    scanner_cfg: dict
    scene_scale: float
    train_cameras: list[Camera]
    test_cameras: list[Camera]
    volume: Optional[torch.Tensor]

    @property
    def bbox(self) -> torch.Tensor:
        off_origin = torch.tensor(self.scanner_cfg["offOrigin"], dtype=torch.float32)
        s_voxel = torch.tensor(self.scanner_cfg["sVoxel"], dtype=torch.float32)
        return torch.stack([off_origin - s_voxel / 2, off_origin + s_voxel / 2], dim=0)


def _scaled_scanner_cfg(data: dict) -> tuple[dict, float]:
    scanner_cfg = {
        "DSD": data["DSD"] / 1000,
        "DSO": data["DSO"] / 1000,
        "nVoxel": data["nVoxel"],
        "dVoxel": (np.array(data["dVoxel"]) / 1000).tolist(),
        "sVoxel": (np.array(data["nVoxel"]) * np.array(data["dVoxel"]) / 1000).tolist(),
        "nDetector": data["nDetector"],
        "dDetector": (np.array(data["dDetector"]) / 1000).tolist(),
        "sDetector": (
            np.array(data["nDetector"]) * np.array(data["dDetector"]) / 1000
        ).tolist(),
        "offOrigin": (np.array(data["offOrigin"]) / 1000).tolist(),
        "offDetector": (np.array(data["offDetector"]) / 1000).tolist(),
        "totalAngle": data["totalAngle"],
        "startAngle": data["startAngle"],
        "accuracy": data["accuracy"],
        "mode": data["mode"],
        "filter": None,
    }

    scene_scale = 2 / max(scanner_cfg["sVoxel"])
    for key in [
        "dVoxel",
        "sVoxel",
        "sDetector",
        "dDetector",
        "offOrigin",
        "offDetector",
        "DSD",
        "DSO",
    ]:
        scanner_cfg[key] = (np.array(scanner_cfg[key]) * scene_scale).tolist()
    return scanner_cfg, scene_scale


def _build_camera(
    scanner_cfg: dict,
    projection: np.ndarray,
    angle: float,
    time: float,
    phase: int,
    uid: int,
    device: str,
) -> Camera:
    c2w = angle2pose(scanner_cfg["DSO"], angle)
    w2c = np.linalg.inv(c2w)
    r = np.transpose(w2c[:3, :3])
    t = w2c[:3, 3]

    fov_x = np.arctan2(scanner_cfg["sDetector"][1] / 2, scanner_cfg["DSD"]) * 2
    fov_y = np.arctan2(scanner_cfg["sDetector"][0] / 2, scanner_cfg["DSD"]) * 2

    image = torch.from_numpy(projection.astype(np.float32))[None]
    return Camera(
        colmap_id=uid,
        scanner_cfg=scanner_cfg,
        R=r,
        T=t,
        angle=float(angle),
        mode=mode_id[scanner_cfg["mode"]],
        FoVx=float(fov_x),
        FoVy=float(fov_y),
        image=image,
        image_name=f"{uid:04d}",
        uid=uid,
        time=float(time),
        phase=int(phase),
        data_device=device,
    )


def _build_split_cameras(
    split: dict,
    scanner_cfg: dict,
    scene_scale: float,
    uid_offset: int,
    device: str,
) -> list[Camera]:
    angles = split["angles"]
    projections = split["projections"]
    times = split.get("time")
    phases = split.get("phase")

    if times is None:
        times = np.linspace(0.0, 1.0, len(angles), endpoint=False, dtype=np.float32)
    if phases is None:
        phases = np.zeros(len(angles), dtype=np.int32)

    cameras: list[Camera] = []
    for index, (angle, projection, time_value, phase_value) in enumerate(
        zip(angles, projections, times, phases)
    ):
        cameras.append(
            _build_camera(
                scanner_cfg=scanner_cfg,
                projection=projection * scene_scale,
                angle=float(angle),
                time=float(time_value),
                phase=int(phase_value),
                uid=uid_offset + index,
                device=device,
            )
        )
    return cameras


def load_naf_scene(
    pickle_path: str,
    *,
    device: str = "cuda",
    eval_split: bool = True,
    include_volume: bool = False,
) -> FusionScene:
    pickle_file = Path(pickle_path)
    with pickle_file.open("rb") as handle:
        data = pickle.load(handle)

    scanner_cfg, scene_scale = _scaled_scanner_cfg(data)
    train_cameras = _build_split_cameras(
        split=data["train"],
        scanner_cfg=scanner_cfg,
        scene_scale=scene_scale,
        uid_offset=0,
        device=device,
    )

    if eval_split:
        test_split = data["val"] if "val" in data else data.get("test")
        if test_split is None:
            test_cameras = []
        else:
            test_cameras = _build_split_cameras(
                split=test_split,
                scanner_cfg=scanner_cfg,
                scene_scale=scene_scale,
                uid_offset=len(train_cameras),
                device=device,
            )
    else:
        test_cameras = []

    volume = None
    if include_volume and "image" in data:
        volume = torch.from_numpy(data["image"]).float().to(device)

    return FusionScene(
        source_path=str(pickle_file),
        scanner_cfg=scanner_cfg,
        scene_scale=scene_scale,
        train_cameras=train_cameras,
        test_cameras=test_cameras,
        volume=volume,
    )


def infer_init_point_cloud_path(source_path: str) -> Path:
    """Find the .npy point cloud generated by x2-gaussian/initialize_pcd.py.

    X²-Gaussian names the file ``init_{stem}_debug.npy`` for .pkl inputs and
    ``init_{stem}.npy`` for Blender-directory inputs.  We try both so that
    the user does not need to rename files or pass --init-path explicitly.
    """
    source = Path(source_path)
    candidate_debug = source.parent / f"init_{source.stem}_debug.npy"
    candidate_plain = source.parent / f"init_{source.stem}.npy"
    if candidate_debug.exists():
        return candidate_debug
    return candidate_plain  # may or may not exist; caller handles FileNotFoundError


def load_init_point_cloud(
    source_path: str,
    init_path: Optional[str] = None,
    *,
    max_gaussians: Optional[int] = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    point_cloud_path = Path(init_path) if init_path is not None else infer_init_point_cloud_path(source_path)
    if not point_cloud_path.exists():
        raise FileNotFoundError(
            f"Cannot find initialization point cloud: {point_cloud_path}. "
            "Generate it with x2-gaussian/initialize_pcd.py or pass --init-path."
        )

    point_cloud = np.load(point_cloud_path)
    xyz = point_cloud[:, :3].astype(np.float32)
    density = point_cloud[:, 3:4].astype(np.float32) if point_cloud.shape[1] > 3 else np.ones((len(xyz), 1), dtype=np.float32)
    keep_indices = np.arange(len(xyz))

    if max_gaussians is not None and len(xyz) > max_gaussians:
        if density is not None:
            keep_indices = np.argsort(-density.squeeze(-1))[:max_gaussians]
        else:
            rng = np.random.default_rng(seed)
            keep_indices = rng.choice(len(xyz), size=max_gaussians, replace=False)
        keep_indices = np.sort(keep_indices)
        xyz = xyz[keep_indices]
        density = density[keep_indices]

    return xyz, density, keep_indices


def load_optional_latents(
    latent_path: Optional[str],
    keep_indices: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    if latent_path is None:
        return None

    latent_file = Path(latent_path)
    if not latent_file.exists():
        raise FileNotFoundError(f"Cannot find latent file: {latent_file}")

    if latent_file.suffix == ".npz":
        payload = np.load(latent_file)
        if "emb" not in payload:
            raise KeyError(f"{latent_file} does not contain an 'emb' array.")
        latent = payload["emb"].astype(np.float32)
    elif latent_file.suffix == ".npy":
        latent = np.load(latent_file).astype(np.float32)
    else:
        raise ValueError(f"Unsupported latent file format: {latent_file.suffix}")

    if keep_indices is not None:
        latent = latent[keep_indices]
    return latent
