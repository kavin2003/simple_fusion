"""Add the compiled CUDA rasterizer to sys.path.

All other dependencies (Camera, HexPlane, loss functions, SFVAE) are now
self-contained inside simple_fusion/ and need no external path setup.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# CUDA rasterizer source lives inside simple_fusion/xray_rasterizer/
XRAY_SUBMODULE_ROOT = Path(__file__).resolve().parent / "xray_rasterizer"


def _prepend(path: Path) -> None:
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)


_prepend(XRAY_SUBMODULE_ROOT)
