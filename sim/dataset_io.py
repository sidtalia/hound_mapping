"""Dataset I/O for HOUND rail sim."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml


def write_meta(out_dir: Path, meta: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "meta.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False)


def load_meta(out_dir: Path) -> Dict[str, Any]:
    with open(out_dir / "meta.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_frame(
    frames_dir: Path,
    index: int,
    depth: np.ndarray,
    trav_rgb: np.ndarray,
    T_wo: np.ndarray,
) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{index:06d}"
    np.save(frames_dir / f"{stem}_depth.npy", depth.astype(np.float32))
    depth_mm = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
    try:
        import cv2

        cv2.imwrite(str(frames_dir / f"{stem}_depth_mm.png"), depth_mm)
        cv2.imwrite(str(frames_dir / f"{stem}_trav.png"), trav_rgb[:, :, ::-1])
    except ImportError:
        pass
    np.save(frames_dir / f"{stem}_T_wo.npy", T_wo.astype(np.float64))
    # Row-major 4x4 for C++ smoke (append).
    with open(frames_dir / "poses_T_wo.bin", "ab") as f:
        f.write(np.ascontiguousarray(T_wo, dtype=np.float64).tobytes())


def write_camera_txt(out_dir: Path, cam: dict) -> None:
    """Simple key=value file for C++ smoke (no yaml dependency)."""
    lines = [
        f"width={int(cam['width'])}",
        f"height={int(cam['height'])}",
        f"fx={float(cam['fx'])}",
        f"fy={float(cam['fy'])}",
        f"cx={float(cam['cx'])}",
        f"cy={float(cam['cy'])}",
        f"max_depth_m={float(cam['max_depth_m'])}",
        f"n_frames={int(cam.get('n_frames', 0))}",
        f"map_res={float(cam.get('map_res', 0.1))}",
        f"window_radius_m={float(cam.get('window_radius_m', 10.0))}",
        f"voxel_size={float(cam.get('voxel_size', 0.05))}",
        f"elev_h={int(cam.get('elev_h', 0))}",
        f"elev_w={int(cam.get('elev_w', 0))}",
    ]
    (out_dir / "camera.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")



def list_frame_indices(frames_dir: Path) -> List[int]:
    ids = []
    for p in sorted(frames_dir.glob("*_depth.npy")):
        ids.append(int(p.name.split("_")[0]))
    return ids
