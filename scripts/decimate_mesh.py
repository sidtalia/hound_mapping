#!/usr/bin/env python3
"""Voxel-weld a nvblox color mesh for mesh_pf (Assimp: obj/ply/dae).

The TSDF PLY is in odom. Pass --map-odom so vertices land in map
(mesh_pf.map_frame). Sibling <stem>.map_odom.yaml is used if present.

  python3 .../decimate_mesh.py /root/colcon_ws/maps/hound_tsdf_mesh.ply
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np


def _load_ply(path: Path):
    spec = importlib.util.spec_from_file_location(
        "show_ply", Path(__file__).with_name("show_ply.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._parse_ply(path)


def _quat_to_R(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def load_map_odom(path: Path) -> np.ndarray:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    xyz = [float(v) for v in data["xyz"]]
    q = [float(v) for v in data["xyzw"]]
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _quat_to_R(*q)
    T[:3, 3] = xyz
    return T


def triangulate(faces) -> np.ndarray:
    tris = []
    for face in faces:
        if len(face) < 3:
            continue
        a = int(face[0])
        for k in range(1, len(face) - 1):
            tris.append((a, int(face[k]), int(face[k + 1])))
    if not tris:
        raise SystemExit("no triangles")
    return np.asarray(tris, dtype=np.int64)


def voxel_weld(verts: np.ndarray, tris: np.ndarray, voxel_m: float):
    keys = np.floor(verts / voxel_m + 0.5).astype(np.int32)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    acc = np.zeros((int(inv.max()) + 1, 3), dtype=np.float64)
    np.add.at(acc, inv, verts)
    counts = np.bincount(inv).astype(np.float64)
    out_v = acc / counts[:, None]
    out_t = inv[tris]
    ok = (
        (out_t[:, 0] != out_t[:, 1])
        & (out_t[:, 1] != out_t[:, 2])
        & (out_t[:, 0] != out_t[:, 2])
    )
    out_t = np.unique(out_t[ok], axis=0)
    used = np.unique(out_t.reshape(-1))
    remap = np.full(out_v.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.size, dtype=np.int64)
    return out_v[used], remap[out_t]


def write_map_yaml(
    path: Path,
    *,
    frame_id: str,
    lo: np.ndarray,
    hi: np.ndarray,
    n_verts: int,
    n_tris: int,
    voxel_m: float,
    mesh_name: str,
) -> None:
    import yaml

    pad_xy = 1.0
    payload = {
        "frame_id": frame_id,
        "mesh": mesh_name,
        "voxel_m": float(voxel_m),
        "n_verts": int(n_verts),
        "n_tris": int(n_tris),
        "bbox": {
            "xmin": float(lo[0]),
            "ymin": float(lo[1]),
            "zmin": float(lo[2]),
            "xmax": float(hi[0]),
            "ymax": float(hi[1]),
            "zmax": float(hi[2]),
        },
        "init_bb": {
            "xmin": float(lo[0] - pad_xy),
            "ymin": float(lo[1] - pad_xy),
            "zmin": float(lo[2] - 0.2),
            "xmax": float(hi[0] + pad_xy),
            "ymax": float(hi[1] + pad_xy),
            "zmax": float(hi[2] + 0.5),
        },
    }
    path.write_text(
        "# mesh_pf map sidecar (init_bb padded from mesh AABB).\n"
        + yaml.safe_dump(payload, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )


def write_obj(path: Path, verts: np.ndarray, tris: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# mesh_pf triangle map (map frame)\n")
        for x, y, z in verts:
            handle.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in tris:
            handle.write(f"f {a + 1} {b + 1} {c + 1}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ply", type=Path)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="default: <ply stem>_pf.obj",
    )
    ap.add_argument(
        "--voxel",
        type=float,
        default=0.15,
        help="weld grid (m). 0.15 ≈ 3× 5 cm TSDF voxels",
    )
    ap.add_argument(
        "--faces",
        type=int,
        default=0,
        help="if set, grow --voxel until triangle count ≤ this",
    )
    ap.add_argument(
        "--map-odom",
        type=Path,
        default=None,
        help="map←odom yaml (default: sibling <layercake stem>.map_odom.yaml)",
    )
    ap.add_argument(
        "--no-map-odom",
        action="store_true",
        help="leave vertices in the PLY frame (odom)",
    )
    args = ap.parse_args()
    if not args.ply.is_file():
        print(f"missing {args.ply}", file=sys.stderr)
        return 1

    verts_l, _colors, faces = _load_ply(args.ply)
    verts = np.asarray(verts_l, dtype=np.float64)
    tris = triangulate(faces)
    print(f"loaded {len(verts)} verts {len(tris)} tris from {args.ply}")

    tf_path = args.map_odom
    if tf_path is None and not args.no_map_odom:
        guess = args.ply.with_name(args.ply.stem.replace("_mesh", "") + ".map_odom.yaml")
        if not guess.is_file():
            guess = args.ply.with_name("hound_tsdf.map_odom.yaml")
        tf_path = guess if guess.is_file() else None
    if tf_path is not None:
        T = load_map_odom(tf_path)
        h = np.ones((len(verts), 4), dtype=np.float64)
        h[:, :3] = verts
        verts = (T @ h.T).T[:, :3]
        print(f"applied map←odom {tf_path}")

    voxel = max(float(args.voxel), 1e-4)
    welded_v, welded_t = voxel_weld(verts, tris, voxel)
    if args.faces > 0:
        while len(welded_t) > args.faces and voxel < 5.0:
            voxel *= 1.15
            welded_v, welded_t = voxel_weld(verts, tris, voxel)
        print(f"target ≤{args.faces} faces → voxel={voxel:.3f}m")

    out = args.out or args.ply.with_name(args.ply.stem + "_pf.obj")
    write_obj(out, welded_v, welded_t)
    lo = welded_v.min(axis=0)
    hi = welded_v.max(axis=0)
    meta = out.with_suffix(".yaml")
    write_map_yaml(
        meta,
        frame_id="map" if tf_path is not None else "odom",
        lo=lo,
        hi=hi,
        n_verts=len(welded_v),
        n_tris=len(welded_t),
        voxel_m=voxel,
        mesh_name=out.name,
    )
    print(
        f"wrote {len(welded_v)} verts {len(welded_t)} tris → {out}\n"
        f"sidecar {meta}  bbox xyz "
        f"[{lo[0]:.1f},{hi[0]:.1f}] [{lo[1]:.1f},{hi[1]:.1f}] "
        f"[{lo[2]:.1f},{hi[2]:.1f}]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
