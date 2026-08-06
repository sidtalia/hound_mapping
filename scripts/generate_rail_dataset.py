#!/usr/bin/env python3
"""Pre-generate depth + traversability along an IGHA* path on a heightfield.

Example:
  python3 scripts/generate_rail_dataset.py \\
    --elev .../race-2_height.npy \\
    --path .../race-2_kinodynamic_elevation_IGHAStar_path.npy \\
    --out data/rail_sim/race-2 \\
    --map-res 0.1 --step 0.5 --max-frames 80
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sim.dataset_io import append_body_pose, save_frame, write_camera_txt, write_meta
from sim.heightfield_camera import (
    DEFAULT_VIEW_YAW_OFFSETS,
    CameraModel,
    densify_path,
    densify_path_timed,
    render_depth_trav,
    slope_cost_from_elev,
    synthetic_stereo_ir,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--elev", type=Path, required=True, help="HxW float32 elevation .npy")
    ap.add_argument(
        "--cost",
        type=Path,
        default=None,
        help="HxW cost 0–255 (255=free). Default: slope from elev",
    )
    ap.add_argument(
        "--path",
        type=Path,
        required=True,
        help="IGHA* path .npy (N,5) goal-first OR (N,>=3) start-first with --path-start-first",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--map-res", type=float, default=0.1)
    ap.add_argument(
        "--dt",
        type=float,
        default=0.05,
        help="Rail sample period (s). Uses path speed col (default 3). "
        "Set <=0 to use --step spatial densify instead.",
    )
    ap.add_argument("--step", type=float, default=0.5, help="Spatial densify (m) if --dt<=0")
    ap.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Cap frames (0 = whole trajectory)",
    )
    ap.add_argument("--path-start-first", action="store_true")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--fx", type=float, default=400.0)
    ap.add_argument("--fy", type=float, default=400.0)
    ap.add_argument("--max-depth", type=float, default=6.0)
    ap.add_argument("--cam-height", type=float, default=1.0)
    ap.add_argument(
        "--cam-pitch-down",
        type=float,
        default=15.0,
        help="Camera pitch down from horizontal (deg)",
    )
    ap.add_argument("--ray-steps", type=int, default=48)
    ap.add_argument("--window-radius", type=float, default=10.0)
    ap.add_argument("--voxel-size", type=float, default=0.05)
    ap.add_argument(
        "--multicam",
        action="store_true",
        help="Render front/left/right RGBD (+ front stereo IR). "
        "Views: NNNNNN_{view}_depth_mm.png etc.",
    )
    ap.add_argument(
        "--views",
        type=str,
        default="front,left,right",
        help="Comma-separated view names when --multicam (default front,left,right)",
    )
    args = ap.parse_args()

    elev = np.load(args.elev).astype(np.float32)
    if args.cost is not None:
        cost = np.load(args.cost).astype(np.float32)
        if cost.ndim == 3:
            cost = cost[..., 0]
    else:
        cost = slope_cost_from_elev(elev, args.map_res)

    path = np.load(args.path)
    if path.ndim != 2 or path.shape[1] < 3:
        raise SystemExit(f"Bad path shape {path.shape}")
    if not args.path_start_first:
        path = path[::-1].copy()  # goal-first → start→goal
    if args.dt > 0 and path.shape[1] >= 4:
        rail = densify_path_timed(path, dt=args.dt)
    else:
        rail = densify_path(path[:, :3], step_m=args.step)
    if args.max_frames > 0 and len(rail) > args.max_frames:
        idx = np.linspace(0, len(rail) - 1, args.max_frames).astype(np.int32)
        rail = rail[idx]
    print(
        f"Rail: {len(rail)} frames  dt={args.dt}  "
        f"path_waypoints={len(path)}  (~{len(rail)*max(args.dt,0):.1f}s)"
    )

    cam = CameraModel(
        width=args.width,
        height=args.height,
        fx=args.fx,
        fy=args.fy,
        cx=0.5 * args.width,
        cy=0.5 * args.height,
        max_depth_m=args.max_depth,
        cam_height_m=args.cam_height,
        pitch_down_deg=args.cam_pitch_down,
    )

    out = args.out
    maps = out / "maps"
    frames = out / "frames"
    maps.mkdir(parents=True, exist_ok=True)
    frames.mkdir(parents=True, exist_ok=True)
    np.save(maps / "elevation.npy", elev)
    np.save(maps / "costmap.npy", cost)
    elev.astype(np.float32).tofile(maps / "elevation.f32")
    cost.astype(np.float32).tofile(maps / "costmap.f32")
    np.save(out / "rail_path.npy", rail)

    poses_bin = frames / "poses_T_wo.bin"
    if poses_bin.exists():
        poses_bin.unlink()
    body_bin = frames / "poses_body.bin"
    if body_bin.exists():
        body_bin.unlink()

    views = [v.strip() for v in args.views.split(",") if v.strip()]
    if not args.multicam:
        views = ["front"]
    yaw_offsets = {
        v: float(DEFAULT_VIEW_YAW_OFFSETS.get(v, 0.0)) for v in views
    }

    write_meta(
        out,
        {
            "map_res": args.map_res,
            "elev_shape": list(elev.shape),
            "multicam": bool(args.multicam),
            "views": views,
            "yaw_offsets_deg": {v: float(np.rad2deg(yaw_offsets[v])) for v in views},
            "camera": {
                "width": cam.width,
                "height": cam.height,
                "fx": cam.fx,
                "fy": cam.fy,
                "cx": cam.cx,
                "cy": cam.cy,
                "max_depth_m": cam.max_depth_m,
                "cam_height_m": cam.cam_height_m,
                "pitch_down_deg": cam.pitch_down_deg,
            },
            "n_frames": len(rail),
            "dt_s": args.dt,
            "step_m": args.step,
            "window_radius_m": args.window_radius,
            "voxel_size": args.voxel_size,
            "source_elev": str(args.elev),
            "source_path": str(args.path),
        },
    )
    write_camera_txt(
        out,
        {
            "width": cam.width,
            "height": cam.height,
            "fx": cam.fx,
            "fy": cam.fy,
            "cx": cam.cx,
            "cy": cam.cy,
            "max_depth_m": cam.max_depth_m,
            "n_frames": len(rail),
            "map_res": args.map_res,
            "window_radius_m": args.window_radius,
            "voxel_size": args.voxel_size,
            "elev_h": elev.shape[0],
            "elev_w": elev.shape[1],
            "n_views": len(views),
            "views": ",".join(views),
        },
    )

    t0 = time.perf_counter()
    for i, pose in enumerate(rail):
        body_T = None
        for view in views:
            depth, trav, T = render_depth_trav(
                elev,
                cost,
                args.map_res,
                float(pose[0]),
                float(pose[1]),
                float(pose[2]),
                cam,
                n_steps=args.ray_steps,
                yaw_offset=yaw_offsets[view],
            )
            ir_l = ir_r = None
            if view == "front":
                ir_l, ir_r = synthetic_stereo_ir(trav, depth)
                body_T = T
            if args.multicam:
                save_frame(
                    frames,
                    i,
                    depth,
                    trav,
                    T,
                    view=view,
                    ir_left=ir_l,
                    ir_right=ir_r,
                    append_pose_bin=False,
                )
            else:
                save_frame(frames, i, depth, trav, T)
        if args.multicam and body_T is not None:
            append_body_pose(frames, body_T)
        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"[{i+1}/{len(rail)}] views={views} "
                f"pose=({pose[0]:.1f},{pose[1]:.1f},{pose[2]:.2f})"
            )
    dt = time.perf_counter() - t0
    print(
        f"Wrote {len(rail)} frames × {len(views)} views → {out} "
        f"in {dt:.1f}s ({dt/max(len(rail),1)*1e3:.0f} ms/pose)"
    )


if __name__ == "__main__":
    main()
