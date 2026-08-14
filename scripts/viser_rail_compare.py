#!/usr/bin/env python3
"""Viser scrubber for rail-sim datasets: GT elev, rail, moving camera, depth/trav.

  python3 scripts/viser_rail_compare.py --dataset data/rail_sim/race-2 --port 8081
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sim.heightfield_camera import sample_elev


def load_mapped_f32(path: Path):
    meta = path.with_suffix(".txt")
    if not meta.exists():
        raise FileNotFoundError(f"Need sidecar {meta} next to {path}")
    kv = {}
    for line in meta.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()
    h, w = int(kv["height"]), int(kv["width"])
    arr = np.fromfile(path, dtype=np.float32).reshape(h, w)
    return (
        arr,
        float(kv["resolution"]),
        (float(kv["origin_x"]), float(kv["origin_y"])),
    )


def elev_to_points(
    me: np.ndarray, res: float, origin_xy, max_pts: int = 80000
):
    yy, xx = np.where(np.isfinite(me))
    if len(xx) == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.uint8),
            0,
        )
    xs = origin_xy[0] + (xx + 0.5) * res
    ys = origin_xy[1] + (yy + 0.5) * res
    zs = me[yy, xx]
    pts = np.stack([xs, ys, zs], axis=-1).astype(np.float32)
    n_fin = len(pts)
    if len(pts) > max_pts:
        idx = np.arange(0, len(pts), max(1, len(pts) // max_pts))
        pts = pts[idx]
        zs = pts[:, 2]
        yy, xx = yy[idx], xx[idx]
    zc = (zs - zs.min()) / (zs.max() - zs.min() + 1e-6)
    cols = np.zeros((len(pts), 3), dtype=np.uint8)
    cols[:, 0] = (255 * zc).astype(np.uint8)
    cols[:, 2] = (255 * (1 - zc)).astype(np.uint8)
    return pts, cols, n_fin


def cost_to_points(
    elev: np.ndarray,
    cost: np.ndarray,
    res: float,
    origin_xy,
    max_pts: int = 80000,
    z_lift: float = 0.05,
):
    """Color elev surface by cost [0,1]: green=free, red=lethal."""
    if elev.shape != cost.shape:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.uint8),
            0,
        )
    yy, xx = np.where(np.isfinite(elev))
    if len(xx) == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.uint8),
            0,
        )
    xs = origin_xy[0] + (xx + 0.5) * res
    ys = origin_xy[1] + (yy + 0.5) * res
    zs = elev[yy, xx] + z_lift
    c = np.clip(cost[yy, xx], 0.0, 1.0)
    pts = np.stack([xs, ys, zs], axis=-1).astype(np.float32)
    n_fin = len(pts)
    if len(pts) > max_pts:
        idx = np.arange(0, len(pts), max(1, len(pts) // max_pts))
        pts = pts[idx]
        c = c[idx]
    cols = np.zeros((len(pts), 3), dtype=np.uint8)
    cols[:, 0] = (255 * c).astype(np.uint8)
    cols[:, 1] = (255 * (1.0 - c)).astype(np.uint8)
    return pts, cols, n_fin


def load_frame_elev(frames_dir: Path, index: int):
    stem = f"{index:06d}"
    path = frames_dir / f"{stem}_elev.f32"
    if not path.exists():
        return None
    return load_mapped_f32(path)


def load_frame_cost(frames_dir: Path, index: int, shape_hw):
    """Load cost using elev sidecar dims (same HxW/origin)."""
    stem = f"{index:06d}"
    path = frames_dir / f"{stem}_cost.f32"
    meta = frames_dir / f"{stem}_elev.txt"
    if not path.exists() or not meta.exists():
        return None
    kv = {}
    for line in meta.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()
    h, w = int(kv["height"]), int(kv["width"])
    if (h, w) != shape_hw:
        return None
    return np.fromfile(path, dtype=np.float32).reshape(h, w)


def rotmat_to_wxyz(R: np.ndarray) -> np.ndarray:
    """SO(3) → viser wxyz quaternion."""
    try:
        from scipy.spatial.transform import Rotation

        q = Rotation.from_matrix(R).as_quat()  # xyzw
        return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)
    except ImportError:
        # Shepperd
        t = np.trace(R)
        if t > 0:
            s = np.sqrt(t + 1.0) * 2
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        return np.array([w, x, y, z], dtype=np.float64)


def height_mesh(z: np.ndarray, res: float, origin_xy, stride: int):
    import trimesh

    zs = z[::stride, ::stride]
    h, w = zs.shape
    xs = origin_xy[0] + np.arange(w) * res * stride
    ys = origin_xy[1] + np.arange(h) * res * stride
    xx, yy = np.meshgrid(xs, ys)
    valid = np.isfinite(zs)
    fill = float(np.nanmean(zs[valid]) if valid.any() else 0.0)
    verts = np.stack(
        [xx, yy, np.nan_to_num(zs, nan=fill)],
        axis=-1,
    ).reshape(-1, 3)
    faces = []
    for r in range(h - 1):
        for c in range(w - 1):
            if not (
                valid[r, c]
                and valid[r, c + 1]
                and valid[r + 1, c]
                and valid[r + 1, c + 1]
            ):
                continue
            i = r * w + c
            faces.append([i, i + 1, i + w])
            faces.append([i + 1, i + w + 1, i + w])
    if not faces:
        return None
    mesh = trimesh.Trimesh(vertices=verts, faces=np.asarray(faces), process=False)
    zc = verts[:, 2]
    zc = (zc - zc.min()) / (zc.max() - zc.min() + 1e-6)
    colors = np.zeros((len(verts), 4), dtype=np.uint8)
    colors[:, 0] = (255 * zc).astype(np.uint8)
    colors[:, 2] = (255 * (1 - zc)).astype(np.uint8)
    colors[:, 3] = 255
    mesh.visual.vertex_colors = colors
    return mesh


def depth_to_rgb(depth_m: np.ndarray, max_depth: float) -> np.ndarray:
    """Viridis-ish depth viz; black = invalid."""
    d = depth_m.astype(np.float32)
    valid = d > 0
    t = np.zeros_like(d)
    t[valid] = np.clip(d[valid] / max(max_depth, 1e-3), 0.0, 1.0)
    rgb = np.zeros((*d.shape, 3), dtype=np.uint8)
    # simple blue→cyan→yellow→red
    rgb[..., 0] = (255 * np.clip(1.5 * t - 0.5, 0, 1)).astype(np.uint8)
    rgb[..., 1] = (255 * np.clip(1.0 - np.abs(t - 0.5) * 2, 0, 1)).astype(np.uint8)
    rgb[..., 2] = (255 * np.clip(1.5 * (1 - t) - 0.5, 0, 1)).astype(np.uint8)
    rgb[~valid] = 0
    return rgb


def load_poses(frames_dir: Path, n: int) -> np.ndarray:
    bin_path = frames_dir / "poses_T_wo.bin"
    if bin_path.exists():
        arr = np.fromfile(bin_path, dtype=np.float64)
        if arr.size >= n * 16:
            return arr[: n * 16].reshape(n, 4, 4)
    body_bin = frames_dir / "poses_body.bin"
    if body_bin.exists():
        arr = np.fromfile(body_bin, dtype=np.float64)
        if arr.size >= n * 16:
            return arr[: n * 16].reshape(n, 4, 4)
    poses = []
    for i in range(n):
        stem = f"{i:06d}"
        for cand in (
            frames_dir / f"{stem}_T_wo.npy",
            frames_dir / f"{stem}_front_T_wo.bin",
            frames_dir / f"{stem}_T_wo.bin",
        ):
            if cand.suffix == ".npy" and cand.exists():
                poses.append(np.load(cand))
                break
            if cand.suffix == ".bin" and cand.exists():
                poses.append(np.fromfile(cand, dtype=np.float64).reshape(4, 4))
                break
        else:
            break
    return np.asarray(poses, dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--mapped-elev", type=Path, default=None)
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--play-hz", type=float, default=10.0)
    args = ap.parse_args()

    try:
        import viser
    except ImportError as e:
        raise SystemExit(f"pip install viser trimesh: {e}") from e

    dataset = args.dataset
    frames_dir = dataset / "frames"
    elev = np.load(dataset / "maps" / "elevation.npy")
    meta = yaml.safe_load((dataset / "meta.yaml").read_text())
    map_res = float(meta.get("map_res", 0.1))
    cam_meta = meta.get("camera", {})
    max_depth = float(cam_meta.get("max_depth_m", 6.0))
    fov_y_deg = float(
        np.rad2deg(
            2.0
            * np.arctan2(
                0.5 * float(cam_meta.get("height", 360)),
                float(cam_meta.get("fy", 400)),
            )
        )
    )
    aspect = float(cam_meta.get("width", 640)) / float(cam_meta.get("height", 360))

    rail = np.load(dataset / "rail_path.npy")
    n = int(meta.get("n_frames", len(rail)))
    poses = load_poses(frames_dir, n)
    n = min(n, len(poses), len(rail))
    if n < 1:
        raise SystemExit("No frames/poses in dataset")

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.gui.set_panel_label("rail scrubber")

    gt = height_mesh(elev, map_res, (0.0, 0.0), args.stride)
    if gt is not None:
        server.scene.add_mesh_trimesh("/gt_elev", gt)

    z_rail = sample_elev(elev, map_res, rail[:n, 0], rail[:n, 1]) + 0.2
    rail_pts = np.stack(
        [rail[:n, 0], rail[:n, 1], np.nan_to_num(z_rail, nan=0.0)], axis=-1
    ).astype(np.float32)
    server.scene.add_point_cloud(
        "/rail", points=rail_pts, colors=(40, 200, 80), point_size=0.12
    )

    show_mapped = server.gui.add_checkbox("show mapped elev (per-frame)", True)
    show_cost = server.gui.add_checkbox("show mapped cost (per-frame)", True)
    elev_cov = server.gui.add_markdown("mapped elev: (run smoke_rail_mapper)")
    n_frame_elev = sum(1 for p in frames_dir.glob("*_elev.f32"))
    n_frame_cost = sum(1 for p in frames_dir.glob("*_cost.f32"))
    if n_frame_elev == 0:
        elev_cov.content = (
            "**No per-frame elev dumps.** Run:\n"
            "`ros2 run hound_mapping smoke_rail_mapper --ros-args "
            f"-p dataset:={dataset}`"
        )
    else:
        elev_cov.content = (
            f"per-frame elev: **{n_frame_elev}/{n}**  "
            f"cost: **{n_frame_cost}/{n}**"
        )

    # Placeholder; updated every frame.
    mapped_cloud = server.scene.add_point_cloud(
        "/mapped_elev_pts",
        points=np.zeros((1, 3), dtype=np.float32),
        colors=(255, 128, 0),
        point_size=0.08,
        visible=bool(show_mapped.value),
    )
    cost_cloud = server.scene.add_point_cloud(
        "/mapped_cost_pts",
        points=np.zeros((1, 3), dtype=np.float32),
        colors=(0, 255, 0),
        point_size=0.09,
        visible=bool(show_cost.value),
    )

    @show_mapped.on_update
    def _mapped(_=None) -> None:
        mapped_cloud.visible = bool(show_mapped.value)

    @show_cost.on_update
    def _cost(_=None) -> None:
        cost_cloud.visible = bool(show_cost.value)

    frame_slider = server.gui.add_slider(
        "frame", min=0, max=n - 1, step=1, initial_value=0
    )
    play_cb = server.gui.add_checkbox("play", False)
    hz_num = server.gui.add_number("play Hz", args.play_hz, min=0.5, max=60.0, step=0.5)
    info = server.gui.add_markdown("frame 0")
    depth_gui = server.gui.add_image(
        np.zeros((180, 320, 3), dtype=np.uint8), label="depth"
    )
    trav_gui = server.gui.add_image(
        np.zeros((180, 320, 3), dtype=np.uint8), label="trav"
    )

    cam_frame = server.scene.add_frame(
        "/camera",
        wxyz=rotmat_to_wxyz(poses[0, :3, :3]),
        position=poses[0, :3, 3],
        axes_length=0.6,
        axes_radius=0.03,
    )
    frustum = server.scene.add_camera_frustum(
        "/camera/frustum",
        fov=fov_y_deg * np.pi / 180.0,
        aspect=aspect,
        scale=0.35,
        color=(255, 180, 40),
    )
    marker = server.scene.add_point_cloud(
        "/camera_marker",
        points=poses[0:1, :3, 3].astype(np.float32),
        colors=(255, 80, 40),
        point_size=0.25,
    )

    state = {"i": 0, "suppress_slider": False}

    def show_frame(i: int) -> None:
        i = int(np.clip(i, 0, n - 1))
        T = poses[i]
        R = T[:3, :3]
        t = T[:3, 3]
        cam_frame.wxyz = rotmat_to_wxyz(R)
        cam_frame.position = tuple(t.tolist())
        frustum.wxyz = cam_frame.wxyz
        frustum.position = tuple(t.tolist())
        marker.points = np.asarray(t, dtype=np.float32).reshape(1, 3)

        stem = f"{i:06d}"
        depth_p = frames_dir / f"{stem}_depth.npy"
        if not depth_p.exists():
            depth_p = frames_dir / f"{stem}_front_depth.npy"
        trav_p = frames_dir / f"{stem}_trav.png"
        if not trav_p.exists():
            trav_p = frames_dir / f"{stem}_front_trav.png"
        if depth_p.exists():
            depth = np.load(depth_p)
            rgb = depth_to_rgb(depth, max_depth)
            rgb_s = rgb[::2, ::2]
            depth_gui.image = rgb_s
            frustum.image = rgb_s
            valid = float(np.count_nonzero(depth > 0)) / depth.size
        else:
            valid = -1.0

        if trav_p.exists():
            try:
                import cv2

                bgr = cv2.imread(str(trav_p), cv2.IMREAD_COLOR)
                if bgr is not None:
                    trav_gui.image = bgr[::2, ::2, ::-1]
            except Exception:
                pass

        loaded = load_frame_elev(frames_dir, i)
        if loaded is not None:
            me, mres, origin = loaded
            pts, cols, n_fin = elev_to_points(me, mres, origin)
            if len(pts) == 0:
                mapped_cloud.points = np.zeros((1, 3), dtype=np.float32)
                mapped_cloud.colors = np.zeros((1, 3), dtype=np.uint8)
            else:
                mapped_cloud.points = pts
                mapped_cloud.colors = cols

            cost = load_frame_cost(frames_dir, i, me.shape)
            if cost is not None:
                cpts, ccols, n_cost = cost_to_points(me, cost, mres, origin)
                if len(cpts) == 0:
                    cost_cloud.points = np.zeros((1, 3), dtype=np.float32)
                    cost_cloud.colors = np.zeros((1, 3), dtype=np.uint8)
                else:
                    cost_cloud.points = cpts
                    cost_cloud.colors = ccols
                lethal_frac = float(np.mean(cost >= 0.5))
                elev_cov.content = (
                    f"frame {i} elev finite: **{n_fin}** "
                    f"({100.0 * n_fin / me.size:.1f}% of {me.shape[1]}×{me.shape[0]})  "
                    f"cost lethal≥0.5: **{100.0 * lethal_frac:.1f}%**"
                )
            else:
                cost_cloud.points = np.zeros((1, 3), dtype=np.float32)
                cost_cloud.colors = np.zeros((1, 3), dtype=np.uint8)
                elev_cov.content = (
                    f"frame {i} elev finite: **{n_fin}** "
                    f"({100.0 * n_fin / me.size:.1f}% of {me.shape[1]}×{me.shape[0]})  "
                    f"(no `*_cost.f32` — re-run smoke)"
                )
        else:
            elev_cov.content = f"frame {i}: no `*_elev.f32` (re-run smoke)"
            mapped_cloud.points = np.zeros((1, 3), dtype=np.float32)
            cost_cloud.points = np.zeros((1, 3), dtype=np.float32)

        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        info.content = (
            f"**frame {i}/{n - 1}**  t≈{i * float(meta.get('dt_s', 0.05)):.2f}s  "
            f"pose=({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f})  "
            f"yaw={yaw:.2f}  depth_valid={100 * valid:.1f}%"
        )
        state["i"] = i

    def set_frame(i: int) -> None:
        """Advance scene + slider without re-entrant on_update deadlock."""
        i = int(np.clip(i, 0, n - 1))
        state["suppress_slider"] = True
        try:
            frame_slider.value = i
            show_frame(i)
        finally:
            state["suppress_slider"] = False

    @frame_slider.on_update
    def _on_slider(_=None) -> None:
        if state["suppress_slider"]:
            return
        show_frame(int(frame_slider.value))

    set_frame(0)

    print(f"rail scrubber: {n} frames  http://0.0.0.0:{args.port}")
    print("Use the frame slider / play checkbox in the viser GUI.")
    last = time.perf_counter()
    try:
        while True:
            time.sleep(0.01)
            if not bool(play_cb.value):
                last = time.perf_counter()
                continue
            hz = max(0.5, float(hz_num.value))
            now = time.perf_counter()
            if now - last < 1.0 / hz:
                continue
            last = now
            set_frame((state["i"] + 1) % n)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
