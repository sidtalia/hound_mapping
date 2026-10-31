"""Heightfield → depth + traversability render (front RGBD on rails)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class CameraModel:
    width: int = 640
    height: int = 360
    fx: float = 400.0
    fy: float = 400.0
    cx: float = 320.0
    cy: float = 180.0
    max_depth_m: float = 6.0
    cam_height_m: float = 1.0
    ## Positive = look down (radians applied as body-Y pitch).
    pitch_down_deg: float = 15.0


def sample_elev(
    elev: np.ndarray, map_res: float, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Bilinear sample elevation; map origin (0,0) at cell (0,0), +x→cols, +y→rows."""
    h, w = elev.shape
    cols = x / map_res
    rows = y / map_res
    c0 = np.floor(cols).astype(np.int32)
    r0 = np.floor(rows).astype(np.int32)
    c1 = c0 + 1
    r1 = r0 + 1
    valid = (c0 >= 0) & (r0 >= 0) & (c1 < w) & (r1 < h)
    c0c = np.clip(c0, 0, w - 1)
    c1c = np.clip(c1, 0, w - 1)
    r0c = np.clip(r0, 0, h - 1)
    r1c = np.clip(r1, 0, h - 1)
    dc = cols - c0
    dr = rows - r0
    z00 = elev[r0c, c0c]
    z01 = elev[r0c, c1c]
    z10 = elev[r1c, c0c]
    z11 = elev[r1c, c1c]
    z = (
        z00 * (1 - dc) * (1 - dr)
        + z01 * dc * (1 - dr)
        + z10 * (1 - dc) * dr
        + z11 * dc * dr
    )
    out = np.full(x.shape, np.nan, dtype=np.float32)
    out[valid] = z[valid]
    return out


def sample_cost(
    cost: np.ndarray, map_res: float, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Nearest cost sample (IGHA* 0–255, 255=free)."""
    h, w = cost.shape
    cols = np.clip(np.round(x / map_res).astype(np.int32), 0, w - 1)
    rows = np.clip(np.round(y / map_res).astype(np.int32), 0, h - 1)
    return cost[rows, cols].astype(np.float32)


def R_world_body(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def R_pitch_down(pitch_down_deg: float) -> np.ndarray:
    """Body-frame pitch: +deg looks down (ROS x-fwd, y-left, z-up)."""
    th = np.deg2rad(pitch_down_deg)
    c, s = np.cos(th), np.sin(th)
    # Ry(+θ): forward (1,0,0) → (c, 0, -s) = nose down
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


# p_body = R_bo @ p_optical  (ROS optical: z fwd, x right, y down)
R_BODY_OPTICAL = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64
)


def camera_pose_world(
    x: float,
    y: float,
    yaw: float,
    elev: np.ndarray,
    map_res: float,
    cam_z_above: float,
    pitch_down_deg: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (t_world, R_world_optical) for camera above terrain."""
    z_g = float(sample_elev(elev, map_res, np.array([x]), np.array([y]))[0])
    if not np.isfinite(z_g):
        z_g = 0.0
    t = np.array([x, y, z_g + cam_z_above], dtype=np.float64)
    R_wo = R_world_body(yaw) @ R_pitch_down(pitch_down_deg) @ R_BODY_OPTICAL
    return t, R_wo


# Default multi-cam body yaw offsets (rad) relative to rail heading.
DEFAULT_VIEW_YAW_OFFSETS = {
    "front": 0.0,
    "left": np.deg2rad(90.0),
    "right": np.deg2rad(-90.0),
}


def render_depth_trav(
    elev: np.ndarray,
    cost: np.ndarray,
    map_res: float,
    x: float,
    y: float,
    yaw: float,
    cam: CameraModel,
    n_steps: int = 64,
    yaw_offset: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render depth (meters, Z in optical) and RGB traversability.

    Returns:
        depth: HxW float32, 0 = invalid
        trav_rgb: HxW x 3 uint8 (luminance from cost: free=white/green)
        T_wo: 4x4 world←optical
    """
    t, R_wo = camera_pose_world(
        x,
        y,
        yaw + float(yaw_offset),
        elev,
        map_res,
        cam.cam_height_m,
        pitch_down_deg=cam.pitch_down_deg,
    )
    H, W = cam.height, cam.width
    us = np.arange(W, dtype=np.float64)
    vs = np.arange(H, dtype=np.float64)
    uu, vv = np.meshgrid(us, vs)
    x_c = (uu - cam.cx) / cam.fx
    y_c = (vv - cam.cy) / cam.fy
    rays_o = np.stack([x_c, y_c, np.ones_like(x_c)], axis=-1)
    rays_o /= np.linalg.norm(rays_o, axis=-1, keepdims=True)
    rays_w = rays_o @ R_wo.T  # (H,W,3)

    ts = np.linspace(0.05, cam.max_depth_m, n_steps, dtype=np.float64)
    depth = np.zeros((H, W), dtype=np.float32)
    trav = np.zeros((H, W, 3), dtype=np.uint8)
    hit = np.zeros((H, W), dtype=bool)

    for t_ray in ts:
        pts = t[None, None, :] + t_ray * rays_w
        zx = sample_elev(elev, map_res, pts[..., 0], pts[..., 1])
        # Hit when ray goes at/under the surface (and was above before).
        below = np.isfinite(zx) & (pts[..., 2] <= zx + 0.02)
        new_hit = below & ~hit
        if not np.any(new_hit):
            continue
        # Optical depth ≈ Z component along optical axis = t_ray * ray_o_z
        # Prefer metric range along ray projected to optical Z:
        z_opt = (t_ray * rays_o[..., 2]).astype(np.float32)
        depth[new_hit] = z_opt[new_hit]
        cval = sample_cost(cost, map_res, pts[..., 0], pts[..., 1])
        # IGHA* cost 0–255 free=255 → paint green-white
        g = np.clip(cval, 0, 255).astype(np.uint8)
        trav[new_hit, 0] = (g[new_hit] // 4).astype(np.uint8)  # R low
        trav[new_hit, 1] = g[new_hit]
        trav[new_hit, 2] = (g[new_hit] // 4).astype(np.uint8)
        hit |= new_hit
        if hit.all():
            break

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_wo
    T[:3, 3] = t
    return depth, trav, T


def synthetic_stereo_ir(trav_rgb: np.ndarray, depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fake stereo IR for capture-contract completeness (not for cuVSLAM smoke).

    Left = luminance of trav; right = left with a tiny horizontal shift where depth is valid.
    """
    if trav_rgb.ndim == 3:
        left = (
            0.299 * trav_rgb[..., 0].astype(np.float32)
            + 0.587 * trav_rgb[..., 1].astype(np.float32)
            + 0.114 * trav_rgb[..., 2].astype(np.float32)
        ).astype(np.uint8)
    else:
        left = trav_rgb.astype(np.uint8)
    right = np.roll(left, shift=2, axis=1)
    right[:, :2] = left[:, :2]
    right[depth <= 0] = 0
    left[depth <= 0] = 0
    return left, right


def densify_path(
    path_xy_yaw: np.ndarray, step_m: float = 0.25
) -> np.ndarray:
    """path (N,>=3) start→goal → denser (M,3) x,y,yaw by arc length."""
    if len(path_xy_yaw) < 2:
        return path_xy_yaw[:, :3].astype(np.float64)
    out = [path_xy_yaw[0, :3].astype(np.float64)]
    for i in range(len(path_xy_yaw) - 1):
        a = path_xy_yaw[i, :3].astype(np.float64)
        b = path_xy_yaw[i + 1, :3].astype(np.float64)
        dxy = b[:2] - a[:2]
        dist = float(np.linalg.norm(dxy))
        n = max(1, int(np.ceil(dist / step_m)))
        for k in range(1, n + 1):
            t = k / n
            p = (1 - t) * a + t * b
            if dist > 1e-3:
                p[2] = np.arctan2(dxy[1], dxy[0])
            out.append(p)
    return np.asarray(out, dtype=np.float64)


def densify_path_timed(
    path: np.ndarray, dt: float = 0.05, speed_col: int = 3
) -> np.ndarray:
    """Resample path at fixed dt using speed (m/s) in column `speed_col`.

    path (N,>=4) start→goal with x,y,yaw,speed[,...]. Returns (M,3) x,y,yaw.
    IGHA* kinodynamic paths are typically ~0.05 s between waypoints.
    """
    if len(path) < 2:
        return path[:, :3].astype(np.float64)
    out = [path[0, :3].astype(np.float64)]
    for i in range(len(path) - 1):
        a = path[i].astype(np.float64)
        b = path[i + 1].astype(np.float64)
        dxy = b[:2] - a[:2]
        dist = float(np.linalg.norm(dxy))
        v = float(a[speed_col]) if path.shape[1] > speed_col else 0.0
        if not np.isfinite(v) or v < 0.05:
            v = max(dist / max(dt, 1e-3), 0.05)
        seg_t = dist / v
        n = max(1, int(np.ceil(seg_t / dt)))
        for k in range(1, n + 1):
            t = k / n
            p = (1 - t) * a[:3] + t * b[:3]
            if dist > 1e-3:
                p[2] = np.arctan2(dxy[1], dxy[0])
            out.append(p)
    return np.asarray(out, dtype=np.float64)


def slope_cost_from_elev(elev: np.ndarray, map_res: float, max_deg: float = 25.0) -> np.ndarray:
    """IGHA*-style 0–255 cost from slope (255 free)."""
    dzdy, dzdx = np.gradient(elev.astype(np.float64), map_res)
    nz = 1.0
    nrm = np.sqrt(dzdx * dzdx + dzdy * dzdy + nz * nz)
    cos_n = nz / (nrm + 1e-9)
    thresh = np.cos(np.deg2rad(max_deg))
    free = cos_n >= thresh
    cost = np.zeros(elev.shape, dtype=np.float32)
    cost[free] = 255.0
    return cost
