#!/usr/bin/env python3
"""Author no-go polygons on a LayerCake top-down BEV (viser).

The overlay does not edit the cake. Mapping loads the yaml and ANDs it
onto LocalMap cost (0 = lethal, 1 = free). Empty path → all free.

  python3 .../author_lethal_map.py
  python3 .../author_lethal_map.py --layercake /root/colcon_ws/maps/hound_tsdf.layercake
  python3 .../author_lethal_map.py --elev /root/colcon_ws/maps/hound_tsdf.elev.yaml

Open the printed URL. Enable Draw, click vertices, Close polygon, Save.
Set nvblox.lethal_map_path to the written yaml and restart mapping.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

try:
    import yaml
except ImportError as exc:
    raise SystemExit("pip install pyyaml") from exc


def _ssot_nvblox() -> dict:
    ssot = os.environ.get("HOUND_SSOT", "")
    if not ssot or not Path(ssot).is_file():
        for cand in (
            Path("/root/colcon_ws/src/hound_core/config/SSoT.yaml"),
            Path.home() / "colcon_ws/src/hound_core/config/SSoT.yaml",
        ):
            if cand.is_file():
                ssot = str(cand)
                break
    if not ssot or not Path(ssot).is_file():
        return {}
    with open(ssot, encoding="utf-8") as handle:
        return (yaml.safe_load(handle) or {}).get("nvblox") or {}


def _default_cake() -> Path:
    nv = _ssot_nvblox()
    return Path(
        str(nv.get("layer_cake_path") or "/root/colcon_ws/maps/hound_tsdf.layercake")
    )


def _load_elev_yaml(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    width = int(data["width"])
    height = int(data["height"])
    res = float(data["resolution"])
    ox = float(data["origin_x"])
    oy = float(data["origin_y"])
    parent = path.parent

    def _bin(key: str) -> np.ndarray:
        name = str(data[key])
        p = Path(name)
        if not p.is_file():
            p = parent / name
        arr = np.fromfile(p, dtype=np.float32)
        if arr.size != width * height:
            raise ValueError(f"{p} has {arr.size} floats, want {width * height}")
        return arr.reshape(height, width)

    elev = _bin("elevation_f32")
    cost = _bin("cost_f32")
    meta = {
        "width": width,
        "height": height,
        "resolution": res,
        "origin_x": ox,
        "origin_y": oy,
        "frame_id": str(data.get("frame_id") or "map"),
    }
    return elev, cost, meta


def _find_layercake_to_bev() -> list[str] | None:
    which = shutil.which("layercake_to_bev")
    if which:
        return [which]
    roots = [Path("/root/colcon_ws/install"), Path.home() / "colcon_ws/install"]
    for parent in Path(__file__).resolve().parents:
        roots.append(parent / "install")
    for root in roots:
        binary = root / "hound_mapping/lib/hound_mapping/layercake_to_bev"
        if binary.is_file():
            return [str(binary)]
    if shutil.which("ros2"):
        return ["ros2", "run", "hound_mapping", "layercake_to_bev", "--"]
    return None


def _ensure_elev(cake: Path, elev: Path | None, nv: dict) -> Path:
    if elev is not None:
        if not elev.is_file():
            raise SystemExit(f"--elev not found: {elev}")
        return elev
    sibling = cake.with_suffix(".elev.yaml")
    if sibling.is_file():
        return sibling
    if not cake.is_file():
        raise SystemExit(
            f"LayerCake not found: {cake}\n"
            "Pass --elev <stem>.elev.yaml or save a cake first."
        )
    cmd = _find_layercake_to_bev()
    if not cmd:
        raise SystemExit(
            "Need layercake_to_bev (colcon build hound_mapping) or pass --elev."
        )
    res = float(nv.get("elevation_resolution", nv.get("voxel_size", 0.05)))
    voxel = float(nv.get("voxel_size", 0.05))
    robot_h = float(nv.get("robot_height_m", 0.4))
    bev_occ = float(nv.get("bev_occ_tsdf_max_m", 0.025))
    unobs = float(nv.get("unobserved_luminance", 252.0))
    argv = cmd + [
        "--layercake",
        str(cake),
        "--out",
        str(sibling),
        "--resolution",
        str(res),
        "--voxel-size",
        str(voxel),
        "--robot-height",
        str(robot_h),
        "--bev-occ-tsdf-max",
        str(bev_occ),
        "--unobserved-luminance",
        str(unobs),
    ]
    print("extracting BEV:", " ".join(argv), flush=True)
    proc = subprocess.run(argv, check=False)
    if proc.returncode != 0 or not sibling.is_file():
        raise SystemExit(f"layercake_to_bev failed (exit {proc.returncode})")
    return sibling


def _load_lethal(path: Path) -> tuple[np.ndarray | None, list[list[list[float]]], dict]:
    if not path.is_file():
        return None, [], {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    polys = data.get("polygons") or []
    cleaned: list[list[list[float]]] = []
    for poly in polys:
        verts = [[float(p[0]), float(p[1])] for p in poly if len(p) >= 2]
        if len(verts) >= 3:
            cleaned.append(verts)
    mask = None
    name = str(data.get("lethal_f32") or "")
    if name:
        bin_path = Path(name)
        if not bin_path.is_file():
            bin_path = path.parent / name
        w = int(data.get("width") or 0)
        h = int(data.get("height") or 0)
        if bin_path.is_file() and w > 0 and h > 0:
            arr = np.fromfile(bin_path, dtype=np.float32)
            if arr.size == w * h:
                mask = arr.reshape(h, w)
    meta = {
        "width": int(data.get("width") or 0),
        "height": int(data.get("height") or 0),
        "resolution": float(data.get("resolution") or 0.0),
        "origin_x": float(data.get("origin_x") or 0.0),
        "origin_y": float(data.get("origin_y") or 0.0),
        "frame_id": str(data.get("frame_id") or "map"),
    }
    return mask, cleaned, meta


def _world_to_px(
    verts: list[list[float]], origin_x: float, origin_y: float, res: float
) -> np.ndarray:
    pts = np.asarray(verts, dtype=np.float32)
    ix = np.floor((pts[:, 0] - origin_x) / res).astype(np.int32)
    iy = np.floor((pts[:, 1] - origin_y) / res).astype(np.int32)
    return np.stack([ix, iy], axis=-1)


def _fill_poly(mask: np.ndarray, verts_xy: np.ndarray, value: float) -> None:
    h, w = mask.shape
    try:
        import cv2

        cv2.fillPoly(mask, [verts_xy.astype(np.int32)], float(value))
        return
    except ImportError:
        pass
    from PIL import Image, ImageDraw

    im = Image.fromarray((mask * 255.0).astype(np.uint8), mode="L")
    draw = ImageDraw.Draw(im)
    draw.polygon([(int(x), int(y)) for x, y in verts_xy], fill=int(value * 255))
    filled = np.asarray(im, dtype=np.float32) / 255.0
    mask[:, :] = filled[:h, :w]


def rasterize(
    height: int,
    width: int,
    origin_x: float,
    origin_y: float,
    res: float,
    polygons: list[list[list[float]]],
    base: np.ndarray | None = None,
) -> np.ndarray:
    if base is not None and base.shape == (height, width):
        mask = np.array(base, dtype=np.float32, copy=True)
    else:
        mask = np.ones((height, width), dtype=np.float32)
    for poly in polygons:
        if len(poly) < 3:
            continue
        _fill_poly(mask, _world_to_px(poly, origin_x, origin_y, res), 0.0)
    return mask


def compose_rgb(
    cost: np.ndarray,
    elev: np.ndarray,
    mask: np.ndarray,
    draft: list[list[float]],
    origin_x: float,
    origin_y: float,
    res: float,
) -> np.ndarray:
    # Height shade so a flat/white cost (unobserved≈252) still shows terrain.
    e = elev.astype(np.float32)
    finite = np.isfinite(e)
    shade = np.full((*e.shape, 3), 28.0, dtype=np.float32)
    if np.any(finite):
        lo, hi = np.percentile(e[finite], [2.0, 98.0])
        if hi <= lo + 1e-4:
            hi = lo + 1.0
        t = np.clip((e - lo) / (hi - lo), 0.0, 1.0)
        shade[..., 0] = 35.0 + 90.0 * t
        shade[..., 1] = 45.0 + 100.0 * t
        shade[..., 2] = 55.0 + 80.0 * t
        shade[~finite] = 28.0

    # Cost: 0 lethal=red, 255 free=green. Mid/unobserved stays yellowish.
    c = np.clip(cost.astype(np.float32), 0.0, 255.0) / 255.0
    tint = np.empty((*c.shape, 3), dtype=np.float32)
    tint[..., 0] = 40.0 + 200.0 * (1.0 - c)
    tint[..., 1] = 30.0 + 200.0 * c
    tint[..., 2] = 25.0
    rgb = np.clip(0.4 * shade + 0.6 * tint, 0.0, 255.0).astype(np.uint8)
    lethal = mask <= 0.5
    rgb[lethal, 0] = 220
    rgb[lethal, 1] = 32
    rgb[lethal, 2] = 32
    if len(draft) >= 1:
        pts = _world_to_px(draft, origin_x, origin_y, res)
        try:
            import cv2

            if len(pts) >= 2:
                cv2.polylines(rgb, [pts], False, (255, 220, 40), 2, cv2.LINE_AA)
            for x, y in pts:
                cv2.circle(rgb, (int(x), int(y)), 3, (255, 255, 80), -1)
        except ImportError:
            for x, y in pts:
                if 0 <= y < rgb.shape[0] and 0 <= x < rgb.shape[1]:
                    rgb[int(y), int(x)] = (255, 255, 80)
    return rgb


def ray_plane_xy(origin, direction, z: float = 0.0):
    o = np.asarray(origin, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    if abs(d[2]) < 1e-9:
        return None
    t = (z - o[2]) / d[2]
    if t < 0:
        return None
    p = o + t * d
    return float(p[0]), float(p[1])


def write_lethal(
    path: Path,
    meta: dict,
    mask: np.ndarray,
    polygons: list[list[list[float]]],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name.endswith(".lethal.yaml"):
        bin_path = path.with_name(path.name[: -len(".lethal.yaml")] + ".lethal.f32")
    else:
        bin_path = path.with_suffix(".lethal.f32")
    np.ascontiguousarray(mask, dtype=np.float32).tofile(bin_path)
    payload = {
        "frame_id": meta.get("frame_id") or "map",
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "resolution": float(meta["resolution"]),
        "origin_x": float(meta["origin_x"]),
        "origin_y": float(meta["origin_y"]),
        "lethal_f32": bin_path.name,
        "polygons": [[[float(x), float(y)] for x, y in poly] for poly in polygons],
    }
    header = (
        "# Lethal overlay. Same XY frame as the LayerCake BEV.\n"
        "# mask: 1 = free, 0 = lethal. Out-of-bounds treated as free.\n"
        "# polygons: map-frame vertices (RViz). Mask grid is LayerCake/odom XY.\n"
        "# Authored by author_lethal_map.py / author_lethal_rviz.py.\n"
    )
    path.write_text(
        header + yaml.safe_dump(payload, sort_keys=False, default_flow_style=None),
        encoding="utf-8",
    )
    return path


def _bind_clicks(server, on_event) -> None:
    scene = server.scene
    # viser>=1: on_click() is a decorator. Older: on_click(cb).
    if hasattr(scene, "on_click"):
        bind = scene.on_click
        try:
            bind(on_event)
            return
        except TypeError:
            bind()(on_event)
            return
    if hasattr(scene, "on_pointer_event"):
        scene.on_pointer_event(event_type="click")(on_event)
        return
    if hasattr(server, "on_scene_pointer"):
        server.on_scene_pointer("click")(on_event)
        return
    raise SystemExit("viser build has no scene click API (need viser>=0.2)")


def _add_ground_image(server, name: str, image: np.ndarray, meta: dict):
    res = float(meta["resolution"])
    ox = float(meta["origin_x"])
    oy = float(meta["origin_y"])
    h, w = image.shape[:2]
    extent_x = w * res
    extent_y = h * res
    cx = ox + 0.5 * extent_x
    cy = oy + 0.5 * extent_y
    flipped = np.ascontiguousarray(image[::-1])
    add_image = getattr(server.scene, "add_image", None)
    if callable(add_image):
        try:
            return add_image(
                name,
                image=flipped,
                render_width=extent_x,
                render_height=extent_y,
                position=(cx, cy, 0.0),
            )
        except TypeError:
            return add_image(
                name,
                flipped,
                render_width=extent_x,
                render_height=extent_y,
                position=(cx, cy, 0.0),
            )
    try:
        import trimesh
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("viser scene.add_image missing; pip install trimesh pillow") from exc
    mesh = trimesh.creation.box(extents=(extent_x, extent_y, 1e-3))
    mesh.apply_translation((cx, cy, 0.0))
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    if len(mesh.vertices) >= 4:
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=np.repeat(uv, max(1, len(mesh.vertices) // 4), axis=0)[: len(mesh.vertices)],
            image=Image.fromarray(flipped),
        )
    return server.scene.add_mesh_trimesh(name, mesh)


def main(argv=None) -> int:
    nv = _ssot_nvblox()
    cake_default = _default_cake()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layercake", type=Path, default=cake_default)
    parser.add_argument(
        "--elev",
        type=Path,
        default=None,
        help="existing BEV yaml (default: <cake stem>.elev.yaml or extract)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="lethal yaml (default: nvblox.lethal_map_path or <cake stem>.lethal.yaml)",
    )
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args(argv)

    try:
        import viser
    except ImportError as exc:
        raise SystemExit("pip install viser") from exc

    elev_path = _ensure_elev(args.layercake, args.elev, nv)
    elev, cost, meta = _load_elev_yaml(elev_path)
    finite = np.isfinite(elev)
    print(
        f"BEV elev finite={int(np.count_nonzero(finite))}/{elev.size} "
        f"z=[{float(np.nanmin(elev)):.2f},{float(np.nanmax(elev)):.2f}] "
        f"cost min/p50/max="
        f"{float(np.min(cost)):.1f}/{float(np.median(cost)):.1f}/{float(np.max(cost)):.1f}",
        flush=True,
    )
    out = args.out
    if out is None:
        configured = str(nv.get("lethal_map_path") or "").strip()
        out = Path(configured) if configured else args.layercake.with_suffix(
            ".lethal.yaml"
        )

    existing_mask, polygons, _old = _load_lethal(out)
    if existing_mask is not None and existing_mask.shape != (meta["height"], meta["width"]):
        print(
            f"warning: existing mask {existing_mask.shape} != BEV "
            f"{meta['height']}x{meta['width']}; keeping polygons only",
            flush=True,
        )
        existing_mask = None

    mask = rasterize(
        meta["height"],
        meta["width"],
        meta["origin_x"],
        meta["origin_y"],
        meta["resolution"],
        polygons,
        base=None if polygons else existing_mask,
    )

    server = viser.ViserServer(host=args.host, port=args.port)
    server.gui.set_panel_label("lethal overlay")
    cx = meta["origin_x"] + 0.5 * meta["width"] * meta["resolution"]
    cy = meta["origin_y"] + 0.5 * meta["height"] * meta["resolution"]
    span = max(meta["width"], meta["height"]) * meta["resolution"]

    @server.on_client_connect
    def _on_connect(client) -> None:
        client.camera.position = (cx, cy, max(span, 8.0) * 1.15)
        client.camera.look_at = (cx, cy, 0.0)
        if hasattr(client.camera, "up_direction"):
            client.camera.up_direction = (0.0, 1.0, 0.0)

    state = {"draft": [], "polygons": list(polygons), "mask": mask}
    rgb0 = compose_rgb(
        cost, elev, mask, [], meta["origin_x"], meta["origin_y"],
        meta["resolution"],
    )
    img_handle = _add_ground_image(server, "/bev", rgb0, meta)
    draft_cloud = server.scene.add_point_cloud(
        "/draft_pts",
        points=np.zeros((1, 3), dtype=np.float32),
        colors=(255, 220, 40),
        point_size=max(0.08, meta["resolution"] * 2.0),
    )

    draw_cb = server.gui.add_checkbox("Draw (click adds vertex)", True)
    status = server.gui.add_markdown("")
    save_info = server.gui.add_markdown(f"out: `{out}`")

    def refresh(message: str = "") -> None:
        state["mask"] = rasterize(
            meta["height"],
            meta["width"],
            meta["origin_x"],
            meta["origin_y"],
            meta["resolution"],
            state["polygons"],
        )
        rgb = compose_rgb(
            cost,
            elev,
            state["mask"],
            state["draft"],
            meta["origin_x"],
            meta["origin_y"],
            meta["resolution"],
        )
        try:
            img_handle.image = np.ascontiguousarray(rgb[::-1])
        except Exception:
            pass
        if state["draft"]:
            pts = np.array(
                [[x, y, 0.04] for x, y in state["draft"]], dtype=np.float32
            )
            draft_cloud.points = pts
            draft_cloud.colors = np.tile(
                np.array([[255, 220, 40]], dtype=np.uint8), (len(pts), 1)
            )
        else:
            draft_cloud.points = np.zeros((1, 3), dtype=np.float32)
        n_lethal = int(np.count_nonzero(state["mask"] <= 0.5))
        status.content = (
            f"**{meta['width']}×{meta['height']}** @ {meta['resolution']:.3f} m  \n"
            f"polygons: **{len(state['polygons'])}**  "
            f"draft verts: **{len(state['draft'])}**  "
            f"lethal cells: **{n_lethal}**  \n"
            f"{message}"
        )

    def add_xy(x: float, y: float) -> None:
        ox, oy, res = meta["origin_x"], meta["origin_y"], meta["resolution"]
        if not (
            ox <= x < ox + meta["width"] * res
            and oy <= y < oy + meta["height"] * res
        ):
            refresh("click is outside the BEV")
            return
        state["draft"].append([x, y])
        refresh(f"vertex ({x:.2f}, {y:.2f})")

    def on_click(event) -> None:
        if not bool(draw_cb.value):
            return
        origin = getattr(event, "ray_origin", None)
        direction = getattr(event, "ray_direction", None)
        if origin is None or direction is None:
            return
        xy = ray_plane_xy(origin, direction, z=0.0)
        if xy is None:
            return
        add_xy(*xy)

    _bind_clicks(server, on_click)

    close_btn = server.gui.add_button("Close polygon")
    undo_v = server.gui.add_button("Undo vertex")
    undo_p = server.gui.add_button("Undo polygon")
    clear_btn = server.gui.add_button("Clear all")
    save_btn = server.gui.add_button("Save overlay")

    @close_btn.on_click
    def _close(_) -> None:
        if len(state["draft"]) < 3:
            refresh("need ≥3 vertices to close")
            return
        state["polygons"].append(list(state["draft"]))
        state["draft"] = []
        refresh("closed polygon")

    @undo_v.on_click
    def _undo_v(_) -> None:
        if state["draft"]:
            state["draft"].pop()
            refresh("undid vertex")
        else:
            refresh("no draft vertex")

    @undo_p.on_click
    def _undo_p(_) -> None:
        if state["polygons"]:
            state["polygons"].pop()
            refresh("undid polygon")
        else:
            refresh("no closed polygon")

    @clear_btn.on_click
    def _clear(_) -> None:
        state["polygons"] = []
        state["draft"] = []
        refresh("cleared")

    @save_btn.on_click
    def _save(_) -> None:
        if state["draft"] and len(state["draft"]) >= 3:
            state["polygons"].append(list(state["draft"]))
            state["draft"] = []
        state["mask"] = rasterize(
            meta["height"],
            meta["width"],
            meta["origin_x"],
            meta["origin_y"],
            meta["resolution"],
            state["polygons"],
        )
        written = write_lethal(out, meta, state["mask"], state["polygons"])
        save_info.content = f"wrote `{written}`"
        refresh(f"wrote {written}")
        print(f"wrote {written}", flush=True)

    refresh(
        f"BEV from {elev_path}. Enable Draw, click vertices, Close polygon, Save."
    )
    print(
        f"viser http://<host>:{args.port}/  BEV={elev_path}  out={out}",
        flush=True,
    )
    try:
        while True:
            time.sleep(0.25)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
