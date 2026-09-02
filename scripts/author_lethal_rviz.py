#!/usr/bin/env python3
"""Author no-go polygons in RViz (map-frame mesh + 2D Pose Estimate).

  ros2 launch hound_core hound_author_lethal.launch.py

Opens a dedicated RViz config. Mesh, clicks, and overlay grid are all map.
No bag / TF required.

Undo last vertex / open polygon:
  ros2 topic pub --once /lethal_author/undo std_msgs/msg/Empty {}
"""

from __future__ import annotations

import math
import signal
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from author_lethal_map import (  # noqa: E402
    _default_cake,
    _ensure_elev,
    _load_elev_yaml,
    _load_lethal,
    _ssot_nvblox,
    rasterize,
    write_lethal,
)

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Point, PoseStamped, PoseWithCovarianceStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import ColorRGBA, Empty
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray
import tf2_geometry_msgs  # noqa: F401


def _ssot() -> dict:
    env = __import__("os").environ.get("HOUND_SSOT", "")
    cands = []
    if env:
        cands.append(Path(env))
    cands.extend(
        [
            Path("/root/colcon_ws/src/hound_core/config/SSoT.yaml"),
            Path.home() / "colcon_ws/src/hound_core/config/SSoT.yaml",
        ]
    )
    for p in cands:
        if p.is_file():
            with p.open(encoding="utf-8") as handle:
                return yaml.safe_load(handle) or {}
    return {}


class LethalAuthorRviz(Node):
    def __init__(self) -> None:
        super().__init__("lethal_author")
        ssot = _ssot()
        nv = _ssot_nvblox() or dict(ssot.get("nvblox") or {})
        mesh_pf = dict(ssot.get("mesh_pf") or {})

        self.declare_parameter("click_topic", "/initialpose")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("grid_frame_id", "map")
        self.declare_parameter("close_radius_m", 1.0)
        self.declare_parameter("mesh_file", str(mesh_pf.get("map_file") or ""))
        self.declare_parameter("mesh_topic", "/ply_mesh")
        self.declare_parameter("mesh_alpha", 0.7)
        cake = _default_cake()
        self.declare_parameter("layercake", str(cake))
        configured = str(nv.get("lethal_map_path") or "").strip()
        default_out = configured or str(cake.with_suffix(".lethal.yaml"))
        self.declare_parameter("out", default_out)

        self._frame = str(self.get_parameter("frame_id").value)
        self._grid_frame = str(self.get_parameter("grid_frame_id").value)
        self._close_r = float(self.get_parameter("close_radius_m").value)
        self._out = Path(str(self.get_parameter("out").value))
        self._dirty = False

        elev_path = _ensure_elev(Path(str(self.get_parameter("layercake").value)), None, nv)
        _, _, self._meta = _load_elev_yaml(elev_path)
        existing_mask, self._polygons, old_meta = _load_lethal(self._out)
        if old_meta.get("width"):
            self._meta.update({k: old_meta[k] for k in ("width", "height", "resolution", "origin_x", "origin_y") if old_meta.get(k)})
        self._mask = existing_mask
        self._draft: list[list[float]] = []

        self._tf = Buffer()
        self._tf_listener = TransformListener(self._tf, self)

        latch = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._mesh_pub = self.create_publisher(
            Marker, str(self.get_parameter("mesh_topic").value), latch
        )
        self._mark_pub = self.create_publisher(MarkerArray, "~/markers", latch)
        self._mesh_msg = self._make_mesh_marker()
        self.create_subscription(
            PoseWithCovarianceStamped,
            str(self.get_parameter("click_topic").value),
            self._on_initialpose,
            10,
        )
        self.create_subscription(Empty, "~/undo", self._on_undo, 10)
        self.create_subscription(Empty, "~/clear", self._on_clear, 10)
        self.create_service(Trigger, "~/save", self._on_save_srv)
        self.create_timer(1.0, self._republish)
        self._publish_mesh()
        self._publish_markers()
        self.get_logger().info(
            f"lethal RViz author: mesh={self.get_parameter('mesh_file').value} "
            f"frame={self._frame} clicks=/initialpose close={self._close_r:.1f}m "
            f"out={self._out} n_poly={len(self._polygons)}"
        )

    def _make_mesh_marker(self) -> Marker | None:
        mesh = Path(str(self.get_parameter("mesh_file").value)).expanduser()
        if not mesh.is_file():
            self.get_logger().error(f"mesh not found: {mesh}")
            return None
        msg = Marker()
        msg.header.stamp.sec = 0
        msg.header.stamp.nanosec = 0
        msg.header.frame_id = self._frame
        msg.ns = "map_mesh"
        msg.id = 0
        msg.type = Marker.MESH_RESOURCE
        msg.action = Marker.ADD
        msg.pose.orientation.w = 1.0
        msg.scale.x = msg.scale.y = msg.scale.z = 1.0
        a = float(self.get_parameter("mesh_alpha").value)
        msg.color = ColorRGBA(r=0.75, g=0.78, b=0.82, a=a)
        msg.mesh_resource = "file://" + str(mesh.resolve())
        msg.mesh_use_embedded_materials = False
        msg.frame_locked = True
        return msg

    def _publish_mesh(self) -> None:
        if self._mesh_msg is not None:
            self._mesh_pub.publish(self._mesh_msg)

    def _transform_xy(self, x: float, y: float, src: str, dst: str) -> tuple[float, float] | None:
        src = (src or dst).strip()
        if not src or src == dst:
            return x, y
        ps = PoseStamped()
        ps.header.frame_id = src
        ps.header.stamp = Time().to_msg()
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.orientation.w = 1.0
        try:
            out = self._tf.transform(ps, dst, timeout=Duration(seconds=0.3))
        except TransformException as exc:
            self.get_logger().warn(
                f"click frame={src!r} (need {dst}); no TF ({exc}) — using xy as map"
            )
            return x, y
        return float(out.pose.position.x), float(out.pose.position.y)

    def _on_initialpose(self, msg: PoseWithCovarianceStamped) -> None:
        src = str(msg.header.frame_id or self._frame).strip()
        xy = self._transform_xy(
            float(msg.pose.pose.position.x),
            float(msg.pose.pose.position.y),
            src,
            self._frame,
        )
        if xy is None:
            return
        x, y = xy
        if self._draft:
            dx = x - self._draft[0][0]
            dy = y - self._draft[0][1]
            if len(self._draft) >= 3 and math.hypot(dx, dy) <= self._close_r:
                self._polygons.append(list(self._draft))
                self._draft.clear()
                self._dirty = True
                self._rebuild_mask()
                self.save()
                self.get_logger().info(
                    f"closed polygon → {len(self._polygons)} (snapped to first vertex)"
                )
                self._publish_markers()
                return
            lx, ly = self._draft[-1]
            if math.hypot(x - lx, y - ly) <= self._close_r:
                self.get_logger().warn("ignored click (within 1 m of last vertex)")
                return
        self._draft.append([x, y])
        self._dirty = True
        self._publish_markers()
        self.get_logger().info(
            f"vertex {len(self._draft)}: ({x:.2f}, {y:.2f}) {self._frame} "
            f"(click first vertex ±{self._close_r:.1f}m to close)"
        )

    def _on_undo(self, _msg: Empty) -> None:
        if self._draft:
            self._draft.pop()
            self.get_logger().info(f"undo vertex → {len(self._draft)} in draft")
        elif self._polygons:
            self._polygons.pop()
            self._rebuild_mask()
            self.get_logger().info(f"undo polygon → {len(self._polygons)}")
        self._dirty = True
        self._publish_markers()

    def _on_clear(self, _msg: Empty) -> None:
        self._draft.clear()
        self._polygons.clear()
        self._mask = None
        self._dirty = True
        self._publish_markers()
        self.get_logger().info("cleared all polygons")

    def _on_save_srv(self, _req: Trigger.Request, res: Trigger.Response):
        ok = self.save()
        res.success = ok
        res.message = str(self._out) if ok else "save failed"
        return res

    def _rebuild_mask(self) -> bool:
        self._mask = rasterize(
            int(self._meta["height"]),
            int(self._meta["width"]),
            float(self._meta["origin_x"]),
            float(self._meta["origin_y"]),
            float(self._meta["resolution"]),
            self._polygons,
        )
        return True

    def save(self) -> bool:
        if self._draft:
            self.get_logger().warn("open polygon not saved (close it first)")
        if not self._polygons:
            self.get_logger().warn("no closed polygons — not saving")
            return False
        if not self._rebuild_mask() or self._mask is None:
            return False
        meta = dict(self._meta)
        meta["frame_id"] = self._frame
        write_lethal(self._out, meta, self._mask, self._polygons)
        self._dirty = False
        self.get_logger().info(f"saved {len(self._polygons)} polygons → {self._out}")
        return True

    def shutdown_save(self) -> None:
        if self._dirty and self._polygons:
            self.save()

    def _line(self, ns: str, mid: int, verts: list[list[float]], rgba, closed: bool) -> Marker:
        m = Marker()
        m.header.frame_id = self._frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = ns
        m.id = mid
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.12
        m.color = ColorRGBA(r=rgba[0], g=rgba[1], b=rgba[2], a=rgba[3])
        m.pose.orientation.w = 1.0
        m.frame_locked = True
        pts = list(verts)
        if closed and pts:
            pts = pts + [pts[0]]
        for x, y in pts:
            m.points.append(Point(x=float(x), y=float(y), z=0.15))
        return m

    def _spheres(self, ns: str, verts: list[list[float]], rgba, start_id: int) -> list[Marker]:
        out = []
        for i, (x, y) in enumerate(verts):
            m = Marker()
            m.header.frame_id = self._frame
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = ns
            m.id = start_id + i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.scale.x = m.scale.y = m.scale.z = 0.35
            m.color = ColorRGBA(r=rgba[0], g=rgba[1], b=rgba[2], a=rgba[3])
            m.pose.position.x = float(x)
            m.pose.position.y = float(y)
            m.pose.position.z = 0.2
            m.pose.orientation.w = 1.0
            m.frame_locked = True
            out.append(m)
        return out

    def _publish_markers(self) -> None:
        arr = MarkerArray()
        wipe = Marker()
        wipe.action = Marker.DELETEALL
        arr.markers.append(wipe)
        for i, poly in enumerate(self._polygons):
            arr.markers.append(self._line("closed", i, poly, (0.9, 0.15, 0.1, 0.95), True))
        if self._draft:
            arr.markers.append(
                self._line("draft", 0, self._draft, (1.0, 0.85, 0.1, 1.0), False)
            )
            arr.markers.extend(self._spheres("draft_pt", self._draft, (1.0, 0.95, 0.2, 1.0), 0))
        self._mark_pub.publish(arr)

    def _republish(self) -> None:
        self._publish_mesh()
        self._publish_markers()


def main() -> int:
    rclpy.init()
    node = LethalAuthorRviz()
    done = {"v": False}

    def _sig(_s, _f):
        if done["v"]:
            return
        done["v"] = True
        node.shutdown_save()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if not done["v"]:
            done["v"] = True
            try:
                node.shutdown_save()
            except Exception:
                pass
            try:
                node.destroy_node()
            except Exception:
                pass
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
