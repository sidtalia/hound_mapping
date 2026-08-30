#!/usr/bin/env python3
"""Write the next LocalMap (elevation + cost) and exit.

  python3 .../save_elevation.py
  # optional explicit topic:
  python3 .../save_elevation.py --topic /debug/hound_mapping/local_map

Keep a node and write on demand:

  python3 .../save_elevation.py --serve
  ros2 service call /save_elevation/save std_srvs/srv/Trigger
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_srvs.srv import Trigger

from hound_mapping.msg import LocalMap


def _ssot_nvblox() -> dict:
    ssot = os.environ.get("HOUND_SSOT", "")
    if not ssot or not Path(ssot).is_file():
        return {}
    import yaml

    with open(ssot, encoding="utf-8") as handle:
        return (yaml.safe_load(handle) or {}).get("nvblox") or {}


def _default_topics() -> list[str]:
    nv = _ssot_nvblox()
    prefix = str(nv.get("bag_replay_topic_prefix") or "debug").strip().strip("/")
    topics = [
        "/hound_mapping/local_map",
        f"/{prefix}/hound_mapping/local_map" if prefix else "",
        "/debug/hound_mapping/local_map",
    ]
    out: list[str] = []
    for t in topics:
        if t and t not in out:
            out.append(t)
    return out


def _default_out() -> str:
    nv = _ssot_nvblox()
    cake = str(nv.get("layer_cake_path") or "/root/colcon_ws/maps/hound_tsdf.layercake")
    out = str(nv.get("elevation_map_path") or "").strip()
    if out:
        return out
    return str(Path(cake).with_suffix(".elev.yaml"))


def _image_f32(img) -> np.ndarray:
    if img.height == 0 or img.width == 0:
        raise ValueError("empty image")
    row = max(img.width, img.step // 4)
    arr = np.frombuffer(bytes(img.data), dtype=np.float32)
    return np.ascontiguousarray(arr.reshape(img.height, row)[:, : img.width])


def write_local_map(msg: LocalMap, yaml_path: Path) -> Path:
    elev = _image_f32(msg.elevation)
    cost = _image_f32(msg.costmap)
    yaml_path = Path(yaml_path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    base = yaml_path.with_suffix("")
    if base.suffix == ".elev":
        base = base.with_suffix("")
    elev_bin = Path(str(base) + ".elev.f32")
    cost_bin = Path(str(base) + ".cost.f32")
    elev.tofile(elev_bin)
    cost.tofile(cost_bin)
    stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    origin = msg.info.origin.position
    yaml_path.write_text(
        "# LocalMap snapshot from topic (same grid as ~/local_map).\n"
        f"frame_id: {msg.header.frame_id}\n"
        f"stamp_sec: {stamp:.9f}\n"
        f"width: {msg.info.width}\n"
        f"height: {msg.info.height}\n"
        f"resolution: {msg.info.resolution}\n"
        f"origin_x: {origin.x}\n"
        f"origin_y: {origin.y}\n"
        f"origin_z: {origin.z}\n"
        f"elevation_f32: {elev_bin.name}\n"
        f"cost_f32: {cost_bin.name}\n",
        encoding="utf-8",
    )
    return yaml_path


class SaveElevation(Node):
    def __init__(self, topics: list[str], out: str, once: bool) -> None:
        super().__init__("save_elevation")
        self._out = Path(out)
        self._once = once
        self._last: LocalMap | None = None
        self._subs: dict[str, object] = {}
        self._qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        for topic in topics:
            self._subscribe(topic)
        self._discover_topics()
        if not once:
            self.create_service(Trigger, "~/save", self._on_save)
        names = ", ".join(sorted(self._subs)) or "(none yet)"
        if once:
            self.get_logger().info(
                f"waiting for LocalMap on {names} → write {self._out} and exit"
            )
            self.create_timer(2.0, self._on_watchdog)
        else:
            self.get_logger().info(
                f"latching {names} → {self._out} "
                "(call /save_elevation/save to write)"
            )

    def _subscribe(self, topic: str) -> None:
        if not topic or topic in self._subs:
            return
        self._subs[topic] = self.create_subscription(
            LocalMap, topic, self._on_map, self._qos
        )

    def _discover_topics(self) -> None:
        for name, types in self.get_topic_names_and_types():
            if not name.endswith("/hound_mapping/local_map"):
                continue
            if any("LocalMap" in t for t in types):
                self._subscribe(name)

    def _on_watchdog(self) -> None:
        if self._last is not None:
            return
        before = set(self._subs)
        self._discover_topics()
        added = set(self._subs) - before
        if added:
            self.get_logger().info(
                "found LocalMap: " + ", ".join(sorted(added))
            )
        elif not self._subs:
            self.get_logger().warn("no */hound_mapping/local_map on the graph yet")
        else:
            self.get_logger().warn(
                "no LocalMap yet on " + ", ".join(sorted(self._subs))
            )

    def _on_map(self, msg: LocalMap) -> None:
        self._last = msg
        if self._once:
            path = write_local_map(msg, self._out)
            self.get_logger().info(f"wrote {path}")
            rclpy.shutdown()

    def _on_save(self, _req, res):
        if self._last is None:
            res.success = False
            res.message = "no LocalMap yet"
            return res
        path = write_local_map(self._last, self._out)
        res.success = True
        res.message = f"wrote {path}"
        self.get_logger().info(res.message)
        return res


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topic",
        default="",
        help="LocalMap topic (default: live + /debug + graph discovery)",
    )
    parser.add_argument("--out", default=_default_out())
    parser.add_argument(
        "--once",
        action="store_true",
        default=True,
        help="write the next LocalMap and exit (default)",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="keep running; write only on /save_elevation/save",
    )
    args, ros_args = parser.parse_known_args(argv)
    once = not bool(args.serve)
    rclpy.init(args=ros_args)
    topics = [args.topic] if args.topic else _default_topics()
    node = SaveElevation(topics, args.out, once)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
