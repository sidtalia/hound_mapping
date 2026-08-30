#!/usr/bin/env python3
"""Show a triangle mesh in RViz as Marker MESH_RESOURCE.

  python3 .../show_mesh.py /root/colcon_ws/maps/hound_tsdf_mesh_pf.obj

RViz: Add → Marker, topic /ply_mesh. Fixed Frame = map (this OBJ is in map).
Need map←odom on TF or the mesh will sit in the wrong place vs odom clouds.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "mesh",
        type=Path,
        nargs="?",
        default=Path("/root/colcon_ws/maps/hound_tsdf_mesh_pf.obj"),
    )
    ap.add_argument("--frame", default="map")
    ap.add_argument("--topic", default="/ply_mesh")
    ap.add_argument("--alpha", type=float, default=0.7)
    args = ap.parse_args()
    mesh = args.mesh.resolve()
    if not mesh.is_file():
        print(f"missing {mesh}", file=sys.stderr)
        return 1

    import rclpy
    from geometry_msgs.msg import Vector3
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
    from builtin_interfaces.msg import Time
    from std_msgs.msg import ColorRGBA, Header
    from visualization_msgs.msg import Marker

    rclpy.init()
    node = Node("show_mesh")
    qos = QoSProfile(
        depth=1,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
    )
    pub = node.create_publisher(Marker, args.topic, qos)
    msg = Marker()
    msg.header = Header(stamp=Time(sec=0, nanosec=0), frame_id=args.frame)
    msg.ns = "mesh_pf"
    msg.id = 0
    msg.type = Marker.MESH_RESOURCE
    msg.action = Marker.ADD
    msg.pose.orientation.w = 1.0
    msg.scale = Vector3(x=1.0, y=1.0, z=1.0)
    msg.color = ColorRGBA(r=0.75, g=0.78, b=0.82, a=float(args.alpha))
    msg.mesh_resource = "file://" + str(mesh)
    msg.mesh_use_embedded_materials = False
    msg.frame_locked = True

    def publish():
        pub.publish(msg)

    publish()
    node.create_timer(1.0, publish)
    print(f"Marker MESH_RESOURCE on {args.topic}  frame={args.frame}")
    print(f"  {msg.mesh_resource}")
    print("RViz: Marker → /ply_mesh. Fixed Frame: map.")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
