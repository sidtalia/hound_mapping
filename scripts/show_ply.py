#!/usr/bin/env python3
"""Publish a .ply as PointCloud2 (RViz2 has no Mesh display; huge TRIANGLE
Markers will not show).

  python3 .../show_ply.py /root/colcon_ws/maps/hound_tsdf_mesh.ply

RViz: Add → PointCloud2, topic /ply_cloud, RGB8, fixed frame odom.
If RViz has use_sim_time from a bag, uncheck it or this cloud is dropped.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path


def _parse_ply(path: Path):
    data = path.read_bytes()
    if not data.startswith(b"ply"):
        raise SystemExit(f"not a PLY: {path}")
    header_end = data.find(b"end_header")
    if header_end < 0:
        raise SystemExit("PLY missing end_header")
    # skip end_header line
    nl = data.find(b"\n", header_end)
    body = data[nl + 1 :]
    header = data[:header_end].decode("ascii", errors="replace")
    fmt = "ascii"
    n_vert = n_face = 0
    vert_props: list[tuple[str, str]] = []
    face_list = False
    section = None
    for line in header.splitlines():
        t = line.strip().split()
        if not t:
            continue
        if t[0] == "format":
            fmt = t[1]
        elif t[0] == "element" and t[1] == "vertex":
            section = "vertex"
            n_vert = int(t[2])
        elif t[0] == "element" and t[1] == "face":
            section = "face"
            n_face = int(t[2])
        elif t[0] == "property":
            if section == "vertex":
                vert_props.append((t[1], t[2]))
            elif section == "face" and t[1] == "list":
                face_list = True

    if fmt == "ascii":
        return _ascii(body.decode("ascii", errors="replace"), n_vert, n_face, vert_props)
    if "binary" not in fmt:
        raise SystemExit(f"unsupported PLY format {fmt}")
    le = "little" in fmt
    return _binary(body, n_vert, n_face, vert_props, face_list, le)


_SIZES = {
    "char": 1,
    "uchar": 1,
    "int8": 1,
    "uint8": 1,
    "short": 2,
    "ushort": 2,
    "int16": 2,
    "uint16": 2,
    "int": 4,
    "uint": 4,
    "int32": 4,
    "uint32": 4,
    "float": 4,
    "float32": 4,
    "double": 8,
    "float64": 8,
}
_UNPACK = {
    "char": "b",
    "int8": "b",
    "uchar": "B",
    "uint8": "B",
    "short": "h",
    "int16": "h",
    "ushort": "H",
    "uint16": "H",
    "int": "i",
    "int32": "i",
    "uint": "I",
    "uint32": "I",
    "float": "f",
    "float32": "f",
    "double": "d",
    "float64": "d",
}


def _ascii(text, n_vert, n_face, vert_props):
    names = [p[1] for p in vert_props]
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("comment")]
    verts = []
    colors = []
    i = 0
    for _ in range(n_vert):
        parts = lines[i].split()
        i += 1
        rec = {names[j]: float(parts[j]) for j in range(len(names))}
        verts.append((rec["x"], rec["y"], rec["z"]))
        if "red" in rec:
            colors.append((rec["red"] / 255.0, rec["green"] / 255.0, rec["blue"] / 255.0))
        else:
            colors.append((0.7, 0.7, 0.7))
    faces = []
    for _ in range(n_face):
        parts = list(map(int, lines[i].split()))
        i += 1
        n = parts[0]
        faces.append(tuple(parts[1 : 1 + n]))
    return verts, colors, faces


def _read(buf, off, fmt, le):
    endian = "<" if le else ">"
    size = struct.calcsize(endian + fmt)
    (val,) = struct.unpack_from(endian + fmt, buf, off)
    return val, off + size


def _binary(body, n_vert, n_face, vert_props, face_list, le):
    off = 0
    verts, colors = [], []
    for _ in range(n_vert):
        rec = {}
        for typ, name in vert_props:
            val, off = _read(body, off, _UNPACK[typ], le)
            rec[name] = val
        verts.append((float(rec["x"]), float(rec["y"]), float(rec["z"])))
        if "red" in rec:
            colors.append(
                (rec["red"] / 255.0, rec["green"] / 255.0, rec["blue"] / 255.0)
            )
        else:
            colors.append((0.7, 0.7, 0.7))
    faces = []
    for _ in range(n_face):
        n, off = _read(body, off, "B", le)
        idx = []
        for _j in range(n):
            v, off = _read(body, off, "i", le)
            idx.append(v)
        faces.append(tuple(idx))
    return verts, colors, faces


def _cloud_msg(verts, colors, frame, max_points: int, log: bool):
    import numpy as np
    from builtin_interfaces.msg import Time
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Header

    n = len(verts)
    step = max(1, (n + max_points - 1) // max_points)
    idx = np.arange(0, n, step, dtype=np.int64)
    xyz = np.asarray(verts, dtype=np.float64)[idx]
    rgb = np.asarray(colors, dtype=np.float64)[idx]
    if log:
        print(
            f"publishing {idx.size} / {n} verts (stride={step})  "
            f"bbox xyz [{xyz[:,0].min():.1f},{xyz[:,0].max():.1f}] "
            f"[{xyz[:,1].min():.1f},{xyz[:,1].max():.1f}] "
            f"[{xyz[:,2].min():.1f},{xyz[:,2].max():.1f}]"
        )
    ru = np.clip(rgb[:, 0] * 255.0, 0, 255).astype(np.uint32)
    gu = np.clip(rgb[:, 1] * 255.0, 0, 255).astype(np.uint32)
    bu = np.clip(rgb[:, 2] * 255.0, 0, 255).astype(np.uint32)
    packed = ((ru << 16) | (gu << 8) | bu).view(np.float32)
    raw = np.empty(
        idx.size,
        dtype=np.dtype(
            [("x", np.float32), ("y", np.float32), ("z", np.float32), ("rgb", np.float32)]
        ),
    )
    raw["x"] = xyz[:, 0]
    raw["y"] = xyz[:, 1]
    raw["z"] = xyz[:, 2]
    raw["rgb"] = packed
    msg = PointCloud2()
    # Zero stamp: RViz uses latest TF (bag use_sim_time otherwise hides this).
    msg.header = Header(stamp=Time(sec=0, nanosec=0), frame_id=frame)
    msg.height = 1
    msg.width = int(idx.size)
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = 16 * msg.width
    msg.is_dense = True
    msg.data = raw.tobytes()
    return msg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ply", type=Path)
    ap.add_argument("--frame", default="odom")
    ap.add_argument("--topic", default="/ply_cloud")
    ap.add_argument(
        "--max-points",
        type=int,
        default=400000,
        help="subsample vertices (4e6-face Marker will not fit in RViz)",
    )
    args = ap.parse_args()
    if not args.ply.is_file():
        print(f"missing {args.ply}", file=sys.stderr)
        return 1

    verts, colors, faces = _parse_ply(args.ply)
    print(f"loaded {len(verts)} verts {len(faces)} faces from {args.ply}")

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
    from sensor_msgs.msg import PointCloud2

    rclpy.init()
    node = Node("show_ply")
    qos = QoSProfile(
        depth=1,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
    )
    pub = node.create_publisher(PointCloud2, args.topic, qos)
    msg = _cloud_msg(verts, colors, args.frame, args.max_points, log=True)

    def publish():
        pub.publish(msg)

    publish()
    node.create_timer(1.0, publish)
    print(f"PointCloud2 on {args.topic}  frame={args.frame}")
    print("RViz: Add → PointCloud2 (not Marker). Color Transformer: RGB8. Size 0.05.")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
