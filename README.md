# hound_mapping

nvblox **TSDF** integrate + GPU **TSDF→BEV** elevation / cost / normals for HOUND.
Publishes `LocalMap` (robot-centered, odom axes — not yaw-rotated). No ESDF layer.

## Hotpath

```
depth/lidar/color → MapperCore (TSDF+color)
  → layer snapshot (extract worker)
  → block-scatter TSDF→BEV
  → low-res Telea inpaint (unknowns)
  → GPU normals + slope OR into cost
  → ~/local_map  (+ ~/elev_color viz)
```

Slope lethality (`lethal_slope_deg`) is applied here; nav consumes the costmap as-is.

## Layout

```
hound_mapping/
  src/           mapping_node, mapper_core, elevation_cost, smoke_rail_mapper
  cuda/          tsdf_color_to_bev.cu  (scatter + slope/normals)
  include/       MapperCore, ElevationCostExtractor, TsdfColorToBev
  msg/           LocalMap.msg
  scripts/       generate_rail_dataset.py, viser_rail_compare.py
  sim/           heightfield camera + rail dataset I/O
  data/          rail_sim datasets (built locally; gitignored)
```

Config / bring-up: `hound_core` (`SSoT.yaml` → `nvblox.*`, launch `backend: hound`).

## Build

```bash
cd /root/colcon_ws && colcon build --packages-select hound_mapping --symlink-install
source install/setup.bash
```

## Smoke (rail replay)

```bash
ros2 run hound_mapping smoke_rail_mapper --ros-args \
  -p dataset:=/root/colcon_ws/src/hound_mapping/data/rail_sim/race-2
```

Viser compare:

```bash
python3 scripts/viser_rail_compare.py --dataset data/rail_sim/race-2 --port 8081
```

## Topics

| Topic | Type | Notes |
|-------|------|--------|
| `~/local_map` | `LocalMap` | elev, costmap `[0,1]`, normals, observed |
| `~/elev_color` | `Image` bgr8 | luminance viz (debug) |
| `~/extract_timing_ms` | `Float64MultiArray` | `[bev, snapshot, inpaint, slope, colorize, total]` ms |
