# Atomic local map: meta + elevation + cost + normals (+ optional observed).
# Image convention (OccupancyGrid-style):
#   row 0, col 0 = cell at info.origin (min X, min Y)
#   world_x = origin.position.x + (col + 0.5) * resolution
#   world_y = origin.position.y + (row + 0.5) * resolution
# Robot-centered window, odom axes (not yaw-rotated).
std_msgs/Header header
nav_msgs/MapMetaData info
sensor_msgs/Image elevation   # 32FC1 meters (world Z)
sensor_msgs/Image costmap     # 32FC1, 0 free .. 1 lethal (semantic OR slope)
sensor_msgs/Image normals     # 32FC3 unit surface normals (nx, ny, nz)
sensor_msgs/Image observed    # 8UC1, 255 = TSDF-observed (pre-inpaint)
