# Atomic local map: elevation + cost only.
# Image convention (OccupancyGrid-style):
#   row 0, col 0 = cell at info.origin (min X, min Y)
#   world_x = origin.position.x + (col + 0.5) * resolution
#   world_y = origin.position.y + (row + 0.5) * resolution
# Robot-centered window, odom axes (not yaw-rotated).
#
# costmap: IGHA free-ness, 255 = free, 0 = lethal.
#   Unobserved / no color uses SSoT nvblox.unobserved_luminance (default 128).
std_msgs/Header header
nav_msgs/MapMetaData info
sensor_msgs/Image elevation   # 32FC1 meters (world Z)
sensor_msgs/Image costmap     # 32FC1 free-ness [0,255]
