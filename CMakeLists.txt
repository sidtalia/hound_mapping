cmake_minimum_required(VERSION 3.18)
project(hound_mapping)

enable_language(CUDA)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs photo)
find_package(CUDAToolkit REQUIRED)
find_package(glog REQUIRED)
find_package(gflags REQUIRED)
find_package(nvblox REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/LocalMap.msg"
  DEPENDENCIES std_msgs nav_msgs geometry_msgs sensor_msgs
)

add_library(hound_mapping_core
  src/mapper_core.cpp
  src/elevation_cost.cpp
  cuda/tsdf_color_to_bev.cu
)
target_include_directories(hound_mapping_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(hound_mapping_core
  nvblox::nvblox_lib
  ${OpenCV_LIBS}
  CUDA::cudart
)
target_compile_features(hound_mapping_core PUBLIC cxx_std_17)
set_target_properties(hound_mapping_core PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES "87"
)
target_compile_options(hound_mapping_core PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -O3>
)

add_executable(mapping_node src/mapping_node.cpp)
target_include_directories(mapping_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
ament_target_dependencies(mapping_node
  rclcpp
  sensor_msgs
  geometry_msgs
  nav_msgs
  std_msgs
  tf2
  tf2_ros
  tf2_eigen
)
target_link_libraries(mapping_node
  hound_mapping_core
  nvblox::nvblox_lib
  ${OpenCV_LIBS}
)
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(mapping_node "${cpp_typesupport_target}")
target_compile_features(mapping_node PUBLIC cxx_std_17)

add_executable(smoke_rail_mapper src/smoke_rail_mapper.cpp)
ament_target_dependencies(smoke_rail_mapper rclcpp)
target_link_libraries(smoke_rail_mapper
  hound_mapping_core
  nvblox::nvblox_lib
  ${OpenCV_LIBS}
)
target_compile_features(smoke_rail_mapper PUBLIC cxx_std_17)

install(TARGETS hound_mapping_core mapping_node smoke_rail_mapper
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(PROGRAMS
  scripts/generate_rail_dataset.py
  scripts/viser_rail_compare.py
  scripts/smoke_multicam_rail.sh
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY sim
  DESTINATION share/${PROJECT_NAME}
)
install(DIRECTORY include/
  DESTINATION include
)

ament_export_dependencies(
  rclcpp sensor_msgs geometry_msgs nav_msgs std_msgs
  tf2 tf2_ros tf2_eigen nvblox
)
ament_export_include_directories(include)
ament_package()
