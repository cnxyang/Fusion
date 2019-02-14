cmake_minimum_required(VERSION 3.4)
project(slam VERSION 0.1.0 LANGUAGES CUDA CXX)
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
set(CMAKE_DISABLE_SOURCE_CHANGES  ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(Ceres 2.0 REQUIRED)
find_package(Eigen3 3.2 REQUIRED)
find_package(OpenCV 3.4 REQUIRED)
find_package(OpenGL 2.0 REQUIRED)
find_package(Pangolin REQUIRED)

add_subdirectory(slam_util)
add_subdirectory(slam_cuda)
add_subdirectory(slam_core)
add_subdirectory(slam_gui)
add_subdirectory(slam_app)