cmake_minimum_required(VERSION 3.11)
project(slams CUDA CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
set(CMAKE_DISABLE_SOURCE_CHANGES  ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(CUDA 9.2 REQUIRED)
find_package(Eigen3 3.2 REQUIRED)
find_package(OpenCV 3.4 REQUIRED)
find_package(OpenGL 2.0 REQUIRED)
find_package(Pangolin REQUIRED)
message(WARNING ${OpenCV_INCLUDE_DIRS})
add_executable(${PROJECT_NAME} "")

target_include_directories(${PROJECT_NAME}
PUBLIC
${CMAKE_CURRENT_SOURCE_DIR}/Core
${CMAKE_CURRENT_SOURCE_DIR}/GUI
${CMAKE_CURRENT_SOURCE_DIR}/Mapping
${CMAKE_CURRENT_SOURCE_DIR}/Optimization
${CMAKE_CURRENT_SOURCE_DIR}/Tracking
${CMAKE_CURRENT_SOURCE_DIR}/Utility
${CUDA_INCLUDE_DIRS}
/usr/local/include/opencv2/
)

target_sources(${PROJECT_NAME}
PRIVATE
GUI/Viewer.cc
Mapping/DeviceMap.cu
Mapping/FuseMap.cu
Mapping/Mapping.cc
Mapping/MeshScene.cu
Mapping/RenderScene.cu
Core/Frame.cc
Core/Camera.cc
Core/KeyFrame.cc
Core/System.cc
Optimization/Optimizer.cc
Optimization/Solver.cc
Tracking/KeyPoints.cu
Tracking/Pyrdown.cu
Tracking/Reduction.cu
Tracking/Tracking.cc
MainTum.cc
)

target_link_libraries(${PROJECT_NAME}
Eigen3::Eigen
${OpenCV_LIBRARIES}
${OpenGL_LIBRARIES}
${CUDA_LIBRARIES}
)
