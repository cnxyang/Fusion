cmake_minimum_required(VERSION 2.8)
project(newsys CUDA CXX)

IF(NOT DEFINED CMAKE_CXX_STANDARD)
    SET(CMAKE_CXX_STANDARD 11)
    SET(CMAKE_CXX_STANDARD_REQUIRED TRUE)
ENDIF()

IF(NOT DEFINED CMAKE_CUDA_STANDARD)
    SET(CMAKE_CUDA_STANDARD 11)
    SET(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
ENDIF()

find_package(g2o REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CUDA 9.1 REQUIRED)
find_package(OpenCV 3.4 REQUIRED)

#message(FATAL_ERROR ${g2o_INCLUDE_DIR})

include_directories(cuda)
include_directories(util)
include_directories(include)
include_directories(/usr/include/openni2)
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

file(GLOB SRC_FILES src/*.cxx cuda/*.cu)
add_executable(newsys example/TumExample.cxx ${SRC_FILES})

target_link_libraries(newsys g2o_core g2o_stuff g2o_types_icp g2o_types_slam3d)
target_link_libraries(newsys ${OpenCV_LIBS})
target_link_libraries(newsys ${Pangolin_LIBRARY})
target_link_libraries(newsys ${OPENGL_gl_LIBRARY})
