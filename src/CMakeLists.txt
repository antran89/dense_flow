cmake_minimum_required(VERSION 2.8)
project(denseFlow)

# -- options
set(USE_LDOF OFF)
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "OpenCV found (${OpenCV_CONFIG_PATH})")

# ldof header
if(USE_LDOF)
    message(STATUS "Building with LDOF libs")
    link_directories(${CMAKE_CURRENT_SOURCE_DIR})
    list(APPEND External_LIBS libldof_gpu.so libldof.so)
endif()

add_executable(denseFlow_gpu denseFlow_gpu.cpp)
target_link_libraries(denseFlow_gpu ${OpenCV_LIBS} ${External_LIBS})

add_executable(denseFlow_gpu_with_segment denseFlow_gpu_with_segment.cpp)
target_link_libraries(denseFlow_gpu_with_segment ${OpenCV_LIBS} ${External_LIBS})

add_executable(denseFlow denseFlow.cpp)
target_link_libraries(denseFlow ${OpenCV_LIBS} ${External_LIBS})

add_executable(dense_color_flow dense_color_flow.cpp color_flow.cpp color_flow.hpp)
target_link_libraries(dense_color_flow ${OpenCV_LIBS} ${External_LIBS})
