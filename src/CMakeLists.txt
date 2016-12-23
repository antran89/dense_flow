cmake_minimum_required(VERSION 2.8)
project( denseFlow )

set( CMAKE_BUILD_TYPE Release )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# ldof header
include_directories( ~/Desktop/opencv-workspace/ldof_opencv/src )
link_directories(~/local/lib)

# external dependencies
list( APPEND External_LIBS libldof_gpu.so libldof.so )

add_executable( denseFlow_gpu denseFlow_gpu.cpp )
target_link_libraries( denseFlow_gpu ${OpenCV_LIBS} ${External_LIBS} )

add_executable( denseFlow denseFlow.cpp )
target_link_libraries( denseFlow ${OpenCV_LIBS} ${External_LIBS} )

add_executable(dense_color_flow dense_color_flow.cpp color_flow.cpp color_flow.hpp)
target_link_libraries(dense_color_flow ${OpenCV_LIBS} ${External_LIBS})
