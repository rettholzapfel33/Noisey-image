cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

add_executable(yolo_test yolo_test.cpp)
target_link_libraries(yolo_test "${TORCH_LIBRARIES}")
set_property(TARGET yolo_test PROPERTY CXX_STANDARD 14)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( yolo_test ${OpenCV_LIBS} )