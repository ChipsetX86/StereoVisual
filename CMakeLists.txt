cmake_minimum_required(VERSION 3.11)

project(StereoVisual LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

add_executable(StereoVisual main.cpp)

target_link_libraries(StereoVisual ${OpenCV_LIBS} pthread)