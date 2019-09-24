cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(prj)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(../../libs)

add_executable(prj test.cpp kernel_wrapper.cpp)


