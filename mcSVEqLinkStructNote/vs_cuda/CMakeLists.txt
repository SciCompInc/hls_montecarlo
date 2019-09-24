cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(mc_pricer_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(../../libs ../../montecarlo/include ../src)

add_executable(mc_pricer_cuda pricer_kernel.cu test.cpp ../src/main.cpp ../src/pricer_host.cpp)

