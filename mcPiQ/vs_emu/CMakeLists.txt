cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(mc_pricer_emu)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(mc_pricer_emu ../src/pricer_kernel.cpp ../hls/pricer_kernel_wrapper.cpp ../src/test.cpp ../src/main.cpp ../src/pricer_host.cpp)

include_directories(../../libs ../../montecarlo/include ../src $ENV{VIVADO_HLS_ROOT}/include)



