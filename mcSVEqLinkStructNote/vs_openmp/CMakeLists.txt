cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(mc_pricer_openmp)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(mc_pricer_openmp pricer_kernel.cpp pricer_kernel_wrapper.cpp test.cpp ../src/main.cpp ../src/pricer_host.cpp)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(mc_pricer_openmp PUBLIC OpenMP::OpenMP_CXX)
endif()

include_directories(../../libs ../../montecarlo/include ../src)


