cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(prj)

include_directories(../include ../../libs "$ENV{BOOST_ROOT}" "$ENV{QUANTLIB_ROOT}")

add_executable(prj 
main.cpp
quantlib_lib.cpp
test_brownian_bridge.cpp
test_normal_distribution.cpp
test_sobol_joe_kuo.cpp  
)


