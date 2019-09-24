##############################################
# Project settings

set CFLAGS_K "-D__GMP_WITHIN_CONFIGURE -I ../../montecarlo/include -I ../src"
set CFLAGS_TB "-I ../src -I ../../libs -I ../../montecarlo/include"

set ::env(SCI_DATAFILE) input_data.json

# Create a project
open_project	-reset prj

# The source file and test bench
add_files	"../src/pricer_kernel.cpp" -cflags $CFLAGS_K
add_files -tb	"../src/pricer_host.cpp pricer_kernel_wrapper.cpp ../src/main.cpp ../src/test.cpp" -cflags $CFLAGS_TB
add_files -tb   "../input_data.json"

# Specify the top-level function for synthesis
set_top		pricer_kernel

###########################
# Solution settings

# Create solution1
open_solution -reset solution1

# Specify a Xilinx device and clock period
# - Do not specify a clock uncertainty (margin)
# - Let the  margin to default to 12.5% of clock period
#set_part {xcvu9p-flgb2104-2-i} -tool vivado
set_part {xcu200-fsgd2104-2-e} -tool vivado
create_clock -period 4 -name default


# Simulate the C code 
csim_design

# Do not perform any other steps
# - The basic project will be opened in the GUI 
exit

