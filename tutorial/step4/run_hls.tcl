# Create a project
open_project	-reset prj

# The source file and test bench
add_files	"kernel.cpp" -cflags "-I ../../montecarlo/include"
add_files -tb	"host.cpp kernel_wrapper.cpp" -cflags "-I ../../montecarlo/include"

# Specify the top-level function for synthesis
set_top		mc_kernel

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

