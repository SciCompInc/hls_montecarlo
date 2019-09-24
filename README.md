# Financial Code development for FPGA

Welcome to the HLS Monte Carlo repository. This repository contains the set of Monte Carlo
simulation examples targeting Xilinx FPGA acceleration boards. The repository has the following structure:

- libs - contains the third-party libraries used by examples
- mcArithmAsian(Q) - Arithmetic Average Asian option (Quasi-) Monte Carlo code 
- mcPi(Q) - the number pi estimation (Quasi-) Monte Carlo code 
- mcSVEQLinkStructNote(Q) - Equity Linked Note (Quasi-) Monte Carlo code 
- montecarlo - (Quasi-) Monte Carlo HLS library
- tutorial - How-To Guide to get started with Monte Carlo code development for FPGA
- utility - SDAccel Makefile utilities

## Getting started

The `tutorial` directory contains a detailed step-by-step how-to guide for the Monte Carlo code development for Xilinx FPGA. It provide a basis for understanding the concepts behind more advanced examples.

## Example directory structure

 - src - source codes
 - .cproject - SDAccel project file
 - .project - SDAccel project file
 - project.sdx - SDAccel project file
 - Makefile - Makefile for the command-line build
 - run_u200.sh - Command-line run script (Alveo U200 target hardware)
 - sdaccel.ini - SDAccel configuration file
 - test.launch - SDAccel launch configuration
 - utils.mk - Utility include Makefile
 - hls - Vivado HLS project directory with as simple kernel wrapper and TCL script

## Examples: FPGA projects

The Vivado HLS project can be created and opened using the provided TCL script:

```
cd hls
vivado_hls -f run_hls.tcl
vivado_hls -p prj
```

The SDAccel build can be performed from command line using Makefile (see example in run_u200.sh script) or using SDAccel project files and importing them into SDAccel IDE workspace.

## Examples: non-FPGA projects

Optionally, the example directory can also include the following extra items:
 - data - input data file and generator scripts   
 - vs_emu - software emulation project directory
 - vs_opemp - OpenMP implementation for the multi-core CPU 
 - vs_cuda - Nvidia CUDA implementation

All optional projects directories has Visual Studio 2015 project file and CMake build scripts for the cross-platform build.
The CMake scripts require CMAke 3.9 or later and the following commands can be used to generate the Release version of the code:

```
cd vs_*
mkdir build
cd build
```

Windows build commands:

```
cmake .. -A x64
cmake --build . --config Release
```

Linux build commands:

```
cmake .. -DCMAKE_BUILD_TYPE =Release
cmake --build .
```

The OpenMP project executable will use all available CPU cores and the specific number of cores can be specified via OpenMP environment variable `OMP_NUM_THREADS`.

## Example code parameters

The projects executable read the input file, if any, from the current working directory, assuming the name `input_data.json`, or the input file path can be specified via environment variable `SCI_DATAFILE`.

The examples kernel code is controlled by `NUM_BLOCKS` (number of parallel blocks for the full unroll) and `NUM_SIMS` (the pipelined simulation batch size). It is recommended to reduce the value of those parameters in order to ensure a reasonable hardware emulation build and run times.

