# Getting started with Monte Carlo code development for FPGA

## Introduction

The purpose of this guide is a walk through the standard and quasi Monte Carlo code development basics for Xilinx FPGA. Monte Carlo method is essentially
a way to compute expected values by generating random scenarios and then
averaging them and it is very efficient to parallelize.

## Prerequisites

This guide assumes that user is familiar with basics of SDAccel and Vivado HLS environment, setup, programming and debugging Flow. The following resources provide a good starting point:

- [SDAccel portal](https://www.xilinx.com/products/design-tools/software-zone/sdaccel.html#gettingstarted)

- [Getting Started on AWS F1 with SDAccel and C/C++ Kernels](https://github.com/Xilinx/SDAccel-Tutorials/tree/master/docs/aws-getting-started/CPP)

- [SDAccel Design Hub](https://www.xilinx.com/support/documentation-navigation/design-hubs/dh0058-sdaccel-hub.html)

- [HLS Sesign Hub](https://www.xilinx.com/support/documentation-navigation/design-hubs/dh0012-vivado-high-level-synthesis-hub.html)


## Standard Monte Carlo number pi estimation

We start from one of the most straightforward cases of Monte Carlo simulations, the
calculation of number pi. The number pi can be calculated from the ratio of
the area of the quarter of the unit circle to the area of the unit square
multiplied by 4. Generating pairs of uniformly distributed in (0,1) random
numbers, corresponding to the coordinates of the point on 2D plane, we can
calculate the ratio of points inside the circle to the total number of points.
This ratio multiplied by 4 gives the number pi estimation. A simple serial
C++ implementation of the standard Monte Carlo simulation is presented 
in `step1` directory.

## Host and kernel code
In the typical development scenario, the original code is split into the
serial CPU-only part, called host code, and the kernel code that runs in
parallel on the accelerator hardware. While host code mostly remains
unchanged, the kernel code must be modified according to the target platform
requirements. Also, the kernel wrapper code must be added to match the
hardware interface.

The first step is to decide on the kernel language. The possible choice is
plain C/C++, or OpenCL C. C++ kernels offers the most flexibility, and they
require minimal changes from the original code. The next step is to choose
the kernel development flow. The standard approach is the top-down flow when
the developer writes an OpenCL kernel wrapper and works only in SDAccel
Integrated Development Environment (IDE). In this case, the Vivado High-Level
Synthesis (HLS) tool that translates the kernel code to HDL runs behind the
scene. Alternatively, in the case of C/C++ kernel code, there is a bottom-up
flow where the developer writes a minimal kernel wrapper and the test
harness, called the test bench, and validates the kernel and performs the
kernel optimizations from Vivado HLS project. It enables a clear separation
of the kernel code and host code/application development processes.

The result of the code re-organization and the corresponding Vivado HLS TCL script can be found in `step2` directory. The following command lines can be used to create Vivado HLS project and open it:

```
cd step2
vivado_hls -f run_hls.tcl
vivado_hls -p prj
```

## Standard Monte Carlo functional testing

A simple single simulation result accuracy test is not enough to validate the correctness
of the Monte Carlo code. It is important to verify the convergence to a true analytic value and provide a confidence interval estimation. These tests should be treated as functional/integration tests and must be run only after the final FPGA-accelerated code is built. The example of the convergence test is presented in `step2/test.cpp`. It uses Catch2 unit testing framework and can be built using the provided `CMakeLists.txt` CMake file.


## Baseline HLS kernel

Then we proceed with the development of a baseline version of the C/C++ kernel
code. The target kernel code is refactored using the HLS kernel code
guidelines. For example, the dynamic memory allocations must be replaced with
parametrized static memory declarations and all library calls must be replaced
with HLS compatible libraries. 

The Monte Carlo code is a massively parallel application. The most straightforward FPGA compatible approach is to divide the simulation into several
blocks, and each block is processed using a part of the FPGA resources area.
These blocks are processed in parallel, and it is equivalent to unrolling
a path loop with a small factor, say 8 or 16, so there are enough resources
to process each block. 

The parallel code structure and FPGA hardware impose special requirements on the possible implementations of the Random Number Generator (RNG). It should have the capability to generate independent sequences of the random number, so it should have a fast-forward (fast jump ahead in the sequence) functionality. While many RNGs have a very long period, there is no guarantee that the different initial seeds produce independent (non-overlapping, uncorrelated) sequences of random
numbers. Without the fast-forward functionality, the RNG initialization can be very time consuming eliminating all the benefits of the parallel MC code. The second requirement is that the RNG state must be small and state advance function must FPGA resource-efficient, i.e., it should have only simple bitwise operations with minimum multiplication/division operations.

We use xoshiro128** RNG from the FPGA Monte Carlo component library. It has a small state and jump-ahead routine that enables an independent subsequence generation required for each simulation block.
For example, to set up a generation of the random sequence with a given seed split into the given number of subsequences/blocks, we can write the following code for the simulation block:

```
sci::hls::xoshiro128starstar rng;
// block-splitting random sequence
rng.seed(seed);
for (int j = 0; j < block_id + 1; j++)
{
    rng.jump();
}
```

After implementing the above procedures the code should compile and run the C simulation ("Project->Run C Simulation") in Vivado HLS and kernel code must synthesize ("Solution -> Run C Synthesis -> Active Solution") without errors. 
The resulting code can be found in `step3` directory. 

## Optimized kernel

At this stage, the kernel code must be optimized by changing its structure and
using the HLS pragma directives. It can be done in both Vivado HLS and
SDAccel. The most of the initial optimization efforts are concentrated around
the loops, unrolling and pipelining. Each block is then processed in batches, where the batch size depends on the resources available for the
block. Within a batch, the processing is pipelined, and the goal is to
minimize the initiation interval (II). With a small initiation interval,
the loop latency is reduced to the number of cycles close to the number
of loop iterations. After the synthesis, Vivado HLS reports the latency of loops, and for the loops with the variable index, the latency report can
be produced by specifying `loop_tripcount` HLS pragma. It did not impact the synthesis and used only for reporting purposes. Also, it is convenient to add labels to the code, and it helps to identify the particular loop in the HLS synthesis report. The optimized kernel is provided in `step4` directory. The following HLS pragmas were used:

- unroll
- pipeline
- array_partition
- loop_tripcount

## Kernel validation

After the HLS synthesis report is analyzed for the performance (timing and latency) and utilization, it is a time to perform C/RTL co-simulation 
("Solution -> Run C/RTL Cosimulation", select "Verilog" and "Optimizing compile" options) to validate the synthesized Hardware Description Language (HDL) code. The kernel input parameters, as well as the number of blocks and batch sizes, should be scaled down, to ensure a reasonable hardware emulation execution time. The "Pass" in the report indicates the successful validation.

## Power usage estimation

The power usage can be estimated by using a spreadsheet calculator, Xilinx Power Estimator (XPE), or using Vivado IDE power usage estimation tool. The latter provides a more accurate estimation directly from the synthesized design. In order to use it, we need to create the Vivado project. It can be done from Vivado HLS via "Solution -> Export RTL" and selecting "Vivado synthesis" option.
After the export report is generated, the Vivado project can be found in `solution1/impl/verilog` directory, named `project.xpr`. Open the project in Vivado IDE, go to "Project Manager -> Synthesis -> Open Synthesized Design".
After the Device info is loaded, choose "Report Power" with the default options.
It produces the pre-implementation power estimation from Synthesized netlist.

## Complete application

We now can switch to SDAccel from Vivado HLS to develop a complete FPGA-accelerated application. The kernel code interface must be specified
using HLS pragma directives according to SDAccel requirements. Then the host code call to the kernel function must be wrapped using the OpenCL. After that, the code should compile and run in SDAccel software and hardware emulation mode without any issues.
Both Makefile and SDAccel project files are provided.
The hardware emulation run can provide a reasonably good ballpark
estimate for the runtime performance and energy efficiency.
The resulting code can be found in `step5` directory. The Vivado HLS script and a simple kernel wrapper are also provided in `step5/hls` for convenience.

## Hardware build

Once the hardware emulation mode builds and runs successfully, the code is
ready for the real hardware build. In contrast with CPU and GPU, the real hardware
build for FPGA takes a significant amount of time (hours) due to the circuit's optimal placing and routing. While it can be considered an inconvenience within the standard developing practices, there is typically need to do it only at the code release point,
because the software and hardware emulation typically provide a reasonable estimate of the real hardware code performance. Also, the hardware build time can be hidden in the overnight build/tests within a continuous integration practice. The resulting FPGA accelerated application can be validated using the integration tests from `step2`.
It is not always possible to get an exact match for the serial code and the FPGA code results, so the test reference results must be modified accordingly.

## Quasi Monte Carlo code development

At this point, we have developed, built, and validated the standard Monte Carlo simulation code for the number pi estimation. Now we modify the code for quasi-Monte Carlo simulation. In this case, the pseudo-random generator is replaced with a quasi-random (Sobol) generator with 2 dimensions. The critical aspect of the quasi-random generator implementation on FPGA is that the code must be split between host and kernel to minimize the FPGA area consumption and maximize the performance. So typically, it makes sense to keep all initializations on the host side and then move the pre-calculated
data to kernel. 

The kernel wrapper should now include the code to generate a Sobol direction 
numbers required for the sequence generation and pass it to the kernel.
For example, in this case, the initialization code looks like:

```
const int nbit = sci::new_sobol_joe_kuo_6_21201_nbit;
const int ndim = 2;
std::vector<unsigned int> dirnum(ndim * nbit);
std::vector<unsigned int> shift(ndim);
sci::new_sobol_joe_kuo_6_21201_dirnum(ndim, dirnum.data(), shift.data());
```

where `dirnum` is the direction numbers table, and `shift` is the initial state vector (it is non-trivial if the sequence scrambling is enabled; otherwise it is zero).

The kernel code is modified to accept Sobol direction numbers data and the sequence offset (quasi-Monte Carlo does not provide the standard error estimation, and therefore it should be computed as a statistics using difference sequence offsets).
Before the kernel block call, the Sobol generator is initialized and replicated to provide a private copy for every block to enable parallel processing after the unrolling. 

Inside the kernel block, the sequence is block-split using sequence skip functionality:

```
qrng.skip((block_id + seq * NUM_BLOCKS) * pmax);
```

Then the batch of the random 32-bit integers is generated before the pipelined processing where the uniform variates are produced and evaluated.

The convergence test harness must be modified to enable the standard error calculation using sequence offset statistics since the standard error can not be estimated from the simulation statistics in the quasi-Monte Carlo case.
