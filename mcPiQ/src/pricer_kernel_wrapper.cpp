/*
 * Copyright 2019 SciComp, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file pricer_kernel_wrapper.cpp
 *
 */

#include "xcl2.hpp"
#include <cmath>
#include <vector>

#include <sci/sobol_joe_kuo_dirnum.h>

#include "kernel_global.h"

uint64_t get_duration_ns(const cl::Event& event)
{
    uint64_t nstimestart, nstimeend;
    event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &nstimestart);
    event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &nstimeend);
    return (nstimeend - nstimestart);
}

void pricer_kernel_wrapper(
    int pmax,
    int seq,
    double& Vx,
    double& devx,
    double& tk)
{
    const int nbit = sci::new_sobol_joe_kuo_6_21201_nbit;
    const int ndim = 2;
    std::vector<unsigned int> dirnum(ndim * nbit);
    std::vector<unsigned int> shift(ndim);
    sci::new_sobol_joe_kuo_6_21201_dirnum(ndim, dirnum.data(), shift.data());

    std::vector<int, aligned_allocator<int>> V_hw(1);

    // OPENCL HOST CODE AREA START
    // get_xil_devices() is a utility API which will find the Xilinx
    // platforms and will return list of devices connected to Xilinx platform
    cl_int err;
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(
        err,
        cl::CommandQueue queue(
            context, device, CL_QUEUE_PROFILING_ENABLE, &err));
    OCL_CHECK(
        err, std::string device_name = device.getInfo<CL_DEVICE_NAME>(&err));

    const char* xcl_binary = getenv("XCL_BINARY");
    std::string binaryFile(xcl_binary);

    unsigned fileBufSize;
    char* fileBuf = xcl::read_binary_file(binaryFile, fileBufSize);
    cl::Program::Binaries bins{{fileBuf, fileBufSize}};
    devices.resize(1);
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));
    OCL_CHECK(err, cl::Kernel krnl_mcPricer(program, "pricer_kernel", &err));

    // Allocate Buffer in Global Memory
    // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and
    // Device-to-host communication
    std::vector<cl::Memory> inBufVec, outBufVec;

    OCL_CHECK(
        err,
        cl::Buffer buffer_dirnum(
            context,
            CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
            dirnum.size() * sizeof(unsigned int),
            dirnum.data(),
            &err));
    inBufVec.push_back(buffer_dirnum);

    OCL_CHECK(
        err,
        cl::Buffer buffer_V(
            context,
            CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
            sizeof(real_t),
            V_hw.data(),
            &err));
    outBufVec.push_back(buffer_V);

    int arg_id = 0;
    OCL_CHECK(err, err = krnl_mcPricer.setArg(arg_id++, pmax));
    OCL_CHECK(err, err = krnl_mcPricer.setArg(arg_id++, seq));
    OCL_CHECK(err, err = krnl_mcPricer.setArg(arg_id++, buffer_dirnum));
    OCL_CHECK(err, err = krnl_mcPricer.setArg(arg_id++, buffer_V));

    cl::Event event;
    uint64_t kernel_duration = 0;

    // Launch the Kernel
    // For HLS kernels global and local size is always (1,1,1). So, it is
    // recommended to always use enqueueTask() for invoking HLS kernel
    OCL_CHECK(err, err = queue.enqueueTask(krnl_mcPricer, NULL, &event));

    // Copy Result from Device Global Memory to Host Local Memory
    OCL_CHECK(
        err,
        err = queue.enqueueMigrateMemObjects(
            outBufVec, CL_MIGRATE_MEM_OBJECT_HOST));
    queue.finish();

    delete[] fileBuf;

    kernel_duration = get_duration_ns(event);
    tk = kernel_duration / 1000000;
    //    std::cout << "Wall Clock Time (Kernel execution) (ms): "
    //              << kernel_duration / 1000000 << std::endl;

    double ratio = double(V_hw[0]) / double(pmax);
    Vx = 4.0 * ratio;
    devx = double(0);
}
