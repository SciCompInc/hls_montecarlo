#include "xcl2.hpp"
#include <cmath>
#include <vector>

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
    std::vector<cl::Memory> outBufVec;

    OCL_CHECK(
        err,
        cl::Buffer buffer_V(
            context,
            CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
            sizeof(int),
            V_hw.data(),
            &err));
    outBufVec.push_back(buffer_V);

    int arg_id = 0;
    OCL_CHECK(err, err = krnl_mcPricer.setArg(arg_id++, pmax));
    OCL_CHECK(err, err = krnl_mcPricer.setArg(arg_id++, seq));
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
    devx = (ratio - ratio * ratio);
    devx = 4.0 * std::sqrt(devx / double(pmax));
}
