#include "xcl2.hpp"
#include <cmath>
#include <vector>

#include <sci/brownian_bridge_setup.h>
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
    int steps,
    double dt,
    double vol,
    double r,
    double q,
    double spot,
    double strike,
    int call,
    int pmax,
    int seq,
    double& Vx,
    double& devx,
    double& tk)
{
    const int nbit = sci::new_sobol_joe_kuo_6_21201_nbit;
    const int ndim = steps;
    std::vector<unsigned int, aligned_allocator<unsigned int>> dirnum(
        ndim * nbit);
    std::vector<unsigned int, aligned_allocator<unsigned int>> shift(ndim, 0);
    sci::new_sobol_joe_kuo_6_21201_dirnum(
        ndim, dirnum.data(), shift.data(), false);

    std::vector<int, aligned_allocator<int>> c_data(steps + 1);
    std::vector<int, aligned_allocator<int>> l_data(steps + 1);
    std::vector<int, aligned_allocator<int>> r_data(steps + 1);
    std::vector<real_t, aligned_allocator<real_t>> qasave(steps + 1);
    std::vector<real_t, aligned_allocator<real_t>> qbsave(steps + 1);

    std::vector<real_t, aligned_allocator<real_t>> tsample(steps + 1);
    for (int n = 0; n <= steps; ++n)
    {
        tsample[n] = real_t(n) * real_t(dt);
    }

    sci::brownian_bridge_setup(
        steps,
        tsample.data(),
        c_data.data(),
        l_data.data(),
        r_data.data(),
        qasave.data(),
        qbsave.data());

    std::vector<real_t, aligned_allocator<real_t>> V_hw(1);

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
        cl::Buffer buffer_c_data(
            context,
            CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
            c_data.size() * sizeof(int),
            c_data.data(),
            &err));
    inBufVec.push_back(buffer_c_data);

    OCL_CHECK(
        err,
        cl::Buffer buffer_l_data(
            context,
            CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
            l_data.size() * sizeof(int),
            l_data.data(),
            &err));
    inBufVec.push_back(buffer_l_data);

    OCL_CHECK(
        err,
        cl::Buffer buffer_r_data(
            context,
            CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
            r_data.size() * sizeof(int),
            r_data.data(),
            &err));
    inBufVec.push_back(buffer_r_data);

    OCL_CHECK(
        err,
        cl::Buffer buffer_qasave(
            context,
            CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
            qasave.size() * sizeof(real_t),
            qasave.data(),
            &err));
    inBufVec.push_back(buffer_qasave);

    OCL_CHECK(
        err,
        cl::Buffer buffer_qbsave(
            context,
            CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
            qbsave.size() * sizeof(real_t),
            qbsave.data(),
            &err));
    inBufVec.push_back(buffer_qbsave);

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
    OCL_CHECK(err, err = krnl_mcPricer.setArg(arg_id++, steps));
    OCL_CHECK(err, err = krnl_mcPricer.setArg(arg_id++, real_t(dt)));
    OCL_CHECK(err, err = krnl_mcPricer.setArg(arg_id++, real_t(vol)));
    OCL_CHECK(err, err = krnl_mcPricer.setArg(arg_id++, real_t(r)));
    OCL_CHECK(err, err = krnl_mcPricer.setArg(arg_id++, real_t(q)));
    OCL_CHECK(err, err = krnl_mcPricer.setArg(arg_id++, real_t(spot)));
    OCL_CHECK(err, err = krnl_mcPricer.setArg(arg_id++, real_t(strike)));
    OCL_CHECK(err, err = krnl_mcPricer.setArg(arg_id++, call));
    OCL_CHECK(err, err = krnl_mcPricer.setArg(arg_id++, pmax));
    OCL_CHECK(err, err = krnl_mcPricer.setArg(arg_id++, seq));
    OCL_CHECK(err, err = krnl_mcPricer.setArg(arg_id++, buffer_dirnum));
    OCL_CHECK(err, err = krnl_mcPricer.setArg(arg_id++, buffer_c_data));
    OCL_CHECK(err, err = krnl_mcPricer.setArg(arg_id++, buffer_l_data));
    OCL_CHECK(err, err = krnl_mcPricer.setArg(arg_id++, buffer_r_data));
    OCL_CHECK(err, err = krnl_mcPricer.setArg(arg_id++, buffer_qasave));
    OCL_CHECK(err, err = krnl_mcPricer.setArg(arg_id++, buffer_qbsave));
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

    Vx = double(V_hw[0]) / double(pmax);
    devx = double(0);
}
