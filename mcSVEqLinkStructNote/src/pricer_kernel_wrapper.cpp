#include <cmath>
#include <vector>

#include "xcl2.hpp"

#include "kernel_global.h"

uint64_t get_duration_ns(const cl::Event& event)
{
    uint64_t nstimestart, nstimeend;
    event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &nstimestart);
    event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &nstimeend);
    return (nstimeend - nstimestart);
}

void pricer_kernel_wrapper(
    const std::vector<double>& DeltaT,
    const std::vector<double>& r,
    const std::vector<int[3]>& DEtable,
    const std::vector<double>& DEtimeTable,
    const std::vector<std::vector<double>>& Mcor,
    double AccruedBonus0,
    double BonusCoup,
    const std::vector<double>& CallBarrier,
    const std::vector<double>& kappa,
    const std::vector<double>& KI0,
    double KIBarrier,
    double KINum,
    double MatCoup,
    double Notional,
    int pmax,
    const std::vector<double>& q,
    int seq,
    const std::vector<double>& sigma,
    const std::vector<double>& Spot,
    const std::vector<double>& SRef,
    const std::vector<double>& theta,
    const std::vector<double>& vSpot,
    double& devx,
    double& Vx,
    double& tk)
{
    real_t V, dev;

    int nObs = CallBarrier.size();
    int nD = Spot.size();
    int miMax = r.size() - 1;

    cl_int err;
    // Allocate Memory in Host Memory
    // When creating a buffer with user pointer (CL_MEM_USE_HOST_PTR), under
    // the hood user ptr is used if it is properly aligned. when not aligned,
    // runtime had no choice but to create its own host side buffer. So it is
    // recommended to use this allocator if user wish to create buffer using
    // CL_MEM_USE_HOST_PTR to align user buffer to page boundary. It will
    // ensure that user buffer is used when user create Buffer/Mem object with
    // CL_MEM_USE_HOST_PTR
    std::vector<real_t, aligned_allocator<real_t>> CallBarrier_(nObs + 1);
    for (int i = 0; i < nObs; i++)
    {
        CallBarrier_[i + 1] = (real_t)CallBarrier[i];
    }

    std::vector<real_t, aligned_allocator<real_t>> kappa_(nD + 1);
    std::vector<real_t, aligned_allocator<real_t>> KI0_(nD + 1);
    std::vector<real_t, aligned_allocator<real_t>> q_(nD + 1);
    std::vector<real_t, aligned_allocator<real_t>> sigma_(nD + 1);
    std::vector<real_t, aligned_allocator<real_t>> Spot_(nD + 1);
    std::vector<real_t, aligned_allocator<real_t>> SRef_(nD + 1);
    std::vector<real_t, aligned_allocator<real_t>> theta_(nD + 1);
    std::vector<real_t, aligned_allocator<real_t>> vSpot_(nD + 1);
    for (int i = 0; i < nD; i++)
    {
        kappa_[i + 1] = (real_t)kappa[i];
        KI0_[i + 1] = (real_t)KI0[i];
        q_[i + 1] = (real_t)q[i];
        sigma_[i + 1] = (real_t)sigma[i];
        Spot_[i + 1] = (real_t)Spot[i];
        SRef_[i + 1] = (real_t)SRef[i];
        theta_[i + 1] = (real_t)theta[i];
        vSpot_[i + 1] = (real_t)vSpot[i];
    }

    std::vector<real_t, aligned_allocator<real_t>> DeltaT_(nObs + 1);
    for (int i = 1; i <= nObs; i++)
    {
        DeltaT_[i] = (real_t)DeltaT[i];
    }

    std::vector<real_t, aligned_allocator<real_t>> r_(miMax + 1);
    for (int i = 0; i <= miMax; i++)
    {
        r_[i] = (real_t)r[i];
    }

    std::vector<real_t, aligned_allocator<real_t>> DEtimeTable_(miMax + 1);
    for (int i = 0; i <= miMax; i++)
    {
        DEtimeTable_[i] = (real_t)DEtimeTable[i];
    }

    std::vector<int, aligned_allocator<int>> DEtable_((miMax + 1) * 3);
    for (int i = 0; i <= miMax; i++)
    {
        for (int id = 0; id < 3; id++)
            DEtable_[i * 3 + id] = DEtable[i][id];
    }

    std::vector<real_t, aligned_allocator<real_t>> Mcor_(
        (2 * nD + 1) * (2 * nD + 1));
    for (int i = 1; i <= 2 * nD; i++)
    {
        for (int j = 1; j <= 2 * nD; j++)
        {
            Mcor_[i * (2 * nD + 1) + j] = (real_t)Mcor[i - 1][j - 1];
        }
    }

    std::vector<real_t, aligned_allocator<real_t>> V_hw(1);
    std::vector<real_t, aligned_allocator<real_t>> dev_hw(1);

    // OPENCL HOST CODE AREA START
    // get_xil_devices() is a utility API which will find the Xilinx
    // platforms and will return list of devices connected to Xilinx platform
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
    OCL_CHECK(
        err,
        cl::Kernel krnl_mcSVEqLinkStructNote1(program, "pricer_kernel", &err));

    // Allocate Buffer in Global Memory
    // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and
    // Device-to-host communication
    std::vector<cl::Memory> inBufVec, outBufVec;
    OCL_CHECK(
        err,
        cl::Buffer buffer_CallBarrier(
            context,
            CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
            CallBarrier_.size() * sizeof(real_t),
            CallBarrier_.data(),
            &err));
    inBufVec.push_back(buffer_CallBarrier);

    OCL_CHECK(
        err,
        cl::Buffer buffer_kappa(
            context,
            CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
            kappa_.size() * sizeof(real_t),
            kappa_.data(),
            &err));
    inBufVec.push_back(buffer_kappa);

    OCL_CHECK(
        err,
        cl::Buffer buffer_KI0(
            context,
            CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
            KI0_.size() * sizeof(real_t),
            KI0_.data(),
            &err));
    inBufVec.push_back(buffer_KI0);

    OCL_CHECK(
        err,
        cl::Buffer buffer_q(
            context,
            CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
            q_.size() * sizeof(real_t),
            q_.data(),
            &err));
    inBufVec.push_back(buffer_q);

    OCL_CHECK(
        err,
        cl::Buffer buffer_sigma(
            context,
            CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
            sigma_.size() * sizeof(real_t),
            sigma_.data(),
            &err));
    inBufVec.push_back(buffer_sigma);

    OCL_CHECK(
        err,
        cl::Buffer buffer_Spot(
            context,
            CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
            Spot_.size() * sizeof(real_t),
            Spot_.data(),
            &err));
    inBufVec.push_back(buffer_Spot);

    OCL_CHECK(
        err,
        cl::Buffer buffer_SRef(
            context,
            CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
            SRef_.size() * sizeof(real_t),
            SRef_.data(),
            &err));
    inBufVec.push_back(buffer_SRef);

    OCL_CHECK(
        err,
        cl::Buffer buffer_theta(
            context,
            CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
            theta_.size() * sizeof(real_t),
            theta_.data(),
            &err));
    inBufVec.push_back(buffer_theta);

    OCL_CHECK(
        err,
        cl::Buffer buffer_vSpot(
            context,
            CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
            vSpot_.size() * sizeof(real_t),
            vSpot_.data(),
            &err));
    inBufVec.push_back(buffer_vSpot);

    OCL_CHECK(
        err,
        cl::Buffer buffer_DeltaT(
            context,
            CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
            DeltaT_.size() * sizeof(real_t),
            DeltaT_.data(),
            &err));
    inBufVec.push_back(buffer_DeltaT);

    OCL_CHECK(
        err,
        cl::Buffer buffer_r(
            context,
            CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
            r_.size() * sizeof(real_t),
            r_.data(),
            &err));
    inBufVec.push_back(buffer_r);

    OCL_CHECK(
        err,
        cl::Buffer buffer_DEtimeTable(
            context,
            CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
            DEtimeTable_.size() * sizeof(real_t),
            DEtimeTable_.data(),
            &err));
    inBufVec.push_back(buffer_DEtimeTable);

    OCL_CHECK(
        err,
        cl::Buffer buffer_DEtable(
            context,
            CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
            DEtable_.size() * sizeof(int),
            DEtable_.data(),
            &err));
    inBufVec.push_back(buffer_DEtable);

    OCL_CHECK(
        err,
        cl::Buffer buffer_Mcor(
            context,
            CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
            Mcor_.size() * sizeof(real_t),
            Mcor_.data(),
            &err));
    inBufVec.push_back(buffer_Mcor);

    OCL_CHECK(
        err,
        cl::Buffer buffer_V(
            context,
            CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
            sizeof(real_t),
            V_hw.data(),
            &err));
    outBufVec.push_back(buffer_V);

    OCL_CHECK(
        err,
        cl::Buffer buffer_dev(
            context,
            CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
            sizeof(real_t),
            dev_hw.data(),
            &err));
    outBufVec.push_back(buffer_dev);

    /* Copy input vectors to memory */
    queue.enqueueMigrateMemObjects(inBufVec, 0 /* 0 means from host*/);

    int arg_id = 0;
    OCL_CHECK(
        err,
        err = krnl_mcSVEqLinkStructNote1.setArg(
            arg_id++, (real_t)AccruedBonus0));
    OCL_CHECK(
        err,
        err = krnl_mcSVEqLinkStructNote1.setArg(arg_id++, (real_t)BonusCoup));
    OCL_CHECK(
        err,
        err = krnl_mcSVEqLinkStructNote1.setArg(arg_id++, buffer_CallBarrier));
    OCL_CHECK(
        err, err = krnl_mcSVEqLinkStructNote1.setArg(arg_id++, buffer_kappa));
    OCL_CHECK(
        err, err = krnl_mcSVEqLinkStructNote1.setArg(arg_id++, buffer_KI0));
    OCL_CHECK(
        err,
        err = krnl_mcSVEqLinkStructNote1.setArg(arg_id++, (real_t)KIBarrier));
    OCL_CHECK(
        err, err = krnl_mcSVEqLinkStructNote1.setArg(arg_id++, (real_t)KINum));
    OCL_CHECK(
        err,
        err = krnl_mcSVEqLinkStructNote1.setArg(arg_id++, (real_t)MatCoup));
    OCL_CHECK(err, err = krnl_mcSVEqLinkStructNote1.setArg(arg_id++, nD));
    OCL_CHECK(
        err,
        err = krnl_mcSVEqLinkStructNote1.setArg(arg_id++, (real_t)Notional));
    OCL_CHECK(err, err = krnl_mcSVEqLinkStructNote1.setArg(arg_id++, pmax));
    OCL_CHECK(
        err, err = krnl_mcSVEqLinkStructNote1.setArg(arg_id++, buffer_q));
    OCL_CHECK(err, err = krnl_mcSVEqLinkStructNote1.setArg(arg_id++, seq));
    OCL_CHECK(
        err, err = krnl_mcSVEqLinkStructNote1.setArg(arg_id++, buffer_sigma));
    OCL_CHECK(
        err, err = krnl_mcSVEqLinkStructNote1.setArg(arg_id++, buffer_Spot));
    OCL_CHECK(
        err, err = krnl_mcSVEqLinkStructNote1.setArg(arg_id++, buffer_SRef));
    OCL_CHECK(
        err, err = krnl_mcSVEqLinkStructNote1.setArg(arg_id++, buffer_theta));
    OCL_CHECK(
        err, err = krnl_mcSVEqLinkStructNote1.setArg(arg_id++, buffer_vSpot));
    OCL_CHECK(err, err = krnl_mcSVEqLinkStructNote1.setArg(arg_id++, nObs));
    OCL_CHECK(err, err = krnl_mcSVEqLinkStructNote1.setArg(arg_id++, miMax));
    OCL_CHECK(
        err,
        err = krnl_mcSVEqLinkStructNote1.setArg(arg_id++, buffer_DEtimeTable));
    OCL_CHECK(
        err,
        err = krnl_mcSVEqLinkStructNote1.setArg(arg_id++, buffer_DEtable));
    OCL_CHECK(
        err, err = krnl_mcSVEqLinkStructNote1.setArg(arg_id++, buffer_DeltaT));
    OCL_CHECK(
        err, err = krnl_mcSVEqLinkStructNote1.setArg(arg_id++, buffer_Mcor));
    OCL_CHECK(
        err, err = krnl_mcSVEqLinkStructNote1.setArg(arg_id++, buffer_r));
    OCL_CHECK(
        err, err = krnl_mcSVEqLinkStructNote1.setArg(arg_id++, buffer_V));
    OCL_CHECK(
        err, err = krnl_mcSVEqLinkStructNote1.setArg(arg_id++, buffer_dev));

    cl::Event event;
    uint64_t kernel_duration = 0;

    // Launch the Kernel
    // For HLS kernels global and local size is always (1,1,1). So, it is
    // recommended to always use enqueueTask() for invoking HLS kernel
    OCL_CHECK(
        err,
        err = queue.enqueueTask(krnl_mcSVEqLinkStructNote1, NULL, &event));

    // Copy Result from Device Global Memory to Host Local Memory
    OCL_CHECK(
        err,
        err = queue.enqueueMigrateMemObjects(
            outBufVec, CL_MIGRATE_MEM_OBJECT_HOST));
    queue.finish();

    delete[] fileBuf;

    kernel_duration = get_duration_ns(event);
    tk = double(kernel_duration) / double(1000000);
    //    std::cout << "Wall Clock Time (Kernel execution) (ms): "
    //              << kernel_duration / 1000000 << std::endl;

    Vx = double(V_hw[0]) / double(pmax);
    devx = (double(dev_hw[0]) / double(pmax) - Vx * Vx);
    devx = std::sqrt(devx / double(pmax));
}
