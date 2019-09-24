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
 * @file core_utils.h
 *
 */

#pragma once

#include <exception>
#include <sstream>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#ifdef __CUDACC__
#define KERNEL_ARGS2(grid, block) << <grid, block>>>
#define KERNEL_ARGS3(grid, block, sh_mem) << <grid, block, sh_mem>>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream)                             \
<< <grid, block, sh_mem, stream>>>
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#define __CUDACC__
#endif

#define SCI_CUDART_CALL(err) sci::sciCUDAErrorCheck(err, __FILE__, __LINE__)

namespace sci
{

inline void sciCUDAErrorCheck(
    cudaError_t cudaResult,
    const char* file = 0,
    const int line = 0)
{
    if (cudaSuccess != cudaResult)
    {
        std::stringstream ss;
        if (file)
            ss << file << "(" << line << ") : ";
        ss << "FATAL ERROR: ";
        ss << cudaGetErrorString(cudaResult);
        std::string msg = ss.str();
        throw std::runtime_error(msg.c_str());
    }
}

inline void sciCUDAGetAvailableDevices(int& nGPU)
{
    SCI_CUDART_CALL(cudaGetDeviceCount(&nGPU));

    for (int iGPU = 0; iGPU < nGPU; iGPU++)
    {
        struct cudaDeviceProp deviceProperties;
        SCI_CUDART_CALL(cudaGetDeviceProperties(&deviceProperties, iGPU));
        if (deviceProperties.major * 10 + deviceProperties.minor < 13)
        {
            if (nGPU == 1 || iGPU < nGPU - 1)
                throw std::runtime_error(
                    "Devices with compute capability below "
                    "1.3 are not supported");
            nGPU--; // assume that the last device with CC < 1.3 is the display
                    // card
        }
    }
}

inline void sciCUDAOptimumGrid1D(unsigned int& size, int iGPU)
{
    struct cudaDeviceProp deviceProperties;
    SCI_CUDART_CALL(cudaGetDeviceProperties(&deviceProperties, iGPU));
    // Aim to launch around ten or more times as many blocks as there
    // are multiprocessors on the target device.
    unsigned int blocksPerSM = 10;
    unsigned int numSMs = deviceProperties.multiProcessorCount;
    while (size > 2 * blocksPerSM * numSMs)
        size >>= 1;
}

//****************************************************************************
// Because dynamically sized shared memory arrays are declared "extern",
// we can't templatize them directly.  To get around this, we declare a
// simple wrapper struct that will declare the extern array with a different
// name depending on the type.  This avoids compiler errors about duplicate
// definitions.
//
// To use dynamically allocated shared memory in a templatized __global__ or
// __device__ function, just replace code like this:
//      template<class T>
//      __global__ void
//      foo( T* g_idata, T* g_odata)
//      {
//          // Shared mem size is determined by the host app at run time
//          extern __shared__  T sdata[];
//          ...
//          x = sdata[i];
//          sdata[i] = x;
//          ...
//      }
//
// With this:
//      template<class T>
//      __global__ void
//      foo( T* g_idata, T* g_odata)
//      {
//          // Shared mem size is determined by the host app at run time
//          SharedMemory<T> sdata;
//          ...
//          x = sdata[i];
//          sdata[i] = x;
//          ...
//      }
//****************************************************************************

// This is the un-specialized struct.  Note that we prevent instantiation of
// this struct by making it abstract (i.e. with pure virtual methods).
template<typename T>
struct SharedMemoryBlock {
    // Ensure that we won't compile any un-specialized types
    // virtual __device__ T &operator*() = 0;
    // virtual __device__ T &operator[](int i) = 0;
    virtual __device__ T& operator[](int i) = 0;
};

#define BUILD_SHAREDMEMORY_TYPE(t, n)                                         \
    template<>                                                                \
    struct SharedMemoryBlock<t> {                                             \
        unsigned int dim;                                                     \
        __device__ SharedMemoryBlock(unsigned int dim_)                       \
            : dim(dim_)                                                       \
        {                                                                     \
        }                                                                     \
        __device__ t* operator[](unsigned int i)                              \
        {                                                                     \
            extern __shared__ t n[];                                          \
            return (i < dim) ? &n[i * blockDim.x] : 0;                        \
        }                                                                     \
    }
//__device__ t &operator*() { extern __shared__ t n[]; return *n; } \
        //__device__ t &operator[](int i) { extern __shared__ t n[]; return n[i]; } \

BUILD_SHAREDMEMORY_TYPE(int, s_int);
BUILD_SHAREDMEMORY_TYPE(unsigned int, s_uint);
BUILD_SHAREDMEMORY_TYPE(char, s_char);
BUILD_SHAREDMEMORY_TYPE(unsigned char, s_uchar);
BUILD_SHAREDMEMORY_TYPE(short, s_short);
BUILD_SHAREDMEMORY_TYPE(unsigned short, s_ushort);
BUILD_SHAREDMEMORY_TYPE(long, s_long);
BUILD_SHAREDMEMORY_TYPE(unsigned long, s_ulong);
BUILD_SHAREDMEMORY_TYPE(bool, s_bool);
BUILD_SHAREDMEMORY_TYPE(float, s_float);
BUILD_SHAREDMEMORY_TYPE(double, s_double);

#undef BUILD_SHAREDMEMORY_TYPE

template<typename Real>
__device__ Real reduce_sum(Real in)
{
    SharedMemoryBlock<Real> shared(1);

    Real* sdata = shared[0];

    // Perform first level of reduction:
    // - Write to shared memory
    unsigned int ltid = threadIdx.x;

    sdata[ltid] = in;
    __syncthreads();

    // Do reduction in shared mem
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (ltid < s)
        {
            sdata[ltid] = sdata[ltid] + sdata[ltid + s];
        }
        __syncthreads();
    }

    return sdata[0];
}

} // namespace sci
