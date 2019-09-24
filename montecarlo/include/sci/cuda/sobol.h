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
 * @file sobol.h
 *
 */

#pragma once

#if defined(__CUDACC__)
#include <sci/cuda/cuda_common.h>
#else
#define __host__
#define __device__
#define __scifinance_exec_check_disable__
#endif

#include <sci/cuda/distributions.h>

#ifndef ffs
#if __CUDA_ARCH__ > 0
#define ffs(x) __ffs(x)
#elif defined(_MSC_VER)
#include <intrin.h>
inline int ffs(unsigned int x)
{
    unsigned long i = 0;
    if (_BitScanForward(&i, x))
        return i + 1;
    return 0;
}
#else
#define ffs(x) __builtin_ffs(x)
#endif
#endif

#define SCI_MAXBIT 32
#define SCI_MAXDIM 21201
#define SCI_NORM32 2.3283064365386962890625e-10 /* pow(2.0,-32.0) */

namespace sci
{

template<
    typename Real,
    template<typename>
    class SCIVECTOR,
    template<typename>
    class SCIMATRIX>
class SciSobolSeqC
{

private:
    unsigned int ndim_;
    unsigned int nbit_;
    unsigned int index_;
    unsigned int stride_;
    SCIVECTOR<unsigned int> ix_;
    SCIMATRIX<unsigned int> iv_;

    __scifinance_exec_check_disable__ __host__ __device__ void nextVector()
    {
        unsigned int pos = index_ * stride_;
        unsigned int mask = stride_ - 1;

        if (pos > 0)
        {
            /**
             * x[i] = x[i-stride] ^ v[s2p] ^ v[rmzb]
             * s2p is log2(stride)
             * rmzb is right most zero bit
             * log2(i) = ffs(i) - 1 , i = 2^k
             * rmzb(i) = ffs(~i)
             */
            if (stride_ > 1)
            {
                /**
                 * stride = 4
                 * pos = 0: #0 #1 #2 #3
                 *           |  |  |  |
                 * pos = 1: #4 #5 #6 #7
                 * ...
                 */
                for (unsigned int idim = 1; idim <= ndim_; ++idim)
                {
                    ix_(idim) ^= iv_(idim, ffs(stride_) - 1);
                }
            }
            for (unsigned int idim = 1; idim <= ndim_; ++idim)
            {
                ix_(idim) ^= iv_(idim, ffs(~((pos - 1) | mask)));
            }
        }

        index_++;
    }

public:
    __scifinance_exec_check_disable__ __host__ __device__ SciSobolSeqC(
        unsigned int sskip,
        unsigned int ndim,
        unsigned int nbit,
        const SCIMATRIX<unsigned int>& iv,
        SCIVECTOR<unsigned int>& ix,
        unsigned int nThreads = 1,
        unsigned int tid = 0)
        : ndim_(ndim)
        , nbit_(nbit)
        , index_(sskip / nThreads)
        , stride_(nThreads)
        , ix_(ix)
        , iv_(iv)
    {
        /**
         * Restrictions:
         * nThreads must be power of 2 (2^k)
         * sskip is floored to the nearest multiple of nThreads
         * max(ndim) is SCI_MAXDIM
         * max(nbit) is SCI_MAXBIT
         */
        unsigned int idim, pos;
        unsigned int ibit;
        unsigned int gcode;

        if (index_ == 0)
        {
            pos = tid;
        }
        else
        {
            pos = (index_ - 1) * nThreads + tid;
        }

        /* initialize the sequence */
        for (idim = 1; idim <= ndim_; ++idim)
        {
            ix_(idim) = 0;
        }

        /* skip to given position in the sequence */
        gcode = pos ^ (pos >> 1); /* Gray code */
        for (ibit = 1; ibit <= nbit_; ibit++)
        {
            if (gcode & 1)
            {
                for (idim = 1; idim <= ndim_; idim++)
                {
                    ix_(idim) ^= iv_(idim, ibit);
                }
            }
            gcode >>= 1;
        }
    }

    __scifinance_exec_check_disable__ __host__ __device__ void
    getVectorUniform(SCIVECTOR<Real>& Z)
    {
        nextVector();
        for (unsigned int idim = 1; idim <= ndim_; ++idim)
        {
            Z(idim) =
                static_cast<Real>(ix_(idim)) * static_cast<Real>(SCI_NORM32);
        }
    }

    __scifinance_exec_check_disable__ __host__ __device__ void
    getVectorNormal(SCIVECTOR<Real>& Z)
    {
        nextVector();
        for (unsigned int idim = 1; idim <= ndim_; ++idim)
        {
            Z(idim) = normICdf(
                static_cast<Real>(ix_(idim)) * static_cast<Real>(SCI_NORM32));
        }
    }

    __scifinance_exec_check_disable__ __host__ __device__ void
    getMatrixUniform(SCIMATRIX<Real>& Z, int base = 0)
    {
        nextVector();
        int idim = 1;
        int nhi = int(Z.size0()) - 1;
        int nsamp = int(Z.size1()) - 1;
        for (int j = 1; j <= nsamp; j++)
        {
            for (int i = base; i <= nhi; i++)
            {
                Z(i, j) = static_cast<Real>(ix_(idim++)) *
                    static_cast<Real>(SCI_NORM32);
            }
        }
    }

    __scifinance_exec_check_disable__ __host__ __device__ void
    getMatrixNormal(SCIMATRIX<Real>& Z, int base = 0)
    {
        nextVector();
        int idim = 1;
        int nhi = int(Z.size0()) - 1;
        int nsamp = int(Z.size1()) - 1;
        for (int j = 1; j <= nsamp; j++)
        {
            for (int i = base; i <= nhi; i++)
            {
                Z(i, j) = normICdf(
                    static_cast<Real>(ix_(idim++)) *
                    static_cast<Real>(SCI_NORM32));
            }
        }
    }

    __scifinance_exec_check_disable__ __host__ __device__ void
    getMatrixNormalPartial(SCIMATRIX<Real>& Z, int nrows, int base = 0)
    {
        nextVector();
        int idim = 1;
        int nsamp = int(Z.size1()) - 1;
        int nhi = int(Z.size0()) - 1;
        for (int j = 1; j <= nsamp; j++)
        {
            for (int i = base; i <= base + nrows - 1; i++)
            {
                Z(i, j) = normICdf(
                    static_cast<Real>(ix_(idim++)) *
                    static_cast<Real>(SCI_NORM32));
            }
            for (int i = base + nrows; i <= nhi; i++)
            {
                Z(i, j) = static_cast<Real>(ix_(idim++)) *
                    static_cast<Real>(SCI_NORM32);
            }
        }
    }
};

} // namespace sci
