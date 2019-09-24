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
 * @file blas_utils.h
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

namespace sci
{

// BLAS: DTRMV
__scifinance_exec_check_disable__ template<
    typename Real,
    template<typename>
    class SCIVECTOR,
    template<typename>
    class SCIMATRIX>
__host__ __device__ void
Cholvmm(const SCIMATRIX<Real>& L, SCIVECTOR<Real>& x, int base = 0)
{
    /* L lower triangular matrix*/
    /* x=L*x */
    int nhi = int(L.size0()) - 1;
    for (int i3 = nhi; i3 >= base; i3--)
    {
        Real csum = 0.0;
        for (int j1 = base; j1 <= i3; j1++)
        {
            csum += L(i3, j1) * x(j1);
        }
        x(i3) = csum;
    }
}

__scifinance_exec_check_disable__ template<
    typename Real,
    template<typename>
    class SCIMATRIX>
__host__ __device__ void
Cholvmm(const SCIMATRIX<Real>& L, Real* x, int base = 0)
{
    /* L lower triangular matrix*/
    /* x=L*x */
    int nhi = int(L.size0()) - 1;
    for (int i3 = nhi; i3 >= base; i3--)
    {
        Real csum = 0.0;
        for (int j1 = base; j1 <= i3; j1++)
        {
            csum += L(i3, j1) * x[j1];
        }
        x[i3] = csum;
    }
}

// BLAS: DTRMM
__scifinance_exec_check_disable__ template<
    typename Real,
    template<typename>
    class SCIMATRIX>
__host__ __device__ void
Cholvmm(const SCIMATRIX<Real>& L, SCIMATRIX<Real>& X, int base = 0)
{
    /* L lower triangular matrix*/
    /* X=L*X */
    int nsamp = int(X.size1()) - 1;
    int nhi = int(L.size0()) - 1;
    for (int n = 1; n <= nsamp; n++)
    {
        for (int i3 = nhi; i3 >= base; i3--)
        {
            Real csum = 0.0;
            for (int j1 = base; j1 <= i3; j1++)
            {
                csum += L(i3, j1) * X(j1, n);
            }
            X(i3, n) = csum;
        }
    }
}

} // namespace sci
