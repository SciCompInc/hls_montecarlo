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
 * @file brownian_bridge.h
 *
 */
#pragma once

#include <cmath>

#if defined(__CUDACC__)
#include <sci/cuda/cuda_common.h>
#else
#define __host__
#define __device__
#define __scifinance_exec_check_disable__
#endif

namespace sci
{

__scifinance_exec_check_disable__ template<
    typename Real,
    template<typename>
    class SCIVECTOR>
__host__ __device__ void BrownianBridge(
    SCIVECTOR<Real>& QRSeq,
    SCIVECTOR<Real>& BbQRSeq,
    const SCIVECTOR<int>& child,
    const SCIVECTOR<int>& lparent,
    const SCIVECTOR<int>& rparent,
    const SCIVECTOR<Real>& qasave,
    const SCIVECTOR<Real>& qbsave)
{
    // Input and output vector draw
    // QRSeq and BbQRSeq are 1-based arrays
    Real LeftRV, RightRV, CenterRV;
    size_t nsamp = BbQRSeq.size0() - 1;

    BbQRSeq(nsamp) = qasave(0) * QRSeq(1);
    BbQRSeq(0) = 0.0;

    for (size_t i = 1; i < nsamp; i++)
    {
        LeftRV = BbQRSeq(lparent(i));
        RightRV = BbQRSeq(rparent(i));
        CenterRV = QRSeq(i + 1);
        BbQRSeq(child(i)) =
            qasave(i) * (LeftRV - RightRV) + RightRV + qbsave(i) * CenterRV;
    }
    for (size_t i = nsamp; i >= 1; i--)
    {
        BbQRSeq(i) = BbQRSeq(i) - BbQRSeq(i - 1);
    }
}

__scifinance_exec_check_disable__ template<
    typename Real,
    template<typename>
    class SCIVECTOR,
    template<typename>
    class SCIMATRIX>
__host__ __device__ void BrownianBridge(
    SCIMATRIX<Real>& QRSeq,
    SCIMATRIX<Real>& BbQRSeq,
    const SCIVECTOR<int>& child,
    const SCIVECTOR<int>& lparent,
    const SCIVECTOR<int>& rparent,
    const SCIVECTOR<Real>& qasave,
    const SCIVECTOR<Real>& qbsave,
    int base = 0)
{
    // Input and output vector draw
    // QRSeq and BbQRSeq are 1-based arrays
    Real LeftRV, RightRV, CenterRV;
    int nhi = int(BbQRSeq.size0()) - 1;
    int nsamp = int(BbQRSeq.size1()) - 1;

    for (int iD = base; iD <= nhi; iD++)
    {

        BbQRSeq(iD, nsamp) = qasave(0) * QRSeq(iD, 1);
        BbQRSeq(iD, 0) = 0.0;

        for (int i = 1; i < nsamp; i++)
        {
            LeftRV = BbQRSeq(iD, lparent(i));
            RightRV = BbQRSeq(iD, rparent(i));
            CenterRV = QRSeq(iD, i + 1);
            BbQRSeq(iD, child(i)) = qasave(i) * (LeftRV - RightRV) + RightRV +
                qbsave(i) * CenterRV;
        }
        for (int i = nsamp; i >= 1; i--)
        {
            BbQRSeq(iD, i) = BbQRSeq(iD, i) - BbQRSeq(iD, i - 1);
        }
    } // iD
}

} // namespace sci
