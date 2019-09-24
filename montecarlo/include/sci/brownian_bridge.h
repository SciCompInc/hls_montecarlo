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
 * Brownian Bridge transform class
 */

#pragma once

namespace sci
{
template<class T>
inline void brownian_bridge(
    T* QRSeq,
    T* BbQRSeq,
    int* child,
    int* lparent,
    int* rparent,
    T* qasave,
    T* qbsave,
    int nsamp)
{
    // Input and output vector draw
    // QRSeq and BbQRSeq are 1-based arrays
    T LeftRV, RightRV, CenterRV;
    BbQRSeq[nsamp] = qasave[0] * QRSeq[0];
    BbQRSeq[0] = T(0);

    for (int i = 1; i < nsamp; i++)
    {
        LeftRV = BbQRSeq[lparent[i]];
        RightRV = BbQRSeq[rparent[i]];
        CenterRV = QRSeq[i];
        BbQRSeq[child[i]] =
            qasave[i] * (LeftRV - RightRV) + RightRV + qbsave[i] * CenterRV;
    }
    for (int i = nsamp; i >= 1; i--)
    {
        BbQRSeq[i] = BbQRSeq[i] - BbQRSeq[i - 1];
    }
}
} // namespace sci
