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
 * Kernel wrapper allows the kernel call via the common interface
 * Hides floating point type and hardware interface
 */
#include <cmath>
#include <vector>

#include <sci/sobol_joe_kuo_dirnum.h>

#include "kernel_global.h"

#include "pricer_kernel_wrapper.h"

extern "C" void
pricer_kernel(int pmax, int seq, unsigned int* dirnum, int* sump);

void pricer_kernel_wrapper(
    int pmax,
    int seq,
    double& Vx,
    double& devx,
    double& tk)
{
    int sum;

    const int nbit = sci::new_sobol_joe_kuo_6_21201_nbit;
    const int ndim = 2;
    std::vector<unsigned int> dirnum(ndim * nbit);
    std::vector<unsigned int> shift(ndim);
    sci::new_sobol_joe_kuo_6_21201_dirnum(ndim, dirnum.data(), shift.data());

    pricer_kernel(pmax, seq, dirnum.data(), &sum);

    double ratio = double(sum) / double(pmax);
    Vx = 4.0 * ratio;
    devx = 0;
    tk = 0;
}
