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

#include <sci/brownian_bridge_setup.h>
#include <sci/sobol_joe_kuo_dirnum.h>

#include "kernel_global.h"

#include "pricer_kernel_wrapper.h"

extern "C" void pricer_kernel(
    int steps,
    real_t dt,
    real_t vol,
    real_t r,
    real_t q,
    real_t spot,
    real_t strike,
    int call,
    int pmax,
    int seq,
    unsigned int* dirnum,
    int* c_data,
    int* l_data,
    int* r_data,
    real_t* qasave,
    real_t* qbsave,
    real_t* payoff_sum);

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
    real_t payoff_sum;

    const int nbit = sci::new_sobol_joe_kuo_6_21201_nbit;
    const int ndim = steps;
    std::vector<unsigned int> dirnum(ndim * nbit);
    std::vector<unsigned int> shift(ndim);
    sci::new_sobol_joe_kuo_6_21201_dirnum(ndim, dirnum.data(), shift.data());

    std::vector<int> c_data(steps + 1);
    std::vector<int> l_data(steps + 1);
    std::vector<int> r_data(steps + 1);
    std::vector<real_t> qasave(steps + 1);
    std::vector<real_t> qbsave(steps + 1);

    std::vector<real_t> tsample(steps + 1);
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

    pricer_kernel(
        steps,
        (real_t)dt,
        (real_t)vol,
        (real_t)r,
        (real_t)q,
        (real_t)spot,
        (real_t)strike,
        call,
        pmax,
        seq,
        dirnum.data(),
        c_data.data(),
        l_data.data(),
        r_data.data(),
        qasave.data(),
        qbsave.data(),
        &payoff_sum);

    Vx = double(payoff_sum) / double(pmax);
    devx = 0;
    tk = 0;
}
