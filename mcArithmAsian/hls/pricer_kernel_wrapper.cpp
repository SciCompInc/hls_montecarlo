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
    real_t* payoff_sum,
    real_t* payoff_sum2);

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
    real_t payoff_sum, payoff_sum2;

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
        &payoff_sum,
        &payoff_sum2);

    Vx = double(payoff_sum) / double(pmax);
    devx = (double(payoff_sum2) / double(pmax) - Vx * Vx);
    devx = std::sqrt(devx / double(pmax));
    tk = 0;
}
