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
 * @file kernel_wrapper.cpp
 *
 */
#include <sci/sobol_joe_kuo_dirnum.h>
#include <vector>

extern "C" void
mc_kernel(int pmax, int seq, unsigned int* dirnum, int* countp);

void kernel_wrapper(int niter, int seq, int& count)
{
    const int nbit = sci::new_sobol_joe_kuo_6_21201_nbit;
    const int ndim = 2;
    std::vector<unsigned int> dirnum(ndim * nbit);
    std::vector<unsigned int> shift(ndim);
    sci::new_sobol_joe_kuo_6_21201_dirnum(ndim, dirnum.data(), shift.data());

    mc_kernel(niter, seq, dirnum.data(), &count);
}
