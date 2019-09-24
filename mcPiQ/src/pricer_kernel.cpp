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
 * @file pricer_kernel.cpp
 *
 * HLS kernel function
 */

#include <hls_math.h>

#if defined(__GNUC__) && defined(_WIN32) && defined(__SYNTHESIS__)
// Vivado HLS 2018.3 GUI console output hack
#include </sci/distributions.h>
#include </sci/hls/sobol_joe_kuo.h>
#else
#include <sci/distributions.h>
#include <sci/hls/sobol_joe_kuo.h>
#endif

#include "kernel_global.h"

#define NUM_BLOCKS 4

#define NUM_SIMS 64

// loop trip count estimations
#if defined(__SYNTHESIS__)
namespace
{
const int tc_pmax = 1024 * 1024;
const int tc_ngroups = tc_pmax / (NUM_BLOCKS * NUM_SIMS);
} // namespace
#endif

void pricer_kernel_block(
    sci::hls::sobol_joe_kuo<2>& qrng,
    int pmax,
    int seq,
    int& sumx,
    int block_id)
{
    qrng.skip((block_id + seq * NUM_BLOCKS) * pmax);

    unsigned int zq[NUM_SIMS][2];
    // path variables arrays
    int circle[NUM_SIMS];

loop_sims_0:
    for (int k = 0; k < NUM_SIMS; k++)
    {
        // clang-format off
#pragma HLS pipeline
        // clang-format on
        circle[k] = 0;
    }

loop_path_groups:
    for (int j = 0; j < pmax / NUM_SIMS; j++)
    {
        // clang-format off
#pragma HLS loop_tripcount min=tc_ngroups max=tc_ngroups
    // clang-format on

    // Run simulation with path-wise pipelining
    loop_sims_1:
        for (int k = 0; k < NUM_SIMS; k++)
        {
            qrng.next(zq[k]);
        } // k

        // Run simulation with path-wise pipelining
    loop_sims_2:
        for (int k = 0; k < NUM_SIMS; k++)
        {
            // clang-format off
#pragma HLS pipeline
            // clang-format on
            // generate a random shock
            real_t zu1, zu2;
            sci::uniform_distribution_transform(zq[k][0], zu1);
            sci::uniform_distribution_transform(zq[k][1], zu2);
            if (zu1 * zu1 + zu2 * zu2 < real_t(1.0))
                circle[k] += 1;
        } // k
    }

loop_reduce:
    int sum = 0;
    for (int k = 0; k < NUM_SIMS; k++)
    {
        sum += circle[k];
    }
    sumx = sum;
}

extern "C" void
pricer_kernel(int pmax, int seq, unsigned int* dirnum, int* sump)
{
    // clang-format off
#pragma HLS INTERFACE m_axi port=dirnum depth=2*32 offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=sump offset=slave bundle=gmem

#pragma HLS INTERFACE s_axilite port=pmax bundle=control
#pragma HLS INTERFACE s_axilite port=seq bundle=control
#pragma HLS INTERFACE s_axilite port=sump bundle=control
#pragma HLS INTERFACE s_axilite port=dirnum bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
	// clang-format off

    // locals
    sci::hls::sobol_joe_kuo<2> qrng_;

    qrng_.init(dirnum, 2);

    int sum[NUM_BLOCKS];
    sci::hls::sobol_joe_kuo<2> qrng[NUM_BLOCKS];
    // clang-format off
#pragma HLS ARRAY_PARTITION variable=sum complete
#pragma HLS ARRAY_PARTITION variable=qrng complete
    // clang-format on

    // replicate input data for each calculation block
    for (int i = 0; i < NUM_BLOCKS; i++)
    {
        // clang-format off
#pragma HLS unroll
        // clang-format on
        qrng[i] = qrng_;
    }

    // run calculation blocks in parallel (full unroll)
    for (int i = 0; i < NUM_BLOCKS; i++)
    {
        // clang-format off
#pragma HLS unroll
        // clang-format on
        pricer_kernel_block(qrng[i], pmax / NUM_BLOCKS, seq, sum[i], i);
    }

    int sumTot = 0;
    for (int i = 0; i < NUM_BLOCKS; i++)
    {
        sumTot += sum[i];
    }

    *sump = sumTot;
}
