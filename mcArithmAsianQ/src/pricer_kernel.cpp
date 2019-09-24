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
#include </sci/hls/brownian_bridge.h>
#include </sci/hls/sobol_joe_kuo.h>
#else
#include <sci/distributions.h>
#include <sci/hls/brownian_bridge.h>
#include <sci/hls/sobol_joe_kuo.h>
#endif

#include "kernel_global.h"

#define NUM_BLOCKS 4

#define NUM_SIMS 64

#define TS_MAX 12
// loop trip count estimations
#if defined(__SYNTHESIS__)
namespace
{
const int tc_steps = TS_MAX;
const int tc_num_blocks = NUM_BLOCKS;
const int tc_pmax = 1024 * 1024;
const int tc_ngroups = tc_pmax / (NUM_BLOCKS * NUM_SIMS);
} // namespace
#endif

// kernel input data structure
struct pricer_kernel_data {
    int steps;
    real_t dt;
    real_t vol;
    real_t r;
    real_t q;
    real_t spot;
    real_t strike;
    int call;
};

void pricer_kernel_block(
    sci::hls::sobol_joe_kuo<TS_MAX>& qrng,
    sci::hls::brownian_bridge<real_t, TS_MAX>& bb,
    pricer_kernel_data& data,
    int pmax,
    int seq,
    real_t& V,
    int block_id)
{
    qrng.skip((block_id + seq * NUM_BLOCKS) * pmax);

    unsigned int ZQ[NUM_SIMS][TS_MAX];
    real_t ZQN[NUM_SIMS][TS_MAX];
    real_t ZQB[NUM_SIMS][TS_MAX + 1];
    real_t ZQB2[NUM_SIMS][TS_MAX + 1];
    real_t payoffSum = 0;

    const int NUM_LAT = 16;
    real_t sum_V[NUM_LAT];
    // clang-format off
#pragma HLS array_partition variable=sum_V complete
    // clang-format on

    // init partial accumulators
loop_init_0:
    for (int j = 0; j < NUM_LAT; j++)
    {
        // clang-format off
#pragma HLS pipeline
        // clang-format on
        sum_V[j] = 0.0;
    }

    // path variables arrays
    real_t aveS[NUM_SIMS];
    real_t logS[NUM_SIMS];
    real_t payoffV[NUM_SIMS];

    // simulation constants
    real_t drift =
        (data.r - data.q - real_t(0.5) * data.vol * data.vol) * data.dt;
    real_t volz = data.vol;
    real_t discount = exp(-data.r * data.dt * data.steps);
    real_t logS0 = log(data.spot);

    // init path variables
loop_init_1:
    for (int k = 0; k < NUM_SIMS; k++)
    {
        // clang-format off
#pragma HLS pipeline
        // clang-format on
        logS[k] = logS0;
        aveS[k] = 0.0;
    }

loop_path_groups:
    for (int j = 0; j < pmax / NUM_SIMS; j++)
    {
        // clang-format off
#pragma HLS loop_tripcount min=tc_ngroups max=tc_ngroups
        // clang-format on
        // ---------------------------------------------------------------------------
        // Sobol QRNG
        // Order: steps first (outer: paths, inner: steps)
    loop_sims_01:
        for (int k = 0; k < NUM_SIMS; k++)
        {
            qrng.next(ZQ[k]);
        }
        // ---------------------------------------------------------------------------

        // ---------------------------------------------------------------------------
        // Normal Transform via ICDF
        // Order: any
    loop_time_02:
        for (int n = 0; n < data.steps; n++)
        {
#pragma HLS LOOP_TRIPCOUNT min = 12 max = 12
        loops_sims_02:
            for (int k = 0; k < NUM_SIMS; k++)
            {
#pragma HLS PIPELINE
                real_t xu;
                sci::uniform_distribution_transform(ZQ[k][n], xu);
                ZQN[k][n] = sci::normal_distribution_icdf(xu);
            } // k
        }
        // ---------------------------------------------------------------------------

        bb.transform(ZQN, ZQB, ZQB2, NUM_SIMS);
        // ---------------------------------------------------------------------------
        // Path generation
        // Order: paths first (outer: steps, inner: paths)
    loop_time:
        for (int n = 0; n < data.steps; n++)
        {
            // clang-format off
#pragma HLS loop_tripcount min=tc_steps max=tc_steps
            // clang-format on
        loop_sims_2:
            for (int k = 0; k < NUM_SIMS; k++)
            {
                // clang-format off
#pragma HLS pipeline
                // clang-format on
                real_t dz = ZQB[k][n + 1]; // BB includes 0, so use n + 1
                real_t dlogS = drift + volz * dz;
                logS[k] += dlogS;
                aveS[k] += exp(logS[k]);
            } // k
        } // n

        // Calculate payoffs
    loop_sims_3:
        for (int k = 0; k < NUM_SIMS; k++)
        {
            // clang-format off
#pragma HLS pipeline
            // clang-format on
            real_t aveS_tmp = aveS[k] / data.steps;
            real_t payoff;
            if (data.call)
            {
                payoff = aveS_tmp - data.strike;
            }
            else
            {
                payoff = data.strike - aveS_tmp;
            }
            if (payoff < 0)
                payoff = 0.0;
            payoffV[k] = payoff;

            // reset
            logS[k] = logS0;
            aveS[k] = 0.0;
        }

        // partial accumulation
    loop_sims_4:
        for (int k = 0; k < NUM_SIMS; k += NUM_LAT)
        {
#pragma HLS PIPELINE II = 9
        loop_sums_41:
            for (int j = 0; j < NUM_LAT; j++)
            {
                real_t payoff = payoffV[k + j];
                sum_V[j] += payoff;
            } // j
        } // k
    }
    // final reduction
loop_sum_all_0:
    for (int j = 0; j < NUM_LAT; j++)
    {
#pragma HLS pipeline II = 8
        payoffSum += sum_V[j];
    }
    payoffSum *= discount;
    V = payoffSum;
}

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
    real_t* payoff_sum)
{
    // clang-format off
#pragma HLS INTERFACE m_axi port=dirnum depth=tc_steps*32 offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=c_data depth=tc_steps+1 offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=l_data depth=tc_steps+1 offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=r_data depth=tc_steps+1 offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=qasave depth=tc_steps+1 offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=qbsave depth=tc_steps+1 offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=payoff_sum depth=1 offset=slave bundle=gmem

#pragma HLS INTERFACE s_axilite port=steps bundle=control
#pragma HLS INTERFACE s_axilite port=dt bundle=control
#pragma HLS INTERFACE s_axilite port=vol bundle=control
#pragma HLS INTERFACE s_axilite port=r bundle=control
#pragma HLS INTERFACE s_axilite port=q bundle=control
#pragma HLS INTERFACE s_axilite port=spot bundle=control
#pragma HLS INTERFACE s_axilite port=strike bundle=control
#pragma HLS INTERFACE s_axilite port=call bundle=control
#pragma HLS INTERFACE s_axilite port=pmax bundle=control
#pragma HLS INTERFACE s_axilite port=seq bundle=control
#pragma HLS INTERFACE s_axilite port=payoff_sum bundle=control
#pragma HLS INTERFACE s_axilite port=dirnum bundle=control
#pragma HLS INTERFACE s_axilite port=c_data bundle=control
#pragma HLS INTERFACE s_axilite port=l_data bundle=control
#pragma HLS INTERFACE s_axilite port=r_data bundle=control
#pragma HLS INTERFACE s_axilite port=qasave bundle=control
#pragma HLS INTERFACE s_axilite port=qbsave bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    // clang-format off

    // locals
    sci::hls::sobol_joe_kuo < TS_MAX > qrng_;
    sci::hls::brownian_bridge<real_t, TS_MAX> bb_;

    qrng_.init(dirnum, steps);
    bb_.init(steps, c_data, l_data, r_data, qasave, qbsave);

    // initialize input parameters structure
    pricer_kernel_data data;
    data.steps = steps;
    data.dt = dt;
    data.vol = vol;
    data.r = r;
    data.q = q;
    data.spot = spot;
    data.strike = strike;
    data.call = call;

    real_t payoffSumTot = 0;

    real_t payoffSum[NUM_BLOCKS], payoffSumE[NUM_BLOCKS];
    pricer_kernel_data data_[NUM_BLOCKS];
    sci::hls::sobol_joe_kuo < TS_MAX > qrng[NUM_BLOCKS];
    sci::hls::brownian_bridge<real_t, TS_MAX> bb[NUM_BLOCKS];
    // clang-format off
#pragma HLS ARRAY_PARTITION variable=payoffSum complete
#pragma HLS ARRAY_PARTITION variable=data_ complete
#pragma HLS ARRAY_PARTITION variable=qrng complete
#pragma HLS ARRAY_PARTITION variable=bb complete
    // clang-format on

    // replicate input data for each calculation block
    for (int i = 0; i < NUM_BLOCKS; i++)
    {
        // clang-format off
#pragma HLS unroll
        // clang-format on
        data_[i] = data;
        qrng[i] = qrng_;
        bb[i] = bb_;
    }

    // run calculation blocks in parallel (full unroll)
    for (int i = 0; i < NUM_BLOCKS; i++)
    {
        // clang-format off
#pragma HLS unroll
        // clang-format on
        pricer_kernel_block(
            qrng[i], bb[i], data_[i], pmax / NUM_BLOCKS, seq, payoffSum[i], i);
    }

    for (int i = 0; i < NUM_BLOCKS; i++)
    {
        payoffSumTot += payoffSum[i];
    }

    *payoff_sum = payoffSumTot;
}
