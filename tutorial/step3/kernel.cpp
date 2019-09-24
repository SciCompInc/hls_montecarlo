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
 * @file kernel.cpp
 *
 * HLS kernel function
 */

#include <sci/distributions.h>
#include <sci/hls/xoshiro128.h>

typedef float real_t;

#define NUM_BLOCKS 4

void mc_kernel_block(int pmax, int& count, int block_id)
{
    sci::hls::xoshiro128starstar rng;

    // block-splitting random sequence
    const unsigned int seed = 127;
    rng.seed(seed);
    for (int j = 0; j < block_id + 1; j++)
    {
        rng.jump();
    }

    int sum = 0;

    for (int j = 0; j < pmax; j++)
    {
        // generate a random shock
        real_t zu1, zu2;
        sci::uniform_distribution_transform(rng.next32(), zu1);
        sci::uniform_distribution_transform(rng.next32(), zu2);
        if (zu1 * zu1 + zu2 * zu2 < real_t(1.0))
            sum += 1;
    }

    count = sum;
}

void mc_kernel(int pmax, int* countp)
{
    int sumTot = 0;

    int sum[NUM_BLOCKS];
    // run calculation blocks in parallel (full unroll)
    for (int i = 0; i < NUM_BLOCKS; i++)
    {
        mc_kernel_block(pmax / NUM_BLOCKS, sum[i], i);
    }

    for (int i = 0; i < NUM_BLOCKS; i++)
    {
        sumTot += sum[i];
    }

    *countp = sumTot;
}
