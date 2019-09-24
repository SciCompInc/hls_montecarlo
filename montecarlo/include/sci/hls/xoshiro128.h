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
 * @file xoshiro128.h
 *
 */
#pragma once

namespace sci
{
namespace hls
{

    /**
     * \brief xoshiro128** by David Blackman and Sebastiano Vigna
     *
     * Ported from: http://xoshiro.di.unimi.it/xoshiro128starstar.c
     */
    class xoshiro128starstar
    {
        typedef unsigned int uint;

    private:
        uint s[4];

    public:
        xoshiro128starstar()
        {
            seed(0xA6E9377DU, 0xAF75BDFEU, 0x863F5CB5U, 0x08510D95U);
        }
        xoshiro128starstar(uint seed0) { seed(seed0); }

        /**
         * \brief Generate state from a single 32-bit seed
         * \param seed0 single seed
         *
         * Using Java 8's SplittableRandom 64-bt generator to
         * create the initial state
         */
        void seed(uint seed0)
        {
            unsigned long long u1 = splitmix64(seed0);
            unsigned long long u2 = splitmix64(seed0);
            s[0] = (uint)((u1 & 0xFFFFFFFF00000000LL) >> 32);
            s[1] = (uint)(u1 & 0xFFFFFFFFLL);
            s[2] = (uint)((u2 & 0xFFFFFFFF00000000LL) >> 32);
            s[3] = (uint)(u2 & 0xFFFFFFFFLL);
        }

        /**
         * \brief State initializer with scalar values
         * \param state0 state bits 0-31
         * \param state1 state bits 32-63
         * \param state2 state bits 64-95
         * \param state3 state bits 96-127
         */
        void seed(uint state0, uint state1, uint state2, uint state3)
        {
            s[0] = state0;
            s[1] = state1;
            s[2] = state2;
            s[3] = state3;
        }

        /**
         * \brief Set state
         * \param state
         */
        void set_state(const uint state[4])
        {
            s[0] = state[0];
            s[1] = state[1];
            s[2] = state[2];
            s[3] = state[3];
        }

        /**
         * \brief Get state
         * \param state
         */
        void get_state(uint state[4])
        {
            state[0] = s[0];
            state[1] = s[1];
            state[2] = s[2];
            state[3] = s[3];
        }

        /**
         * \brief Get next element from the sequence
         * \return 32-bit random integer
         */
        uint next32()
        {
// clang-format off
#pragma HLS inline
            // clang-format on
            const uint result_starstar = rotl(s[1] * 5U, 7U) * 9U;

            const uint t = s[1] << 9U;

            s[2] ^= s[0];
            s[3] ^= s[1];
            s[1] ^= s[2];
            s[0] ^= s[3];

            s[2] ^= t;

            s[3] = rotl(s[3], 11U);

            return result_starstar;
        }

        /**
         * \brief Sequence jump ahead (fast-forward)
         *
         * This is the jump function for the generator. It is equivalent
         * to 2^64 calls to next(); it can be used to generate 2^64
         * non-overlapping subsequences for parallel computations.
         */
        void jump()
        {
// clang-format off
#pragma HLS inline
            // clang-format on
            static const uint JUMP[] = {
                0x8764000bU, 0xf542d2d3U, 0x6fa035c3U, 0x77f2db5bU};
            uint s_out[4];

// clang-format off
#pragma HLS dependence variable=s array intra RAW true
            // clang-format on
            s_out[0] = 0;
            s_out[1] = 0;
            s_out[2] = 0;
            s_out[3] = 0;
            for (int i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
            {
                for (int b = 0; b < 32; b++)
                {
                    if (JUMP[i] & 1U << b)
                    {
                        s_out[0] ^= s[0];
                        s_out[1] ^= s[1];
                        s_out[2] ^= s[2];
                        s_out[3] ^= s[3];
                    }
                    next32();
                }
            }
            s[0] = s_out[0];
            s[1] = s_out[1];
            s[2] = s_out[2];
            s[3] = s_out[3];
        }

    private:
        /**
         * \brief Bit rotation utility function
         * \param x input 32-bit integer
         * \param k rotation number
         * \return output number
         */
        uint rotl(const uint x, int k) { return (x << k) | (x >> (32 - k)); }

        /**
         * \brief Fixed-increment version of Java 8's SplittableRandom
         * \param x 64-bit state
         * \return 64-bit random number
         *
         * See http://dx.doi.org/10.1145/2714064.2660195 and
         * http://docs.oracle.com/javase/8/docs/api/java/util/SplittableRandom.html
         */
        unsigned long long splitmix64(unsigned long long x)
        {
            unsigned long long z = (x += 0x9e3779b97f4a7c15LL);
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9LL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebLL;
            return z ^ (z >> 31);
        }
    };

} // namespace hls

} // namespace sci
