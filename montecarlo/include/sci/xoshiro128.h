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

/*  Written in 2018 by David Blackman and Sebastiano Vigna (vigna@acm.org)

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide. This software is distributed without any warranty.

See <http://creativecommons.org/publicdomain/zero/1.0/>. */

/* http://xoshiro.di.unimi.it/xoshiro128starstar.c */

/* This is xoshiro128** 1.1, one of our 32-bit all-purpose, rock-solid
generators. It has excellent speed, a state size (128 bits) that is
large enough for mild parallelism, and it passes all tests we are aware
of.

Note that version 1.0 had mistakenly s[0] instead of s[1] as state
word passed to the scrambler.

For generating just single-precision (i.e., 32-bit) floating-point
numbers, xoshiro128+ is even faster.

The state must be seeded so that it is not everywhere zero. */

#pragma once

namespace sci
{

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

    void seed(uint seed0)
    {
        unsigned long long u1 = splitmix64(seed0);
        unsigned long long u2 = splitmix64(seed0);
        s[0] = (uint)((u1 & 0xFFFFFFFF00000000LL) >> 32);
        s[1] = (uint)(u1 & 0xFFFFFFFFLL);
        s[2] = (uint)((u2 & 0xFFFFFFFF00000000LL) >> 32);
        s[3] = (uint)(u2 & 0xFFFFFFFFLL);
    }

    void seed(uint state0, uint state1, uint state2, uint state3)
    {
        s[0] = state0;
        s[1] = state1;
        s[2] = state2;
        s[3] = state3;
    }

    void set_state(const uint state[4])
    {
        s[0] = state[0];
        s[1] = state[1];
        s[2] = state[2];
        s[3] = state[3];
    }
    void get_state(uint state[4])
    {
        state[0] = s[0];
        state[1] = s[1];
        state[2] = s[2];
        state[3] = s[3];
    }

    uint next32()
    {
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

    /* This is the jump function for the generator. It is equivalent
    to 2^64 calls to next(); it can be used to generate 2^64
    non-overlapping subsequences for parallel computations. */
    void jump(int njumps)
    {
        static const uint JUMP[] = {
            0x8764000bU, 0xf542d2d3U, 0x6fa035c3U, 0x77f2db5bU};
        uint s_out[4];

        for (int k = 0; k < njumps; k++)
        {
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
    }

private:
    uint rotl(const uint x, int k) { return (x << k) | (x >> (32 - k)); }

    /* This is a fixed-increment version of Java 8's SplittableRandom generator
    See http://dx.doi.org/10.1145/2714064.2660195 and
    http://docs.oracle.com/javase/8/docs/api/java/util/SplittableRandom.html
    */
    unsigned long long splitmix64(unsigned long long x)
    {
        unsigned long long z = (x += 0x9e3779b97f4a7c15LL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9LL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebLL;
        return z ^ (z >> 31);
    }
};

} // namespace sci
