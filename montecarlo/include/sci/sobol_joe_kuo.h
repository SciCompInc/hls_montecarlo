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
 * @file sobol_kuo6.h
 * @brief sobol sequences direction numbers up to dim = 21201
 *
 */

#pragma once

namespace sci
{

#ifdef _MSC_VER
#include <intrin.h>
inline int __builtin_ctzl(unsigned long mask)
{
    unsigned long where;
    // Search from LSB to MSB for first set bit.
    // Returns zero if no set bit is found.
    if (_BitScanForward(&where, mask))
        return static_cast<int>(where);
    return 32;
}
inline int __builtin_ctz(unsigned int mask)
{
    // Win32 and Win64 expectations.
    static_assert(sizeof(mask) == 4, "");
    static_assert(sizeof(unsigned long) == 4, "");
    return __builtin_ctzl(static_cast<unsigned long>(mask));
}
#endif

inline unsigned rmzb(unsigned n)
{
    //	return __builtin_ffs(~n) - 1;
    unsigned n1 = ~n;
    return (n1 == 0) ? 0 : __builtin_ctz(n1);
}

class sobol_joe_kuo
{
public:
    // Bit width of element in state vector
    const static int W = 32;

    //  State Y
    unsigned int* myState;

    //  Current index in the sequence
    unsigned int myIndex;

    unsigned int* dir_num_;

    unsigned int size_;

public:
    sobol_joe_kuo()
    {
        size_ = 0;
        myIndex = 0;
    }

    sobol_joe_kuo(unsigned int* dirnum, unsigned int* state, unsigned int size)
    {
        init(dirnum, state, size);
    }
    //  Initializer
    void init(unsigned int* dirnum, unsigned int* state, unsigned int size)
    {
        size_ = size;
        myState = state;
        dir_num_ = dirnum;
        myIndex = 0;
    }

    //	Next point
    void next()
    {
        //	Gray code, find position j
        //		of rightmost zero bit of current index n
        unsigned n = myIndex, j = 0;
        j = rmzb(n);

        //	XOR the appropriate direction number
        //		into each component of the integer sequence
        for (unsigned int i = 0; i < size_; ++i)
        {
            myState[i] ^= dir_num_[i * W + j];
        }

        //	Update count
        ++myIndex;
    }

    //  Skip ahead (from 0 to b)
    void skip(const unsigned b)
    {
        //	Check skip
        if (!b)
            return;

        myIndex = 0;

        //	The actual Sobol skipping algo
        unsigned im = b;
        unsigned two_i = 1, two_i_plus_one = 2;

        unsigned i = 0;
        while (two_i <= im)
        {
            if (((im + two_i) / two_i_plus_one) & 1)
            {
                for (unsigned k = 0; k < size_; ++k)
                {
                    myState[k] ^= dir_num_[k * W + i];
                }
            }

            two_i <<= 1;
            two_i_plus_one <<= 1;
            ++i;
        }

        //	End of skipping algo
        myIndex = unsigned(b);
    }
};

} // namespace sci
