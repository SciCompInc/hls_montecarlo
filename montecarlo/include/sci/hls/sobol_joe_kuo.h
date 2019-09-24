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
 * @file sobol_joe_kuo.h
 *
 */

#pragma once

namespace sci
{
namespace hls
{

#ifdef _MSC_VER
#include <intrin.h>
    /**
     * \brief Search from LSB to MSB for first set bit
     * \param mask input mask
     * \return first set bit or zero if no set bit is found
     */
    inline int __builtin_ctzl(unsigned long mask)
    {
        unsigned long where;
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

    /**
     * \brief Most significant zero bit
     * \param n input 32-bit integer
     * \return __builtin_ffs(~n) - 1
     */
    inline unsigned rmzb(unsigned n)
    {
#pragma HLS INLINE
        unsigned n1 = ~n;
        return (n1 == 0) ? 0 : __builtin_ctz(n1);
    }

    /**
     * \brief Sobol sequence class
     * \tparam DIM maximum number of dimensions
     *
     *  Direction numbers up to dim = 21201
     */
    template<int DIM>
    class sobol_joe_kuo
    {
    public:
        // Bit width of element in state vector
        const static int W = 32;

        //  State Y
        unsigned int myState[DIM];

        //  Current index in the sequence
        unsigned int myIndex;

        unsigned dir_num_[DIM][W];

        unsigned int size_;

    public:
        sobol_joe_kuo()
        {
            size_ = 0;
            myIndex = 0;
        }

        sobol_joe_kuo(unsigned int* dirnum, unsigned int size = DIM)
        {
            init(dirnum, size);
        }

        /**
         * \brief Sobol sequence initializer (using direction numbers table)
         * \param dirnum direction numbers data (dimensions by bits, row-wise)
         * \param size number of dimensions
         */
        void init(unsigned int* dirnum, unsigned int size = DIM)
        {
            size_ = size;
            for (unsigned int idim = 0; idim < size_; ++idim)
            {
// clang-format off
#pragma HLS loop_tripcount min=DIM max=DIM
                // clang-format on
                for (unsigned int ibit = 0; ibit < W; ++ibit)
                {
// clang-format off
#pragma HLS pipeline
                    // clang-format on
                    dir_num_[idim][ibit] = dirnum[idim * W + ibit];
                }
            }
            //  Reset to 0
            reset();
        }

        /**
         * \brief reset generator state
         */
        void reset()
        {
        //  Set state to 0
        loop_reset:
            for (unsigned int idim = 0; idim < size_; ++idim)
            {
// clang-format off
#pragma HLS loop_tripcount min=DIM max=DIM
                // clang-format on
                myState[idim] = 0;
            }
            //  Set index to 0
            myIndex = 0;
        }

        /**
         * \brief Get next sequence element
         * \param x output vector for the sequence element
         */
        void next(unsigned int x[DIM])
        {
            unsigned n = myIndex, j = 0;
            j = rmzb(n);

        loop_next:
            for (unsigned int i = 0; i < size_; ++i)
            {
// clang-format off
#pragma HLS loop_tripcount min=DIM max=DIM
#pragma HLS pipeline
                // clang-format on
                myState[i] ^= dir_num_[i][j];
                x[i] = myState[i];
            }

            //	Update count
            ++myIndex;
        }

        /**
         * \brief Sequence skip-ahead (fast-forward)
         * \param b number of elements to skip
         */
        void skip(const unsigned b)
        {
            //	Check skip
            if (!b)
                return;

            //	Reset Sobol to 0
            reset();

            //	The actual Sobol skipping algo
            unsigned im = b;
            unsigned two_i = 1, two_i_plus_one = 2;

            unsigned i = 0;
        loop_skipto_1:
            while (two_i <= im)
            {
// clang-format off
#pragma HLS LOOP_TRIPCOUNT min=0 max=W-1
                // clang-format on
                if (((im + two_i) / two_i_plus_one) & 1)
                {
                loop_skipto_2:
                    for (unsigned k = 0; k < size_; ++k)
                    {
// clang-format off
#pragma HLS loop_tripcount min=DIM max=DIM
                        // clang-format on
                        myState[k] ^= dir_num_[k][i];
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

} // namespace hls

} // namespace sci
