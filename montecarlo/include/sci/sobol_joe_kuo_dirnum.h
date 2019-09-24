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
 * @file sobol_joe_kuo_dirnum.h
 *
 */

#pragma once
#include <random>
#include <sci/new_joe_kuo_6_21201.h>
#include <vector>

namespace sci
{
/**
 * \brief Function to get a subsequence of bits in 64-bit integer
 * \param n input 64-bit integer
 * \param pos starting position
 * \param length subsequence length
 * \return output subsequence
 */
static inline int64_t
bitsubseq(const int64_t n, const int64_t pos, const int64_t length)
{
    return (n >> pos) & ((1 << length) - 1);
}

static const unsigned int new_sobol_joe_kuo_6_21201_nbit = 32;

//  Initializer
/**
 * \brief Generate direction numbers for Sobol
 * \param ndim number of dimensions
 * \param dirnum output data pointer
 * \param shift initial shift vector
 * \param scramble scrambling flag
 * \param seed scrambling seed
 */
inline void new_sobol_joe_kuo_6_21201_dirnum(
    unsigned int ndim,
    unsigned int* dirnum,
    unsigned int* shift,
    bool scramble = false,
    unsigned int seed = 123)
{
    int nbit = new_sobol_joe_kuo_6_21201_nbit;
    /* first dimension (specical case P = 1) (m_i = 1) */
    for (unsigned int ibit = 1; ibit <= nbit; ++ibit)
    {
        dirnum[ibit - 1] = 1 << (nbit - ibit);
    }
    /* dimensions 2 though ndim */
    unsigned int minitpos = 0;
    for (unsigned int idim = 2; idim <= ndim; ++idim)
    {
        unsigned int pdeg = kPdeg[idim - 2];
        unsigned int pcoeff = kPcoeff[idim - 2];
        for (unsigned int ibit = 1; ibit <= pdeg; ++ibit)
        {
            dirnum[(idim - 1) * nbit + ibit - 1] = kMinit[minitpos++]
                << (nbit - ibit);
        } /* ibit */
        for (unsigned int ibit = pdeg + 1; ibit <= nbit; ++ibit)
        {
            dirnum[(idim - 1) * nbit + ibit - 1] =
                dirnum[(idim - 1) * nbit + ibit - 1 - pdeg] ^
                (dirnum[(idim - 1) * nbit + ibit - 1 - pdeg] >> pdeg);
            for (unsigned int k = 1; k <= pdeg - 1; ++k)
            {
                dirnum[(idim - 1) * nbit + ibit - 1] ^=
                    (((pcoeff >> (pdeg - 1 - k)) & 1) *
                     dirnum[(idim - 1) * nbit + ibit - 1 - k]);
            } /* k */
        } /* ibit */
    } /* idim */

    if (scramble)
    {
        std::mt19937 rng;
        std::uniform_int_distribution<> dist01(0, 1);
        rng.seed(seed);
        std::vector<unsigned int> smtx_(ndim * nbit);
        for (unsigned int idim = 0; idim < ndim; ++idim)
        {
            shift[idim] = 0.0;
            for (unsigned int ibit = 0; ibit < nbit; ++ibit)
            {
                shift[idim] += dist01(rng) * (1 << ibit);
            }
        }

        for (int64_t d = 0; d < ndim; ++d)
        {
            // [WxW] with diag=1 lower trangular, mult by pow2 vector of size
            // smtx_[d][0] = (1 << (W - 1));
            // smtx_[d][1] = dist01(rng) * (1 << (W - 1)) + (1 << (W - 2));
            // smtx_[d][2] = dist01(rng) * (1 << (W - 1)) + dist01(rng) * (1 <<
            // (W - 2)) + (1 << (W - 3));
            for (int64_t j1 = 0; j1 < nbit; ++j1)
            {
                smtx_[d * nbit + j1] = (1 << (nbit - 1 - j1));
                for (int64_t j2 = 0; j2 < j1; ++j2)
                {
                    smtx_[d * nbit + j1] +=
                        dist01(rng) * (1 << (nbit - 1 - j2));
                }
            }
        }

        for (int64_t d = 0; d < ndim; ++d)
        {
            for (int64_t j = 0; j < nbit; ++j)
            {
                int64_t vdj = dirnum[d * nbit + j], l = 1, t2 = 0;
                for (int64_t p = nbit - 1; p >= 0; --p)
                {
                    int64_t lsmdp = smtx_[d * nbit + p];
                    int64_t t1 = 0;
                    for (int64_t k = 0; k < nbit; ++k)
                    {
                        t1 += (bitsubseq(lsmdp, k, 1) * bitsubseq(vdj, k, 1));
                    }
                    t1 = t1 % 2;
                    t2 = t2 + t1 * l;
                    l = l << 1;
                }
                dirnum[d * nbit + j] = t2;
            }
        }
    }
    else
    {
        for (unsigned int idim = 0; idim < ndim; ++idim)
        {
            shift[idim] = 0;
        }
    }
}

} // namespace sci
