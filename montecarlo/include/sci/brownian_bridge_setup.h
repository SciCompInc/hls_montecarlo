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
 * @file brownian_bridge_setup.cpp
 *
 */

#pragma once

#include <cmath>
#include <vector>

namespace sci
{

/**
 * \brief Brownian Bridge setup
 * \tparam Real floating point type
 * \param nsamp number of samples
 * \param tsample pointer to samples array
 * \param c transformation data
 * \param l transformation data
 * \param r transformation data
 * \param qasave transformation data
 * \param qbsave transformation data
 */
template<typename Real>
inline void brownian_bridge_setup(
    int nsamp,
    Real* tsample,
    int* c,
    int* l,
    int* r,
    Real* qasave,
    Real* qbsave)
{
    int p, i, j, k, d, tmp;
    std::vector<int> tmp1(nsamp + 1);
    std::vector<int> tmp2(nsamp + 1);

    /* initialize */
    for (i = 0; i <= nsamp; i++)
    {
        c[i] = 0;
        r[i] = 0;
        l[i] = 0;
    }

    /* Find smallest p such that nsamp < 2**p */

    p = 0;
    tmp = 1;
    while (tmp < nsamp)
    {
        tmp = tmp * 2;
        p = p + 1;
    }

    /* Build Bridge up to 2**(p-1) using parent/child algorithm */

    k = 2;
    d = 1;
    l[1] = 0;
    r[1] = nsamp;
    c[1] = l[1] + (int)(r[1] - l[1]) / 2;
    tmp = 1 << (p - 1);
    while (k < tmp)
    {
        l[k] = l[d];
        r[k] = c[d];
        c[k] = l[k] + (int)(r[k] - l[k]) / 2;
        k = k + 1;
        l[k] = c[d];
        r[k] = r[d];
        c[k] = l[k] + (int)(r[k] - l[k]) / 2;
        k = k + 1;
        d = d + 1;
    }

    /* Flag all integers used by the bridge up to 2**(p-1) */
    std::fill(tmp1.begin(), tmp1.end(), 0);
    for (i = 1; i <= tmp; i++)
    {
        for (k = 1; k <= nsamp; k++)
        {
            if (c[i] == k)
            {
                tmp1[k] = 1;
            }
        }
    }
    tmp1[0] = 1;
    tmp1[nsamp] = 1;

    /* Build array of remaining (unused) integers*/

    std::fill(tmp2.begin(), tmp2.end(), 0);
    d = 1;
    for (i = 1; i < nsamp; i++)
    {
        if (tmp1[i] == 0)
        {
            tmp2[d] = i;
            d = d + 1;
        }
    }

    /* Finish bridge using remainder's and parent child relationship*/
    /* on first come first served basis.*/

    k = 1 << (p - 1);
    d = tmp - (int)tmp / 2;
    i = 1;
    j = 1;
    while (i <= nsamp)
    {
        if ((tmp1[i - 1] + tmp1[i]) == 1)
        {
            l[k] = i - 1;
            r[k] = i + 1;
            c[k] = tmp2[j];
            k = k + 1;
            j = j + 1;
            i = i + 1;
        }
        i = i + 1;
    }

    qasave[0] = sqrt(tsample[nsamp] - tsample[0]);
    for (i = 1; i < nsamp; i++)
    {
        qasave[i] =
            (tsample[r[i]] - tsample[c[i]]) / (tsample[r[i]] - tsample[l[i]]);
        qbsave[i] = sqrt((tsample[c[i]] - tsample[l[i]]) * qasave[i]);
    }
}

} // namespace sci
