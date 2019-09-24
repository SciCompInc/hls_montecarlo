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
 * @file distributions.h
 *
 * Distribution transform function
 */
#pragma once

#include <cmath>

namespace sci
{
// Random floating point numbers in [0,1)
// https://experilous.com/1/blog/post/perfect-fast-random-floating-point-numbers

inline float as_float(unsigned int i)
{
    union {
        unsigned int i;
        float f;
    } pun = {i};
    return pun.f;
}

/**
 * @brief Uniform distribution transfrom from 32-integer to [0,1)
 *
 * @tparam T data type.
 * @p u input 32-bit random number
 * @p xu uniform random number in [0,1) interval
 */
template<typename T>
inline void uniform_distribution_transform(unsigned int u, T& xu)
{
}

template<>
inline void uniform_distribution_transform<float>(unsigned int u, float& xu)
{
    if (u <= 0x000001FFU)
    {
        // If true, then the highest 23 bits must all be zero.
        u = 0x00000200; // set fisrt bit
    }
    xu = as_float(0x3F800000U | (u >> 9)) - 1.0f;
}

template<>
inline void uniform_distribution_transform<double>(unsigned int u, double& xu)
{
    static const double d_2pow32_inv = 2.3283064365386963e-10;
    xu = double(u) * d_2pow32_inv;
}

/**
 * @brief Inverse CumulativeNormal using Acklam's approximation to transform
 * uniform random number to normal random number.
 *
 * Reference: Acklam's approximation: by Peter J. Acklam, University of Oslo,
 * Statistics Division.
 *
 * @tparam T data type.
 * @p input input uniform random number
 * @return normal random number
 */
template<class T>
inline T normal_distribution_icdf(T p)
{
    static const T a1 = T(-3.969683028665376e+01);
    static const T a2 = T(2.209460984245205e+02);
    static const T a3 = T(-2.759285104469687e+02);
    static const T a4 = T(1.383577518672690e+02);
    static const T a5 = T(-3.066479806614716e+01);
    static const T a6 = T(2.506628277459239e+00);
    static const T b1 = T(-5.447609879822406e+01);
    static const T b2 = T(1.615858368580409e+02);
    static const T b3 = T(-1.556989798598866e+02);
    static const T b4 = T(6.680131188771972e+01);
    static const T b5 = T(-1.328068155288572e+01);
    static const T c1 = T(-7.784894002430293e-03);
    static const T c2 = T(-3.223964580411365e-01);
    static const T c3 = T(-2.400758277161838e+00);
    static const T c4 = T(-2.549732539343734e+00);
    static const T c5 = T(4.374664141464968e+00);
    static const T c6 = T(2.938163982698783e+00);
    static const T d1 = T(7.784695709041462e-03);
    static const T d2 = T(3.224671290700398e-01);
    static const T d3 = T(2.445134137142996e+00);
    static const T d4 = T(3.754408661907416e+00);

    T q = p < (T(1) - p) ? p : (T(1) - p);
    T u;
    if (q > T(0.02425))
    {
        // Central region
        T d = q - T(0.5);
        T t = d * d;
        u = d * (((((a1 * t + a2) * t + a3) * t + a4) * t + a5) * t + a6) /
            (((((b1 * t + b2) * t + b3) * t + b4) * t + b5) * t + T(1));
    }
    else
    {
        // Tail region
        T t = std::sqrt(-T(2) * std::log(q));
        u = (((((c1 * t + c2) * t + c3) * t + c4) * t + c5) * t + c6) /
            ((((d1 * t + d2) * t + d3) * t + d4) * t + T(1));
    }

    return p > T(0.5) ? -u : u;
}
} // namespace sci
