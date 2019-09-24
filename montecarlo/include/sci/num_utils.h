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
 * @file num_utils.h
 *
 */

#pragma once
#include <cmath>
#include <stdexcept>
#include <vector>

namespace sci
{

inline double LinearInterpolationUniform(
    std::vector<double> x_data,
    std::vector<double> y_data,
    double x_value,
    double delta_x)
{
    int index = ceil((x_value - x_data[0]) / delta_x);
    if (index <= 0)
        index = 1;
    if (index >= x_data.size())
        index = x_data.size() - 1;
    double f = (x_value - x_data[(index - 1)]) /
        (x_data[index] - x_data[(index - 1)]);
    double y_value = f * y_data[index] + (1 - f) * y_data[(index - 1)];
    return y_value;
}

inline void CholDecomp(
    const std::vector<std::vector<double>>& rho,
    std::vector<std::vector<double>>& Mcor)
{
    int i2, j1, k;
    double sum1;

    int n = rho.size();

    Mcor = rho;

    for (i2 = 0; i2 < n; i2++)
    {
        for (j1 = 0; j1 < n; j1++)
        {
            Mcor[i2][j1] = rho[i2][j1];
        }
    }
    for (i2 = 0; i2 < n; i2++)
    {
        sum1 = Mcor[i2][i2];
        for (k = i2 - 1; k >= 0; k--)
        {
            sum1 = sum1 - Mcor[i2][k] * Mcor[i2][k];
        }
        if (sum1 <= 0.)
        {
            throw std::runtime_error(
                "Non positive definite correlation matrix");
        }
        else
        {
            Mcor[i2][i2] = sqrt(sum1);
        }
        for (j1 = i2 + 1; j1 < n; j1++)
        {
            sum1 = Mcor[i2][j1];
            for (k = i2 - 1; k >= 0; k--)
            {
                sum1 = sum1 - Mcor[i2][k] * Mcor[j1][k];
            }
            Mcor[j1][i2] = sum1 / Mcor[i2][i2];
        }
    }
}

inline void getCorrelation(
    std::vector<std::vector<double>>& rho,
    const std::vector<std::vector<double>>& RhoSS,
    const std::vector<std::vector<double>>& RhoSv,
    const std::vector<std::vector<double>>& Rhovv)
{
    int nD = RhoSS.size();
    std::vector<double> row;
    for (int i = 0; i < nD; i++)
    {
        row = RhoSS[i];
        row.insert(row.end(), RhoSv[i].begin(), RhoSv[i].end());
        rho.push_back(row);
    }
    for (int i = 0; i < nD; i++)
    {
        row = RhoSv[i];
        row.insert(row.end(), Rhovv[i].begin(), Rhovv[i].end());
        rho.push_back(row);
    }
}

} // namespace sci
