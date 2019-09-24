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
 * @file test.cpp
 *
 * Accuracy and convergence Monte Carlo tests
 */

#include <cmath>
#include <iomanip>
#include <iostream>

#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do
                          // this in one cpp file
#include <catch2/catch.hpp>

void kernel_wrapper(int niter, int seq, int& count);

void mc_pi(int niter, int seq, double& V)
{
    int count;
    kernel_wrapper(niter, seq, count);
    double ratio = double(count) / double(niter);
    V = 4.0 * ratio;
}

TEST_CASE("paths_1M", "[integration]")
{
    int pmax = 1 << 20;
    int seq = 1;
    double V;
    mc_pi(pmax, seq, V);
    double V_ref = 3.1416397095;

    CHECK(V == Approx(V_ref).epsilon(0.000001));
}

TEST_CASE("convergence", "[integration]")
{
    double ctable_ref[][3] = {{16384, 3.1415161133, 0.0011942320},
                              {32768, 3.1416503906, 0.0006264955},
                              {65536, 3.1417236328, 0.0004188809},
                              {131072, 3.1416717529, 0.0003658420},
                              {262144, 3.1415992737, 0.0001896192},
                              {524288, 3.1415897369, 0.0001523106},
                              {1048576, 3.1415922165, 0.0000317669}};
    int ntable = sizeof(ctable_ref) / sizeof(double) / 3;

    double V, dev;
    double V_ref = 3.14159265358979323846;
    std::cout << std::endl
              << std::setw(10) << "pmax"
              << "\t" << std::setw(12) << "V"
              << "\t" << std::setw(12) << "dev"
              << "\t" << std::setw(12) << "|V - V_ref|"
              << "\t"
              << "Conf. Int." << std::endl;

    int nseries = 20;
    for (int p = 0; p < ntable; p++)
    {
        int pmax = int(ctable_ref[p][0]);
        double sum = 0;
        double sum2 = 0;
        for (int k = 1; k <= nseries; k++)
        {
            mc_pi(pmax, k, V);
            sum += V;
            sum2 += V * V;
        }
        sum /= nseries;
        V = sum;
        sum2 /= nseries;
        dev = sqrt(sum2 - sum * sum);

        std::string chk = "FAIL";
        if (2.807 * dev > std::fabs(V - V_ref)) // 99.5% conf interval
        {
            chk = "OK";
        }
        std::cout << std::setw(10) << std::fixed << std::setprecision(10)
                  << pmax << "\t" << V << "\t" << dev << "\t"
                  << std::fabs(V - V_ref) << "\t" << chk << std::endl;
        // std::cout << std::setw(10) << std::fixed << std::setprecision(10) <<
        //"{ " << pmax << ",\t" << V << ",\t" << dev << "}," << std::endl;
        REQUIRE(V == Approx(ctable_ref[p][1]).epsilon(0.00001));
        REQUIRE(dev == Approx(ctable_ref[p][2]).epsilon(0.001));
    }
}
