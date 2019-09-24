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

void kernel_wrapper(int niter, int& count);

void mc_pi(int niter, double& V, double& dev)
{
    int count;
    kernel_wrapper(niter, count);
    double ratio = double(count) / double(niter);
    V = 4.0 * ratio;
    // standard error estimate
    dev = 4.0 * std::sqrt((ratio - ratio * ratio) / double(niter));
}

TEST_CASE("paths_1M", "[integration]")
{
    int pmax = 1 << 20;
    double V, dev;
    mc_pi(pmax, V, dev);
    double V_ref = 3.1430091858;
    double dev_ref = 0.0016027322;

    CHECK(V == Approx(V_ref).epsilon(0.000001));
    CHECK(dev == Approx(dev_ref).epsilon(0.000001));
}

TEST_CASE("convergence", "[integration]")
{
    double ctable_ref[][3] = {
        {16384, 3.1535644531, 0.0127640305},
        {32768, 3.1352539063, 0.0090961098},
        {65536, 3.1488647461, 0.0063949378},
        {131072, 3.1458129883, 0.0045278077},
        {262144, 3.1455383301, 0.0032020184},
        {524288, 3.1456375122, 0.0022640732},
        {1048576, 3.1430091858, 0.0016027322},
    };
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
    for (int p = 0; p < ntable; p++)
    {
        int pmax = int(ctable_ref[p][0]);
        // call pricing engine
        mc_pi(pmax, V, dev);
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
        REQUIRE(V == Approx(ctable_ref[p][1]).epsilon(0.000001));
        REQUIRE(dev == Approx(ctable_ref[p][2]).epsilon(0.000001));
    }
}
