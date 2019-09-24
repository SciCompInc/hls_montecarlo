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
 * Long tests are disabled by default using the "[.]" tag
 */

#include <catch2/catch.hpp>
#include <iomanip>
#include <iostream>

#include "pricer_host.h"

TEST_CASE("paths_1k", "[hls]")
{
    mcPi pricer;
    pricer.load_inputs();

    int pmax = 1 << 10;
    int seq = 1;
    double V, dev;
    double tk;
    pricer.calc(pmax, seq, V, dev, tk);
    double V_ref = 3.1171875;
    double dev_ref = 0.051840087;

    CHECK(V == Approx(V_ref).epsilon(0.000001));
    CHECK(dev == Approx(dev_ref).epsilon(0.000001));
    INFO("Kernel time: " << tk << " ms");
    CHECK(true);
}

TEST_CASE("convergence", "[.][integration]")
{
    double ctable_ref[][3] = {
        {16384, 3.1364746094, 0.0128572614},
        {32768, 3.1328125000, 0.0091053939},
        {65536, 3.1421508789, 0.0064132624},
        {131072, 3.1402587891, 0.0045384926},
        {262144, 3.1413269043, 0.0032077501},
        {524288, 3.1429061890, 0.0022667046},
        {1048576, 3.1439514160, 0.0016020909},
    };
    int ntable = sizeof(ctable_ref) / sizeof(double) / 3;

    mcPi pricer;
    pricer.load_inputs();

    int seq = 1;
    double V, dev;
    double tk;
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
        pricer.calc(pmax, seq, V, dev, tk);
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

TEST_CASE("Standard error estimation 100k paths", "[.][integration]")
{
    mcPi pricer;
    pricer.load_inputs();

    int pmax = 1 << 17;
    double V, dev;
    double tk;

    int nseries = 20;
    double sum = 0;
    double sum2 = 0;
    std::cout << std::endl
              << std::setw(12) << "V"
              << "\t" << std::setw(12) << "std.err" << std::endl;
    for (int k = 1; k <= nseries; k++)
    {
        pricer.calc(pmax, k, V, dev, tk);
        std::cout << V << "\t" << dev << std::endl;
        sum += V;
        sum2 += V * V;
    }
    sum /= nseries;
    V = sum;
    sum2 /= nseries;
    dev = sqrt(sum2 - sum * sum);
    std::cout << "Mean: " << V << std::endl;
    std::cout << "Std. dev: " << dev << std::endl;
    double V_ref = 3.1410247803;
    double dev_ref = 0.0042968262;

    CHECK(V == Approx(V_ref).epsilon(0.000001));
    CHECK(dev == Approx(dev_ref).epsilon(0.000001));
}
