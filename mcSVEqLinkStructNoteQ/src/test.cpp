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
    mcSVEqLinkStructNote pricer;
    pricer.load_inputs();

    int pmax = 1 << 10;
    int seq = 1;
    int nsub = 4;
    double V, dev;
    double tk;
    pricer.calc(pmax, seq, nsub, V, dev, tk);
    double V_ref = 9535.04296875;

    CHECK(V == Approx(V_ref).epsilon(0.000001));
    INFO("Kernel time: " << tk << " ms");
    CHECK(true);
}

TEST_CASE("paths_1M", "[.][integration]")
{
    mcSVEqLinkStructNote pricer;
    pricer.load_inputs();

    int pmax = 1 << 20;
    int seq = 3;
    int nsub = 4;
    double V, dev;
    double tk;
    pricer.calc(pmax, seq, nsub, V, dev, tk);
    double V_ref = 9452.6064453125;

    CHECK(V == Approx(V_ref).epsilon(0.000001));
    INFO("Kernel time: " << tk << " ms");
    CHECK(true);
}

TEST_CASE("convergence", "[.][integration]")
{
    double ctable_ref[][3] = {{16384, 9451.8542968750, 13.4842880844},
                              {32768, 9450.9028320313, 7.3111400970},
                              {65536, 9449.4882324219, 5.1044529019},
                              {131072, 9450.6322753906, 5.0512407408},
                              {262144, 9451.5402343750, 3.2119310726},
                              {524288, 9451.2572265625, 2.4476610185},
                              {1048576, 9451.0416992187, 1.2984638191}};
    double V_ref = 9451.0412597656;
    int ntable = sizeof(ctable_ref) / sizeof(double) / 3;

    mcSVEqLinkStructNote pricer;
    pricer.load_inputs();

    int seq = 1;
    int nsub = 4;
    double V, dev;
    double tk;
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
            pricer.calc(pmax, k, nsub, V, dev, tk);
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
        CHECK(V == Approx(ctable_ref[p][1]).epsilon(0.00001));
        CHECK(dev == Approx(ctable_ref[p][2]).epsilon(0.001));
    }
}
