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
    mcArithmAsian pricer;
    pricer.load_inputs();

    int pmax = 1 << 10;
    int seq = 1;
    double V, dev;
    double tk;
    pricer.calc(pmax, seq, V, dev, tk);
    double V_ref = 2.1983985901;
    double dev_ref = 0.1104721236;

    CHECK(V == Approx(V_ref).epsilon(0.000001));
    CHECK(dev == Approx(dev_ref).epsilon(0.000001));
    INFO("Kernel time: " << tk << " ms");
    CHECK(true);
}

TEST_CASE("convergence", "[.][integration]")
{
    double ctable_ref[][3] = {
        {16384, 2.1219377518, 0.0263838284},
        {32768, 2.1045215130, 0.0185048834},
        {65536, 2.1003942490, 0.0130075479},
        {131072, 2.1162571907, 0.0092533268},
        {262144, 2.1187083721, 0.0065361548},
        {524288, 2.1162610054, 0.0046155791},
        {1048576, 2.1166129112, 0.0032631770},
    };
    int ntable = sizeof(ctable_ref) / sizeof(double) / 3;

    mcArithmAsian pricer;
    pricer.load_inputs();

    int seq = 1;
    double V, dev;
    double tk;
    double V_ref = 2.1116903801; // Analytics (Curran approx)
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
    mcArithmAsian pricer;
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
    double V_ref = 2.1130517244;
    double dev_ref = 0.008331028;

    CHECK(V == Approx(V_ref).epsilon(0.000001));
    CHECK(dev == Approx(dev_ref).epsilon(0.000001));
}
