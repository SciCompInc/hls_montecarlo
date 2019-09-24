#include "catch2/catch.hpp"
#include <iomanip>
#include <iostream>

#include <pricer_host.h>

TEST_CASE("test11", "[hls]")
{
    mcSVEqLinkStructNote pricer;
    pricer.load_inputs();

    int pmax = 1 << 10;
    int seq = 1;
    int nsub = 4;
    double V, dev;
    double tk;
    pricer.calc(pmax, seq, nsub, V, dev, tk);
    double V_ref = 9464.1630859375;

    CHECK(V == Approx(V_ref).epsilon(0.000001));
    INFO("Kernel time: " << tk << " ms");
    CHECK(true);
}

TEST_CASE("test13", "[.][integration]")
{
    double ctable_ref[][3] = {{16384, 9445.2792968750, 12.2280696165},
                              {32768, 9448.7847167969, 9.9039921271},
                              {65536, 9449.5662109375, 9.7391229381},
                              {131072, 9449.6002929687, 5.8395357108},
                              {262144, 9452.0356933594, 2.8051552449},
                              {524288, 9455.8560546875, 2.1083290403},
                              {1048576, 9455.8576660156, 1.6502308695}};

    // Scrambled Sobol
    //{ 16384,        9449.5375976563,        11.8104653618},
    //{ 32768,        9451.4765625000,        10.4761420499 },
    //{ 65536,        9449.7967773438,        8.9634402288 },
    //{ 131072,       9449.7084960938,        4.1855949456 },
    //{ 262144,       9452.7068359375,        3.3358144335 },
    //{ 524288,       9455.7122558594,        2.6232835375 },
    //{ 1048576,      9455.6075683594,        1.6183692420 },

    int ntable = sizeof(ctable_ref) / sizeof(double) / 3;

    double V_ref = 9455.8576660156; // pmax = 2^20

    mcSVEqLinkStructNote pricer;
    pricer.load_inputs();

    int seq = 1;
    int nsub = 4;
    double V, dev;
    double tk;
    std::cout << std::endl;
    std::cout << "pmax"
              << "\t"
              << "V"
              << "\t"
              << "dev"
              << "\t"
              << "|V - V_ref|"
              << "\t"
              << "Conf. Interval" << std::endl;
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
        // std::cout << std::setw(10) << std::fixed << std::setprecision(10) <<
        // pmax << "\t" << V << "\t" << dev << "\t" << std::fabs(V - V_ref) <<
        // "\t" << chk << std::endl;
        std::cout << std::setw(10) << std::fixed << std::setprecision(10)
                  << "{ " << pmax << ",\t" << V << ",\t" << dev << "},"
                  << std::endl;
        REQUIRE(V == Approx(ctable_ref[p][1]).epsilon(0.00001));
        REQUIRE(dev == Approx(ctable_ref[p][2]).epsilon(0.001));
    }
}
