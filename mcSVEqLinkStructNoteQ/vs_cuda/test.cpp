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
    double V_ref = 9465.375400093;

    CHECK(V == Approx(V_ref).epsilon(0.000001));
    INFO("Kernel time: " << tk << " ms");
    CHECK(true);
}

TEST_CASE("test13", "[.][integration]")
{
    double ctable_ref[][3] = {{16384, 9445.3555802495, 12.1483411877},
                              {32768, 9448.6787521097, 9.9015511130},
                              {65536, 9449.2807608982, 9.7432465509},
                              {131072, 9449.5807788842, 5.8365788561},
                              {262144, 9450.0867069772, 2.8041544304},
                              {524288, 9449.9823870366, 2.1017164933},
                              {1048576, 9450.2580873787, 1.6421200026}};

    double V_ref = 9451.0412597656;
    int ntable = sizeof(ctable_ref) / sizeof(double) / 3;

    mcSVEqLinkStructNote pricer;
    pricer.load_inputs();

    int seq = 1;
    int nsub = 4;
    double V, dev;
    double tk;
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
        // std::cout << std::setw(10) << std::fixed << std::setprecision(10)
        //          << pmax << "\t" << V << "\t" << dev << "\t"
        //          << std::fabs(V - V_ref) << "\t" << chk << std::endl;
        std::cout << std::setw(10) << std::fixed << std::setprecision(10)
                  << "{ " << pmax << ",\t" << V << ",\t" << dev << "},"
                  << std::endl;
        REQUIRE(V == Approx(ctable_ref[p][1]).epsilon(0.00001));
        REQUIRE(dev == Approx(ctable_ref[p][2]).epsilon(0.001));
    }
}
