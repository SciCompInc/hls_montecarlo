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
    double V_ref = 9476.8262841626;
    double dev_ref = 96.2403984475;

    CHECK(V == Approx(V_ref).epsilon(0.000001));
    CHECK(dev == Approx(dev_ref).epsilon(0.000001));
    INFO("Kernel time: " << tk << " ms");
    CHECK(true);
}

TEST_CASE("test12", "[perf]")
{
    mcSVEqLinkStructNote pricer;
    pricer.load_inputs();

    int pmax = 1 << 20;
    int seq = 1;
    int nsub = 4;
    double V, dev;
    double tk;
    for (int k = 1; k <= 20; k++)
    {
        pricer.calc(pmax, k, nsub, V, dev, tk);
        std::cout << V << "\t" << dev << std::endl;
    }
}

TEST_CASE("test13", "[.][integration]")
{
    double ctable_ref[][3] = {{16384, 9460.6245566664, 23.9002956874},
                              {32768, 9456.6366885693, 16.9621141083},
                              {65536, 9466.0313195491, 11.9646416409},
                              {131072, 9456.8503024117, 8.4672185297},
                              {262144, 9449.4342585503, 5.9864806490},
                              {524288, 9447.7869781792, 4.2376930712},
                              {1048576, 9449.6688478078, 2.9962162099}};
    int ntable = sizeof(ctable_ref) / sizeof(double) / 3;

    mcSVEqLinkStructNote pricer;
    pricer.load_inputs();

    int seq = 1;
    int nsub = 4;
    double V, dev;
    double tk;
    double V_ref = 9450.6284190862; // pmax = 2^27, dev = 0.2648068835
    std::cout << "pmax"
              << "\t"
              << "V"
              << "\t"
              << "dev"
              << "\t"
              << "|V - V_ref|"
              << "\t"
              << "Conf. Interval" << std::endl;
    for (int p = 0; p < ntable; p++)
    {
        int pmax = int(ctable_ref[p][0]);
        // call pricing engine
        pricer.calc(pmax, seq, nsub, V, dev, tk);
        std::string chk = "FAIL";
        if (2.807 * dev > std::fabs(V - V_ref)) // 99.5% conf interval
        {
            chk = "OK";
        }
        std::cout << std::setw(10) << std::fixed << std::setprecision(10)
                  << pmax << "\t" << V << "\t" << dev << "\t"
                  << std::fabs(V - V_ref) << "\t" << chk << std::endl;
        // std::cout << std::setw(10) << std::fixed << std::setprecision(10) <<
        // "{ " << pmax << ",\t" << V << ",\t" << dev << "}," << std::endl;
        REQUIRE(V == Approx(ctable_ref[p][1]).epsilon(0.000001));
        REQUIRE(dev == Approx(ctable_ref[p][2]).epsilon(0.000001));
    }
}

TEST_CASE("test14", "[.][integration]")
{
    mcSVEqLinkStructNote pricer;
    pricer.load_inputs();

    int pmax = 1 << 17;
    int nsub = 4;
    double V, dev;
    double tk;

    int nseries = 20;
    double sum = 0;
    double sum2 = 0;
    for (int k = 1; k <= nseries; k++)
    {
        pricer.calc(pmax, k, nsub, V, dev, tk);
        std::cout << V << "\t" << dev << std::endl;
        sum += V;
        sum2 += V * V;
    }
    sum /= nseries;
    V = sum;
    sum2 /= nseries;
    dev = (sum2 - sum * sum);
    dev = sqrt(dev);
    std::cout << "Mean: " << V << std::endl;
    std::cout << "Std. dev: " << dev << std::endl;
    double V_ref = 9451.1942786213;
    double dev_ref = 7.2264323981;

    CHECK(V == Approx(V_ref).epsilon(0.000001));
    CHECK(dev == Approx(dev_ref).epsilon(0.000001));
}
