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
    double V_ref = 9512.5888671875;
    double dev_ref = 94.2233047485;

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
    for (int k = 1; k <= 5; k++)
    {
        pricer.calc(pmax, k, nsub, V, dev, tk);
        INFO("Kernel time: " << tk << " ms");
        CHECK(false);
    }
}

TEST_CASE("test13", "[.][integration]")
{
    /*  4 threads, double precision
     *  { 16384,        9464.8903479161,        23.8606143402},
    { 32768,        9465.2426725480,        16.9127541155},
    { 65536,        9467.7241541556,        11.9390285004},
    { 131072,       9466.9953639783,        8.4446769147},
    { 262144,       9461.9387707258,        5.9802626584},
    { 524288,       9456.0685602271,        4.2342651699},
    { 1048576,      9449.7267374527,        2.9973167224},
    { 2097152,      9451.6958879633,        2.1182209723},
    { 4194304,      9451.6368485576,        1.4976633600},
    { 8388608,      9451.6162139899,        1.0589344850},
    { 16777216,     9451.3735276875,        0.7489230189},
    { 33554432,     9450.9947562029,        0.5296203278},
    { 67108864,     9450.9330571077,        0.3744791866},
    { 134217728,    9450.6284190862,        0.2648068835}
     */
    double ctable_ref[][3] = {{16384, 9464.8242187500, 23.8637790680},
                              {32768, 9465.3212890625, 16.9114475250},
                              {65536, 9467.9912109375, 11.9367437363},
                              {131072, 9467.0068359375, 8.4381904602},
                              {262144, 9463.9062500000, 5.9632534981},
                              {524288, 9461.9433593750, 4.2065887451},
                              {1048576, 9455.3486328125, 2.9671382904}};
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
    double V_ref = 9450.3503906250;
    double dev_ref = 9.5552006332;

    CHECK(V == Approx(V_ref).epsilon(0.000001));
    CHECK(dev == Approx(dev_ref).epsilon(0.000001));
}
