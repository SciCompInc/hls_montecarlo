#include <cmath>
#include <iostream>
#include <random>

#include <catch2/catch.hpp>

#include <ql/methods/montecarlo/brownianbridge.hpp>

#include <sci/brownian_bridge_setup.h>

#include <sci/hls/brownian_bridge.h>

#define NUM_SIMS 16
#define SZMAX 26

TEST_CASE("test_steps", "[bb]")
{
    const int steps = SZMAX;
    //	auto steps = GENERATE(5, 12, 52, 64, 103, 128, 511, 512, 513, 1023);

    double input[NUM_SIMS][SZMAX];
    double output[NUM_SIMS][SZMAX + 1];
    double output2[NUM_SIMS][SZMAX + 1];

    std::vector<std::vector<double>> inputQL(
        NUM_SIMS, std::vector<double>(steps));
    std::vector<double> resultQL(steps);

    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 1.0);

    for (int k = 0; k < NUM_SIMS; k++)
    {
        for (int i = 0; i < steps; ++i)
        {
            double rd = distribution(generator);
            input[k][i] = rd;
            inputQL[k][i] = rd;
        }
    }

    std::vector<double> tsample(steps + 1);
    for (int n = 0; n <= steps; ++n)
    {
        tsample[n] = n;
    }
    std::vector<int> c_data(steps + 1);
    std::vector<int> l_data(steps + 1);
    std::vector<int> r_data(steps + 1);
    std::vector<double> qasave(steps + 1);
    std::vector<double> qbsave(steps + 1);

    sci::brownian_bridge_setup(
        steps,
        tsample.data(),
        c_data.data(),
        l_data.data(),
        r_data.data(),
        qasave.data(),
        qbsave.data());

    sci::hls::brownian_bridge<double, SZMAX> bb;

    bb.init(
        steps,
        c_data.data(),
        l_data.data(),
        r_data.data(),
        qasave.data(),
        qbsave.data());

    for (int k = 0; k < NUM_SIMS; k++)
    {
        bb.transform(input, output, output2, NUM_SIMS);
    }

    // Get QuantLib function output
    QuantLib::BrownianBridge ql_bridge(steps);

    for (int k = 0; k < NUM_SIMS; k++)
    {
        ql_bridge.transform(
            inputQL[k].begin(), inputQL[k].end(), resultQL.begin());

        // Compare results
        bool hasDiff = false;
        for (int i = 0; i < steps; ++i)
        {
            if (std::fabs(output[k][i + 1] - resultQL[i]) > 1.0e-8)
            {
                std::cout << "Error " << output[k][i + 1] << " " << resultQL[i]
                          << std::endl;
                hasDiff = true;
            }
        }
        REQUIRE_FALSE(hasDiff);
    }
}
