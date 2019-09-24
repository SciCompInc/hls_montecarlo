#include <iostream>

#include <catch2/catch.hpp>

#include <ql/math/randomnumbers/sobolrsg.hpp>

#include <sci/sobol_joe_kuo_dirnum.h>

#include <sci/distributions.h>
#include <sci/hls/sobol_joe_kuo.h>

TEST_CASE("test_skip_ql", "[sobol]")
{
    auto steps = GENERATE(5, 111);

    int pmax = 1011;
    QuantLib::SobolRsg qrng1(steps, 0, QuantLib::SobolRsg::JoeKuoD6);
    auto sample1 = qrng1.nextSequence();
    for (int p = 1; p < pmax; p++)
    {
        sample1 = qrng1.nextSequence();
    }
    QuantLib::SobolRsg qrng2(steps, 0, QuantLib::SobolRsg::JoeKuoD6);
    qrng2.skipTo(pmax - 1);
    auto sample2 = qrng2.nextSequence();

    for (int i = 0; i < steps; i++)
    {
        REQUIRE(sample1.value[i] == sample2.value[i]);
    }
}

TEST_CASE("test_skip", "[sobol]")
{
#ifdef _MSC_VER
    auto steps = GENERATE(12, 52, 252);
    auto pmax = GENERATE(19, 64, 1011);
    auto sskip = GENERATE(0, 1, 123);
#else
    int steps = 12;
    int pmax = 64;
    int sskip = 123;
#endif

    if (steps > sci::kSobolMaxDim)
    {
        REQUIRE(0);
    }

    QuantLib::SobolRsg qrng1(steps, 0, QuantLib::SobolRsg::JoeKuoD6);
    qrng1.skipTo(sskip);
    auto sample1 = qrng1.nextSequence();
    for (int p = 1; p < pmax; p++)
    {
        sample1 = qrng1.nextSequence();
    }

    const int nbit = sci::new_sobol_joe_kuo_6_21201_nbit;
    const int ndim = steps;
    std::vector<unsigned int> dirnum(ndim * nbit);
    std::vector<unsigned int> shift(ndim);
    sci::new_sobol_joe_kuo_6_21201_dirnum(ndim, dirnum.data(), shift.data());

    sci::hls::sobol_joe_kuo<252> qrng;

    qrng.init(dirnum.data(), steps);

    qrng.skip(sskip);
    std::vector<unsigned int> x(ndim);
    for (int p = 0; p < pmax; p++)
    {
        qrng.next(x.data());
    }

    for (int i = 0; i < steps; i++)
    {
        double zu;
        sci::uniform_distribution_transform(x[i], zu);
        REQUIRE(zu == Approx(sample1.value[i]).epsilon(0.000001));
    }
}

TEST_CASE("long jump", "[sobol]")
{
    const int ndim = 16;
    const int pmax = 1 << 20;

    const int nbit = sci::new_sobol_joe_kuo_6_21201_nbit;
    std::vector<unsigned int> dirnum(ndim * nbit);
    std::vector<unsigned int> shift(ndim);
    sci::new_sobol_joe_kuo_6_21201_dirnum(ndim, dirnum.data(), shift.data());

    sci::hls::sobol_joe_kuo<ndim> qrng;

    qrng.init(dirnum.data(), ndim);

    unsigned int zq[ndim];
    float zqu[ndim];
    float zqn[ndim];
    int series = 20;
    qrng.skip(pmax * series);
    for (int i = 0; i < pmax; i++)
    {
        qrng.next(zq);
        for (int k = 0; k < ndim; k++)
        {
            sci::uniform_distribution_transform(zq[k], zqu[k]);
            // check uniforms range
            if (zqu[k] < 0.0f || zqu[k] > 1.0f)
            {
                REQUIRE(false);
            }
            // check ICDF normal
            // sci::inverse_cumulative_transform(zq[k], zqn[k]);
            sci::uniform_distribution_transform(zq[k], zqu[k]);
            zqn[k] = sci::normal_distribution_icdf(zqu[k]);
            if (zqn[k] != zqn[k])
            {
                REQUIRE(false);
            }
        }
    }
}
