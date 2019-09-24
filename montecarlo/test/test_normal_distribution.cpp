#include <catch2/catch.hpp>
#include <iostream>

#include <sci/distributions.h>

TEST_CASE("extreme1", "[normal]")
{
    auto u = GENERATE(4294967168U, 0x000001FFU);
    float xu;
    sci::uniform_distribution_transform(u, xu);
    float xn;
    xn = sci::normal_distribution_icdf(xu);
    REQUIRE(xn == xn);
}
