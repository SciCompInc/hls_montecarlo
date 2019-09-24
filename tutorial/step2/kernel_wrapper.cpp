// kernel_wrapper.cpp
#include <random>
void kernel_wrapper(int niter, int& count)
{
    std::mt19937 rng(127); // seed = 127
    std::uniform_real_distribution<> dist;
    count = 0;
    for (int i = 0; i < niter; ++i)
    {
        // get random points
        double x = dist(rng);
        double y = dist(rng);
        // check to see if point is in unit circle
        if (x * x + y * y < 1)
            ++count;
    }
}
