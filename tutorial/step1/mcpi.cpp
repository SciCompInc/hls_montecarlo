// standard Monte Carlo number pi estimation
#include <iostream>
#include <random>
int main()
{
    std::mt19937 rng(127); // seed = 127
    std::uniform_real_distribution<> dist;
    int niter = 100000;
    int count = 0;
    for (int i = 0; i < niter; ++i)
    {
        // get random points
        double x = dist(rng);
        double y = dist(rng);
        // check to see if point is in unit circle
        if (x * x + y * y < 1)
            ++count;
    }
    double pi_est = 4.0 * double(count) / double(niter);
    std::cout << "pi est: " << pi_est << std::endl;
    return 0;
}
