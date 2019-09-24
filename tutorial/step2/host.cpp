// host.cpp
#include <cmath>
void kernel_wrapper(int niter, int& count);
int main()
{
    int niter = 100000;
    int count;
    kernel_wrapper(niter, count);
    double pi_est = 4.0 * double(count) / double(niter);
    if (fabs(pi_est - 3.14) > 0.01)
        return 1;
    return 0;
}
