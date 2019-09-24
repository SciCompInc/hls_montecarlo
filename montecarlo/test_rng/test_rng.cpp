#include "sci/hls/xoshiro128.h"
#include <iostream>
#include <random>

#if _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#define NRNG 4

int main()
{
#if _WIN32
    _setmode(
        _fileno(stdout),
        _O_BINARY); // needed to allow binary stdout on windows
#endif

    sci::hls::xoshiro128starstar random_engine[NRNG];
    for (int k = 0; k < NRNG; k++)
    {
        random_engine[k].seed(1234);
        for (int j = 0; j < k + 1; j++)
            random_engine[k].jump();
    }

    bool flag = true;
    while (flag)
    {
        for (int k = 0; k < NRNG && flag; k++)
        {
            uint32_t val = random_engine[k].next32();
            size_t bytes_written =
                std::fwrite((void*)&val, 1, sizeof(uint32_t), stdout);
            if (bytes_written != sizeof(uint32_t))
            {
                // Break if pipe is broken
                flag = false;
            }
        }
    }
    std::fflush(stdout);
    return 0;
}
