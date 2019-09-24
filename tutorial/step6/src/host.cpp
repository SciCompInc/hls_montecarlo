/*
 * Copyright 2019 SciComp, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file host.cpp
 *
 */

#include <cmath>
#include <iostream>
void kernel_wrapper(int niter, int seq, int& count);
int main()
{
    int niter = 1 << 20;
    int seq = 1;
    int count;
    kernel_wrapper(niter, seq, count);
    double pi_est = 4.0 * double(count) / double(niter);
    if (fabs(pi_est - 3.14) > 0.01)
    {
        std::cout << "Failed." << std::endl;
        return 1;
    }
    std::cout << "Pass." << std::endl;
    std::cout << pi_est << std::endl;
    return 0;
}
