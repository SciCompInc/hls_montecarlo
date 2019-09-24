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
 * @file pricer_host.cpp
 *
 * See pricer_host.h for more info
 */

#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "pricer_kernel_wrapper.h"

#include "pricer_host.h"

void mcPi::calc(int pmax, int seq, double& Vx, double& devx, double& tk)
{
    // Host side code
    //
    pricer_kernel_wrapper(pmax, seq, Vx, devx, tk);
}

void mcPi::load_inputs() {}
