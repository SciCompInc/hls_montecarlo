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

void mcArithmAsian::calc(
    int pmax,
    int seq,
    double& Vx,
    double& devx,
    double& tk)
{
    // Host side code
    //
    pricer_kernel_wrapper(
        steps, dt, vol, r, q, spot, strike, call, pmax, seq, Vx, devx, tk);
}

void mcArithmAsian::load_inputs()
{
    nlohmann::json j;
    // Read input JSON file specified via environment variable SCI_DATAFILE
    std::string input_file_path("input_data.json");
    const char* input_data_var = getenv("SCI_DATAFILE");
    if (input_data_var)
    {
        input_file_path = input_data_var;
    }
    std::ifstream input(input_file_path);
    if (!input.is_open())
    {
        throw std::runtime_error(
            std::string(input_file_path) + " is not found");
    }
    input >> j;

    steps = j["steps"];
    dt = j["dt"];
    vol = j["vol"];
    r = j["r"];
    q = j["q"];
    spot = j["spot"];
    strike = j["strike"];
    call = j["call"];
}
