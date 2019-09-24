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
 * @file pricer_host.h
 *
 * Arithmetic Asian option pricer object
 * Simple structure for input parameters
 * Loads input data from JSON file and runs the calculation
 * JSON data file name is specified via environment variable: SCI_DATAFILE
 */

#pragma once

struct mcArithmAsian {
    /**
     * \brief Loads input parameters from JSON data file
     */
    void load_inputs();

    /**
     * \brief Run the calculation
     * \param pmax number of Monte Carlo paths
     * \param seq random sequence id (seq >= 1)
     * \param Vx price (present value)
     * \param devx standard error estimation (pseudo random only)
     * \param tk kernel execution time
     */
    void calc(int pmax, int seq, double& Vx, double& devx, double& tk);

    // pricer input data
    int steps; //< number of time steps / fixings
    double dt; //< time step
    double vol; //< volatility
    double r; //< risk-free rate
    double q; //< dividend rate
    double spot; //< spot price
    double strike; //< strike price
    int call; //< option type (call = 1, put = 0)
};
