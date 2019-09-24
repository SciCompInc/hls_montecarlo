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
 * Monte Carlo Pi calculator object
 */

#pragma once

struct mcPi {
    void load_inputs();

    /**
     * \brief Run the calculation
     * \param pmax number of Monte Carlo paths
     * \param seq random sequence id (seq >= 1)
     * \param Vx value estimate
     * \param devx standard error estimation (pseudo random only)
     * \param tk kernel execution time
     */
    void calc(int pmax, int seq, double& Vx, double& devx, double& tk);
};
