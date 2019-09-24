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
 * @file cuda_common.h
 *
 */

#pragma once

#ifdef _MSC_VER
#define PRAGMA(x) __pragma(x)
#else
#define PRAGMA(x) _Pragma(x)
#endif

#if defined(__CUDACC__)

#define __scifinance_exec_check_disable__                                     \
    PRAGMA("hd_warning_disable")                                              \
    PRAGMA("nv_exec_check_disable")

#endif

#include <sci/cuda/core_utils.h>
#include <sci/cuda/device_array.h>
#include <sci/cuda/local_array.h>
