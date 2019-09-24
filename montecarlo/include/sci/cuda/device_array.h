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
 * @file device_array.h
 *
 */

#pragma once

#include <vector>

#include <sci/cuda/core_utils.h>
#include <sci/cuda/local_array.h>

namespace sci
{

template<typename T>
class DeviceArray;

template<typename T>
void swap(DeviceArray<T>& arr1, DeviceArray<T>& arr2);

template<typename T>
class DeviceArray
{
public:
    explicit DeviceArray(size_t size = 0)
        : size_(size)
    {
        alloc(data_, size_);
    }
    explicit DeviceArray(const std::vector<T>& vec)
    {
        size_ = vec.size();
        alloc(data_, size_);
        sciCUDAErrorCheck(cudaMemcpy(
            data_, vec.data(), size_ * sizeof(T), cudaMemcpyHostToDevice));
    }
    explicit DeviceArray(const T* data, size_t size)
    {
        size_ = size;
        alloc(data_, size_);
        sciCUDAErrorCheck(cudaMemcpy(
            data_, data, size_ * sizeof(T), cudaMemcpyHostToDevice));
    }

    ~DeviceArray() { cudaFree(data_); }
    T* array() const { return data_; }
    operator T*() const { return data_; }

    void copy(std::vector<T>& dst)
    {
        sciCUDAErrorCheck(cudaMemcpy(
            dst.data(), data_, size_ * sizeof(T), cudaMemcpyDeviceToHost));
    }

    void sum(T& res)
    {
        res = T(0);
        T* hdata = new T[size_];
        sciCUDAErrorCheck(cudaMemcpy(
            hdata, data_, size_ * sizeof(T), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < size_; ++i)
        {
            res += hdata[i];
        }
        delete[] hdata;
    }

    friend void swap<T>(DeviceArray<T>& arr1, DeviceArray<T>& arr2);

private:
    size_t size_;
    T* data_;
    void alloc(T* data, size_t size)
    {
        sciCUDAErrorCheck(cudaMalloc((void**)&data_, size_ * sizeof(T)));
        sciCUDAErrorCheck(cudaMemset(data_, 0, size_ * sizeof(T)));
    }
};

template<typename T>
void swap(DeviceArray<T>& arr1, DeviceArray<T>& arr2)
{
    T* tmp;
    tmp = arr1.data_;
    arr1.data_ = arr2.data_;
    arr2.data_ = tmp;
}

} // namespace sci