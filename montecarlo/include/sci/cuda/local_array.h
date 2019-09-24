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
 * @file local_array.h
 *
 */

#pragma once

#include <sci/cuda/core_utils.h>

namespace sci
{

template<typename T>
class Array1D
{
public:
    __device__ Array1D()
        : data_(0)
        , size0_(0)
        , tid_(0)
        , nT_(1)
    {
    }

    __device__ Array1D(
        T* data,
        unsigned int size0,
        unsigned int tid = 0,
        unsigned int nT = 1)
        : data_(data)
        , size0_(size0)
        , tid_(tid)
        , nT_(nT)
    {
    }

    __device__ Array1D<T>& operator=(const Array1D<T>& src)
    {
        for (int i = 0; i < size0_; i++)
            data_[i * nT_ + tid_] = src(i);
        return *this;
    }

    __device__ T operator()(unsigned int index1) const
    {
        return data_[index1 * nT_ + tid_];
    }

    __device__ T& operator()(unsigned int index1)
    {
        return data_[index1 * nT_ + tid_];
    }
    __device__ unsigned int size0() const { return size0_; }

    __device__ unsigned int getTID() const { return tid_; }

    __device__ unsigned int getNT() const { return nT_; }

    __device__ T* array() const { return data_; }

    __device__ operator T*() const { return data_; }

private:
    T* data_;
    unsigned int size0_;
    unsigned int tid_;
    unsigned int nT_;
};

template<typename T>
class Array2D
{
public:
    __device__ Array2D()
        : data_(0)
        , size0_(0)
        , size1_(0)
        , tid_(0)
        , nT_(1)
    {
    }

    __device__ Array2D(
        T* data,
        unsigned int size0,
        unsigned int size1,
        unsigned int tid = 0,
        unsigned int nT = 1)
        : data_(data)
        , size0_(size0)
        , size1_(size1)
        , tid_(tid)
        , nT_(nT)
    {
    }

    __device__ Array2D<T>& operator=(const Array2D<T> src)
    {
        data_ = src.array();
        size0_ = src.size0();
        size1_ = src.size1();
        tid_ = src.getTID();
        nT_ = src.getNT();
        return *this;
    }

    __device__ T operator()(unsigned int index1, unsigned int index2) const
    {
        return data_[(index1 * size1_ + index2) * nT_ + tid_];
    }

    __device__ T& operator()(unsigned int index1, unsigned int index2)
    {
        return data_[(index1 * size1_ + index2) * nT_ + tid_];
    }

    __device__ Array1D<T> operator()(unsigned int index1) const
    {
        Array1D<T> slice(data_ + (index1 * size1_) * nT_, size1_, tid_, nT_);
        return slice;
    }

    __device__ unsigned int size0() const { return size0_; }
    __device__ unsigned int size1() const { return size1_; }

    __device__ unsigned int getTID() const { return tid_; }

    __device__ unsigned int getNT() const { return nT_; }

    __device__ T* array() const { return data_; }

private:
    T* data_;
    unsigned int size0_;
    unsigned int size1_;
    unsigned int tid_;
    unsigned int nT_;
};

template<typename T>
class Array3D
{
public:
    __device__ Array3D(
        T* data,
        unsigned int size0,
        unsigned int size1,
        unsigned int size2,
        unsigned int tid = 0,
        unsigned int nT = 1)
        : data_(data)
        , size0_(size0)
        , size1_(size1)
        , size2_(size2)
        , tid_(tid)
        , nT_(nT)
    {
    }

    __device__ T operator()(
        unsigned int index1,
        unsigned int index2,
        unsigned int index3) const
    {
        return data_
            [((index1 * size1_ + index2) * size2_ + index3) * nT_ + tid_];
    }

    __device__ T&
    operator()(unsigned int index1, unsigned int index2, unsigned int index3)
    {
        return data_
            [((index1 * size1_ + index2) * size2_ + index3) * nT_ + tid_];
    }

    __device__ Array1D<T>
    operator()(unsigned int index1, unsigned int index2) const
    {
        return Array1D<T>(
            data_ + (index1 * size1_ + index2) * size2_, size2_, tid_, nT_);
    }

    __device__ Array2D<T> operator()(unsigned int index1) const
    {
        return Array2D<T>(
            data_ + index1 * size1_ * size2_, size1_, size2_, tid_, nT_);
    }

private:
    T* data_;
    unsigned int size0_;
    unsigned int size1_;
    unsigned int size2_;
    unsigned int tid_;
    unsigned int nT_;
};

} // namespace sci