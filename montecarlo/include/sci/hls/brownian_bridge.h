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
 * @file brownian_bridge.h
 *
 * Brownian Bridge transform class
 */
#pragma once

namespace sci
{
namespace hls
{

    /**
     * \brief Brownian Bridge transform class
     * \tparam DT floating point type
     * \tparam SZ maximum sequence length
     */
    template<typename DT, int SZ>
    class brownian_bridge
    {
    public:
        /**
         * \brief Brownian Bridge transform class initializer
         * \param size sequence length
         * \param c_data pre-calculated transformation data
         * \param l_data pre-calculated transformation data
         * \param r_data pre-calculated transformation data
         * \param qasave pre-calculated transformation data
         * \param qbsave pre-calculated transformation data
         */
        void init(
            int size,
            int c_data[SZ + 1],
            int l_data[SZ + 1],
            int r_data[SZ + 1],
            DT qasave[SZ + 1],
            DT qbsave[SZ + 1])
        {
            size_ = size;
        loop_bb_init:
            for (int n = 0; n <= SZ; ++n)
            {
                c_data_[n] = c_data[n];
                l_data_[n] = l_data[n];
                r_data_[n] = r_data[n];
                qasave_[n] = qasave[n];
                qbsave_[n] = qbsave[n];
            }
        }

        /**
         * \brief Brownian Bridge transform for the vector of sequences
         * \param QRSeq input vector of sequences (row = sequence)
         * \param BbQRSeq output vector of transforms
         * \param BbQRSeq2 output vector auxiliary buffer
         * \param sims size of the vector of sequences
         *
         * The sequences are grouped to a vector for the batch processing:
         * row-vise:
         * z[path=1][t=1] z[0][t=2] ... z[0][t=M]
         * ...
         * z[path=N][t=1] z[path=N][t=2] ... z[path=N][t=M]
         * where M is the sequence length, N is the path batch (vector) size
         *
         * Brownian Bridge transform output sequences are shifted by one
         * element to accomodate t=0 point. Each sequence element is
         * multiplied by the square root of the time step:
         * output[i] = bb_transform[i] * sqrt(t[i] - t[i-1]), where i >= 1
         */
        void transform(
            DT QRSeq[][SZ],
            DT BbQRSeq[][SZ + 1],
            DT BbQRSeq2[][SZ + 1],
            int sims)
        {
        loop_trans_01:
            for (int iD = 0; iD < sims; iD++)
            {
// clang-format off
#pragma HLS pipeline
                // clang-format on
                DT bN = qasave_[0] * QRSeq[iD][0];
                BbQRSeq[iD][size_] = bN;
                BbQRSeq[iD][0] = DT(0.0);
                BbQRSeq2[iD][size_] = bN;
                BbQRSeq2[iD][0] = DT(0.0);
            }

        loop_trans_021:
            for (int i = 1; i < size_; i++)
            {
// clang-format off
#pragma HLS loop_tripcount min=SZ max=SZ
                // clang-format on
                int lpos = l_data_[i];
                int rpos = r_data_[i];
                int cpos = c_data_[i];
                DT qa = qasave_[i];
                DT qb = qbsave_[i];
            loop_trans_022:
                for (int iD = 0; iD < sims; iD++)
                {
// clang-format off
#pragma HLS pipeline
                    // clang-format on
                    DT LeftRV = BbQRSeq[iD][lpos];
                    DT RightRV = BbQRSeq2[iD][rpos];
                    DT CenterRV = QRSeq[iD][i];
                    DT bC = qa * (LeftRV - RightRV) + RightRV + qb * CenterRV;
                    BbQRSeq[iD][cpos] = bC;
                    BbQRSeq2[iD][cpos] = bC;
                }
            }

        loop_trans_032:
            for (int i = 1; i <= size_; i++)
            {
// clang-format off
#pragma HLS loop_tripcount min=SZ max=SZ
                // clang-format on
            loop_trans_031:
                for (int iD = 0; iD < sims; iD++)
                {
// clang-format off
#pragma HLS pipeline
                    // clang-format on
                    BbQRSeq[iD][i] = BbQRSeq2[iD][i] - BbQRSeq2[iD][i - 1];
                }
            }
        }

    private:
        int size_;
        int c_data_[SZ + 1];
        int l_data_[SZ + 1];
        int r_data_[SZ + 1];
        DT qasave_[SZ + 1];
        DT qbsave_[SZ + 1];
    };

} // namespace hls

} // namespace sci
