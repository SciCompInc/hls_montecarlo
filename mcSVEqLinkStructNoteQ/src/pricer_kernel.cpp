/*
 * Stochastic Volatility Equity Linked Structured Note model Monte Carlo pricer
 *
  The following code prices an equity-linked structured note in which
  the Note is linked to a basket of nD indices.

  If on observation dates the performance of all indices is
  above the knockout barrier for that date, the Note redeems early
  at par plus a bonus coupon. If early redemption does not occur,
  then at maturity either:
      1. the Note redeems at par plus a maturity coupon, or
      2. if the performance of at least KINum of the indices
      have even been below the knock in barrier during the tenor,
      on a continuously observed basis, then the Note redeems
      at the performance percentage of the worst index.

   The indices follow correlated Heston stochastic volatility processes
   and we allow for a term structure of rates.
   The vector KI is either zero or one for each index depending on
   whether it has dropped below the knock-n barrier KIBarrier.
   The barrier monitoring is made continuous using the continuity correction.
   However, since this Brownian bridge-based correction is strictly valid only
   for a single asset or for uncorrelated assets, we allow for nsub barrier
 monitoring dates between each pair of contracted observation dates.
   The Brownian bridge approximation should converge very rapidly as nsub is
   increased unless the indices are nearly perfectly correlated.
   On contracted observation dates we monitor the indices for knockout
 (discretely monitored), and if knockout has occurred make the early payment
 and set the early redemption flag Redeemed.
   If redemption has not occurred at maturity, the payout is either
   the minimum performance if KI>=KINum, or else par plus the maturity coupon.
 */

#include <hls_math.h>

#if defined(__GNUC__) && defined(_WIN32) && defined(__SYNTHESIS__)
// Vivado HLS 2018.3 GUI console output hack
#include </sci/distributions.h>
#include </sci/hls/brownian_bridge.h>
#include </sci/hls/sobol_joe_kuo.h>
#else
#include <sci/distributions.h>
#include <sci/hls/brownian_bridge.h>
#include <sci/hls/sobol_joe_kuo.h>
#endif

#include "kernel_global.h"

#define NUM_BLOCKS 16

#define NUM_SIMS 32

#define ND_MAX 6
#define NOBS_MAX 6
#define NSUB_MAX 4
#define TS_MAX (NOBS_MAX * NSUB_MAX)

// loop trip count estimations
#if defined(__SYNTHESIS__)
namespace
{
const int tc_pmax = 1024 * 1024;
const int tc_ngroups = tc_pmax / (NUM_BLOCKS * NUM_SIMS);
const int tc_nd = ND_MAX;
const int tc_mimax = TS_MAX;
const int tc_nobs = NOBS_MAX;
} // namespace
#endif

struct pricer_kernel_data {
    real_t AccruedBonus0;
    real_t BonusCoup;
    real_t CallBarrier[NOBS_MAX + 1];
    real_t kappa[ND_MAX + 1];
    real_t KI0[ND_MAX + 1];
    real_t KIBarrier;
    real_t KINum;
    real_t MatCoup;
    int nD;
    real_t Notional;
    real_t q[ND_MAX + 1];
    real_t sigma[ND_MAX + 1];
    real_t Spot[ND_MAX + 1];
    real_t SRef[ND_MAX + 1];
    real_t theta[ND_MAX + 1];
    real_t vSpot[ND_MAX + 1];
    int miMax;
    real_t DEtimeTable[TS_MAX + 1];
    int DEtable[TS_MAX + 1][3];
    real_t DeltaT[NOBS_MAX];
    real_t Mcor[2 * ND_MAX + 1][2 * ND_MAX + 1];
    real_t r[TS_MAX + 1];
};

void pricer_kernel_block(
    sci::hls::sobol_joe_kuo<TS_MAX * 3 * ND_MAX>& qrng,
    sci::hls::brownian_bridge<real_t, TS_MAX>& bb,
    pricer_kernel_data& data,
    int pmax,
    int seq,
    real_t& V,
    int block_id)
{
    const real_t tfloor = real_t(0.00001);
    const real_t vfloor = real_t(0.00001);

    qrng.skip((block_id + seq * NUM_BLOCKS) * pmax);

    real_t payoffSum = 0;

    real_t S[NUM_SIMS][ND_MAX + 1];
    real_t SOldCC[NUM_SIMS][ND_MAX + 1];
    real_t v[NUM_SIMS][ND_MAX + 1];
    real_t KI[NUM_SIMS][ND_MAX + 1];
    real_t vProt[NUM_SIMS][ND_MAX + 1];

    unsigned int ZQ[NUM_SIMS][3 * ND_MAX * TS_MAX];
    real_t ZQN[2 * ND_MAX + 1][NUM_SIMS][TS_MAX];
    real_t ZQB[2 * ND_MAX + 1][NUM_SIMS][TS_MAX + 1];
    real_t ZQB2[NUM_SIMS][TS_MAX + 1];

    real_t KIsum[NUM_SIMS];
    real_t Smin[NUM_SIMS];

    int Redeemed[NUM_SIMS];
    real_t EarlyPayoff[NUM_SIMS];

loop_groups:
    for (int j = 0; j < pmax / NUM_SIMS; j++)
    {
// clang-format off
#pragma HLS loop_tripcount min=tc_ngroups max=tc_ngroups
        // clang-format on
        real_t t = 0.;
        real_t AccruedBonus = data.AccruedBonus0;
        real_t discount = 0.;
        int iObs = 1;
        real_t tOldCC = t;

    loop_id_0:
        for (int iD = 1; iD <= data.nD; iD++)
        {
// clang-format off
#pragma HLS loop_tripcount min=tc_nd max=tc_nd
            // clang-format on
        loop_sim_0:
            for (int k = 0; k < NUM_SIMS; k++)
            {
                S[k][iD] = data.Spot[iD];
                v[k][iD] = data.vSpot[iD];
                KI[k][iD] = data.KI0[iD];
                SOldCC[k][iD] = data.Spot[iD];
            }
        }
    loop_sim_01:
        for (int k = 0; k < NUM_SIMS; k++)
        {
            Redeemed[k] = 0;
            KIsum[k] = 0;
        }

    // ---------------------------------------------------------------------------
    // Sobol QRNG
    // Order: steps first (outer: paths, inner: steps)
    loop_sims_01:
        for (int k = 0; k < NUM_SIMS; k++)
        {
            qrng.next(ZQ[k]);
        }
    // ---------------------------------------------------------------------------

    // ---------------------------------------------------------------------------
    // Normal Transform via ICDF
    // Order: any
    loop_time_021:
        for (int n = 0; n < data.miMax * 2 * data.nD; n++)
        {
// clang-format off
#pragma HLS loop_tripcount min=tc_mimax*2*tc_nd max=tc_mimax*2*tc_nd
            // clang-format on
        loops_sims_021:
            for (int k = 0; k < NUM_SIMS; k++)
            {
// clang-format off
#pragma HLS pipeline
// clang-format on
                real_t xu;
                sci::uniform_distribution_transform(ZQ[k][n], xu);
                ZQN[n / data.miMax + 1][k][n % data.miMax] =
                    sci::normal_distribution_icdf(xu);
            } // k
        }
        // ---------------------------------------------------------------------------

    loop_id_023:
        for (int iD = 1; iD <= 2 * data.nD; iD++)
        {
// clang-format off
#pragma HLS loop_tripcount min=2*tc_nd max=2*tc_nd
            // clang-format on
            bb.transform(ZQN[iD], ZQB[iD], ZQB2, NUM_SIMS);
        }
        // ---------------------------------------------------------------------------

    loop_corr_1:
        for (int n = 0; n <= data.miMax - 1; n++)
        {
// clang-format off
#pragma HLS loop_tripcount min=tc_mimax max=tc_mimax
            // clang-format on
        loop_corr_21:
            for (int iD = 1; iD <= 2 * data.nD; iD++)
            {
// clang-format off
#pragma HLS loop_tripcount min=2*tc_nd max=2*tc_nd
                // clang-format on
            loop_corr_22:
                for (int k = 0; k < NUM_SIMS; k++)
                {
// clang-format off
#pragma HLS PIPELINE
                    // clang-format on
                    ZQN[iD][k][n] = 0;
                } // k
            }

            // Correlate normal variates Zc = L * Zn
            // L is a lower triangular matrix from Cholesky decomposition
        loop_corr_31:
            for (int i3 = 2 * data.nD; i3 >= 1; i3--)
            {
// clang-format off
#pragma HLS loop_tripcount min=2*tc_nd max=2*tc_nd
                // clang-format on
            loop_corr_32:
                for (int j2 = 1; j2 <= i3; j2++)
                {
// clang-format off
#pragma HLS loop_tripcount min=tc_nd max=tc_nd
                    // clang-format on
                loop_corr_33:
                    for (int k = 0; k < NUM_SIMS; k++)
                    {
// clang-format off
#pragma HLS pipeline
#pragma HLS dependence variable=ZQN inter false
                        // clang-format on
                        ZQN[i3][k][n] += ZQB[j2][k][n + 1] * data.Mcor[i3][j2];
                    } // k
                }
            }
        }

    loop_time:
        for (int n = 0; n <= data.miMax - 1; n++)
        {
            // clang-format off
        	#pragma HLS loop_tripcount min=tc_mimax max=tc_mimax
            // clang-format on
            real_t dt = data.DEtimeTable[n + 1] - t;
            discount += dt * (data.r[n] + data.r[n + 1]) * 0.5f;
            real_t riskless = data.r[n];

            // Truncate variance
        loop_id_31:
            for (int iD = 1; iD <= data.nD; iD++)
            {
// clang-format off
#pragma HLS loop_tripcount min=tc_nd max=tc_nd
                // clang-format on
            loop_sims_31:
                for (int k = 0; k < NUM_SIMS; k++)
                {
                    // clang-format off
					#pragma HLS PIPELINE
                    // clang-format on
                    vProt[k][iD] = (v[k][iD] > vfloor) ? v[k][iD] : vfloor;
                } // k
            }
            // Simulate
        loop_id_32:
            for (int iD = 1; iD <= data.nD; iD++)
            {
// clang-format off
#pragma HLS loop_tripcount min=tc_nd max=tc_nd
                // clang-format on
            loop_sims_32:
                for (int k = 0; k < NUM_SIMS; k++)
                {
                    // clang-format off
#pragma HLS PIPELINE
#pragma HLS dependence variable=S inter false
#pragma HLS dependence variable=v inter false
                    // clang-format on
                    S[k][iD] *= std::exp(
                        std::sqrt(vProt[k][iD]) * ZQN[iD][k][n] +
                        dt * (riskless - vProt[k][iD] * 0.5f - data.q[iD]));
                    v[k][iD] +=
                        dt * data.kappa[iD] * (data.theta[iD] - vProt[k][iD]) +
                        std::sqrt(vProt[k][iD]) * data.sigma[iD] *
                            ZQN[iD + data.nD][k][n];
                } // k
            }
            // General discrete event updates
            if (data.DEtable[n + 1][1] == 1)
            {
                real_t dtCC = dt + t - tOldCC;
            loop_id_4:
                for (int iD = 1; iD <= data.nD; iD++)
                {
                    // clang-format off
                	#pragma HLS loop_tripcount min=tc_nd max=tc_nd
                    // clang-format on
                loop_sim_4:
                    for (int k = 0; k < NUM_SIMS; k++)
                    {
                        // clang-format off
#pragma HLS pipeline
#pragma HLS dependence variable=SOldCC inter false
                        // clang-format on
                        if (S[k][iD] / data.SRef[iD] <= data.KIBarrier)
                        {
                            KI[k][iD] = 1.;
                        }
                        else
                        {
                            if (data.KIBarrier > 0.)
                            {
                                // Continuity correction
                                real_t denom = dtCC * vProt[k][iD];
                                if (denom < tfloor)
                                    denom = tfloor;
                                real_t test1 = std::exp(
                                    (real_t(2) *
                                     std::log(S[k][iD] /
                                         (data.SRef[iD] * data.KIBarrier)) *
                                     std::log((data.KIBarrier * data.SRef[iD]) /
                                         SOldCC[k][iD])) /
                                    denom);
                                real_t test;
                                sci::uniform_distribution_transform(
                                    ZQ[k]
                                      [data.miMax * (iD - 1) + n +
                                       data.miMax * 2 * data.nD],
                                    test);
                                if ((test < test1 && test1 <= real_t(1)))
                                {
                                    KI[k][iD] = 1.;
                                }
                            }
                        }
                        SOldCC[k][iD] = S[k][iD];
                    } // k
                }
                tOldCC = dt + t;
            }
            if (data.DEtable[n + 1][2] == 1)
            {
                AccruedBonus += data.BonusCoup * data.DeltaT[iObs];
            loop_sim_49:
                for (int k = 0; k < NUM_SIMS; k++)
                {
                    Smin[k] = 3.402823e+38;
                }
            loop_id_50:
                for (int iD = 1; iD <= data.nD; iD++)
                {
                    // clang-format off
                	#pragma HLS loop_tripcount min=tc_nd max=tc_nd
                    // clang-format on
                loop_sim_50:
                    for (int k = 0; k < NUM_SIMS; k++)
                    {
                        // clang-format off
#pragma HLS PIPELINE
#pragma HLS dependence variable=Smin inter false
                        // clang-format on
                        real_t norm = S[k][iD] / data.SRef[iD];
                        if (norm < Smin[k])
                            Smin[k] = norm;
                    }
                }
            loop_sim_5:
                for (int k = 0; k < NUM_SIMS; k++)
                {
                    // clang-format off
#pragma HLS PIPELINE
                    // clang-format on
                    if (Redeemed[k] == 0)
                    {
                        if (Smin[k] > data.CallBarrier[iObs])
                        {
                            Redeemed[k] = 1;
                            EarlyPayoff[k] =
                                (AccruedBonus + 1) * std::exp(-discount);
                        }
                    }
                } // k
                iObs++;
            }
            /* Update time variables  */
            /* Update time */
            t = dt + t;
        }
    loop_id_51:
        for (int iD = 1; iD <= data.nD; iD++)
        {
            // clang-format off
        	#pragma HLS loop_tripcount min=tc_nd max=tc_nd
            // clang-format on
        loop_sim_51:
            for (int k = 0; k < NUM_SIMS; k++)
            {
                // Dependence pragma is required to prevent false dependence
                // clang-format off
#pragma HLS PIPELINE
#pragma HLS dependence variable=KIsum inter false
                // clang-format on
                KIsum[k] += KI[k][iD];
            }
        }

    loop_sim_520:
        for (int k = 0; k < NUM_SIMS; k++)
        {
            Smin[k] = 3.402823e+38;
        }

    loop_id_52:
        for (int iD = 1; iD <= data.nD; iD++)
        {
            // clang-format off
        	#pragma HLS loop_tripcount min=tc_nd max=tc_nd
            // clang-format on
        loop_sim_52:
            for (int k = 0; k < NUM_SIMS; k++)
            {
                // clang-format off
#pragma HLS PIPELINE
#pragma HLS dependence variable=Smin inter false
                // clang-format on
                real_t norm = S[k][iD] / data.SRef[iD];
                if (norm < Smin[k])
                    Smin[k] = norm;
            }
        }

        real_t payoffSum_c = 0;
    loop_sim_6:
        for (int k = 0; k < NUM_SIMS; k++)
        {
            real_t payoff;
            if (Redeemed[k])
            {
                payoff = EarlyPayoff[k] * data.Notional;
            }
            else
            {
                if (KIsum[k] >= data.KINum)
                {
                    payoff = Smin[k] * data.Notional;
                }
                else
                {
                    payoff = (data.MatCoup + 1) * data.Notional;
                }
            }
            payoffSum_c += payoff;
        } // k
        payoffSum += payoffSum_c;
    }
    V = payoffSum;
}

extern "C" void pricer_kernel(
    real_t AccruedBonus0,
    real_t BonusCoup,
    real_t* CallBarrier,
    real_t* kappa,
    real_t* KI0,
    real_t KIBarrier,
    real_t KINum,
    real_t MatCoup,
    int nD,
    real_t Notional,
    int pmax,
    real_t* q,
    int seq,
    real_t* sigma,
    real_t* Spot,
    real_t* SRef,
    real_t* theta,
    real_t* vSpot,
    int nObs,
    int miMax,
    real_t* DEtimeTable,
    int* DEtable,
    real_t* DeltaT,
    real_t* Mcor,
    real_t* r,
    unsigned int* dirnum,
    int* c_data,
    int* l_data,
    int* r_data,
    real_t* qasave,
    real_t* qbsave,
    real_t* payoff_sum)
{
    // clang-format off
#pragma HLS INTERFACE m_axi port=payoff_sum offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=CallBarrier depth=1024  offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=kappa depth=1024 offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=KI0 depth=1024 offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=q depth=1024 offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=sigma depth=1024 offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=Spot depth=1024 offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=SRef depth=1024 offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=theta depth=1024 offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=vSpot depth=1024 offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=DEtimeTable depth=1024 offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=DEtable depth=1024 offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=DeltaT depth=1024 offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=Mcor depth=1024 offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=r depth=1024 offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=dirnum depth=1024*32 offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=c_data depth=128 offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=l_data depth=128 offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=r_data depth=128 offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=qasave depth=128 offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=qbsave depth=128 offset=slave bundle=gmem

#pragma HLS INTERFACE s_axilite port=AccruedBonus0 bundle=control
#pragma HLS INTERFACE s_axilite port=BonusCoup bundle=control
#pragma HLS INTERFACE s_axilite port=CallBarrier bundle=control
#pragma HLS INTERFACE s_axilite port=kappa bundle=control
#pragma HLS INTERFACE s_axilite port=KI0 bundle=control
#pragma HLS INTERFACE s_axilite port=KIBarrier bundle=control
#pragma HLS INTERFACE s_axilite port=KINum bundle=control
#pragma HLS INTERFACE s_axilite port=MatCoup bundle=control
#pragma HLS INTERFACE s_axilite port=nD bundle=control
#pragma HLS INTERFACE s_axilite port=Notional bundle=control
#pragma HLS INTERFACE s_axilite port=pmax bundle=control
#pragma HLS INTERFACE s_axilite port=q bundle=control
#pragma HLS INTERFACE s_axilite port=seq bundle=control
#pragma HLS INTERFACE s_axilite port=sigma bundle=control
#pragma HLS INTERFACE s_axilite port=Spot bundle=control
#pragma HLS INTERFACE s_axilite port=SRef bundle=control
#pragma HLS INTERFACE s_axilite port=theta bundle=control
#pragma HLS INTERFACE s_axilite port=vSpot bundle=control
#pragma HLS INTERFACE s_axilite port=nObs bundle=control
#pragma HLS INTERFACE s_axilite port=miMax bundle=control
#pragma HLS INTERFACE s_axilite port=DEtimeTable bundle=control
#pragma HLS INTERFACE s_axilite port=DEtable bundle=control
#pragma HLS INTERFACE s_axilite port=DeltaT bundle=control
#pragma HLS INTERFACE s_axilite port=Mcor bundle=control
#pragma HLS INTERFACE s_axilite port=r bundle=control
#pragma HLS INTERFACE s_axilite port=payoff_sum bundle=control
#pragma HLS INTERFACE s_axilite port=dirnum bundle=control
#pragma HLS INTERFACE s_axilite port=c_data bundle=control
#pragma HLS INTERFACE s_axilite port=l_data bundle=control
#pragma HLS INTERFACE s_axilite port=r_data bundle=control
#pragma HLS INTERFACE s_axilite port=qasave bundle=control
#pragma HLS INTERFACE s_axilite port=qbsave bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
	// clang-format off

    sci::hls::sobol_joe_kuo<TS_MAX * 3 * ND_MAX> qrng_;
    sci::hls::brownian_bridge<real_t, TS_MAX> bb_;

    qrng_.init(dirnum, miMax * 3 * nD);
    bb_.init(miMax, c_data, l_data, r_data, qasave, qbsave);

    // initialize input parameters structure
    pricer_kernel_data data;

    data.AccruedBonus0 = AccruedBonus0;
    data.BonusCoup = BonusCoup;
    data.KIBarrier = KIBarrier;
    data.KINum = KINum;
    data.MatCoup = MatCoup;
    data.nD = nD;
    data.Notional = Notional;
    data.miMax = miMax;

    for (int i = 1; i <= nObs; i++)
    {
    	// clang-format off
    	#pragma HLS loop_tripcount min=tc_nobs max=tc_nobs
        // clang-format on
        data.CallBarrier[i] = CallBarrier[i];
        data.DeltaT[i] = DeltaT[i];
    }

    for (int iD = 1; iD <= nD; iD++)
    {
// clang-format off
#pragma HLS loop_tripcount min=tc_nd max=tc_md
        // clang-format on
        data.kappa[iD] = kappa[iD];
        data.KI0[iD] = KI0[iD];
        data.q[iD] = q[iD];
        data.sigma[iD] = sigma[iD];
        data.Spot[iD] = Spot[iD];
        data.SRef[iD] = SRef[iD];
        data.theta[iD] = theta[iD];
        data.vSpot[iD] = vSpot[iD];
    }

    for (int n = 0; n <= miMax; n++)
    {
// clang-format off
#pragma HLS loop_tripcount min=tc_mimax max=tc_mimax
        // clang-format on
        data.DEtimeTable[n] = DEtimeTable[n];
        data.r[n] = r[n];
        for (int id = 0; id < 3; id++)
        {
            data.DEtable[n][id] = DEtable[n * 3 + id];
        }
    }

    for (int iD1 = 1; iD1 <= 2 * nD; iD1++)
    {
        // clang-format off
#pragma HLS loop_tripcount min=2*tc_nd max=2*tc_nd
        // clang-format on
        for (int iD2 = 1; iD2 <= 2 * nD; iD2++)
        {
            // clang-format off
    #pragma HLS loop_tripcount min=2*tc_nd max=2*tc_nd
            // clang-format on
            data.Mcor[iD1][iD2] = Mcor[iD1 * (2 * nD + 1) + iD2];
        }
    }

    real_t payoffSumTot = 0;

    real_t payoffSum[NUM_BLOCKS], payoffSumE[NUM_BLOCKS];
    pricer_kernel_data data_[NUM_BLOCKS];
    sci::hls::sobol_joe_kuo<TS_MAX * 3 * ND_MAX> qrng[NUM_BLOCKS];
    sci::hls::brownian_bridge<real_t, TS_MAX> bb[NUM_BLOCKS];
    // clang-format off
#pragma HLS ARRAY_PARTITION variable=payoffSum complete
#pragma HLS ARRAY_PARTITION variable=data_ complete
#pragma HLS ARRAY_PARTITION variable=qrng complete
#pragma HLS ARRAY_PARTITION variable=bb complete
    // clang-format on

    // replicate input data for each calculation block
    for (int i = 0; i < NUM_BLOCKS; i++)
    {
        // clang-format off
#pragma HLS unroll
        // clang-format on
        data_[i] = data;
        qrng[i] = qrng_;
        bb[i] = bb_;
    }

    // run calculation blocks in parallel (full unroll)
    for (int i = 0; i < NUM_BLOCKS; i++)
    {
        // clang-format off
#pragma HLS unroll
        // clang-format on
        pricer_kernel_block(
            qrng[i],
            bb[i],
            data_[i],
            pmax / NUM_BLOCKS,
            seq,
            payoffSum[i],
            i);
    }

    for (int i = 0; i < NUM_BLOCKS; i++)
    {
        payoffSumTot += payoffSum[i];
    }

    *payoff_sum = payoffSumTot;
}
