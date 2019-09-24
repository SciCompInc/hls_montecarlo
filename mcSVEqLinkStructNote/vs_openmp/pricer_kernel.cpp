#include <algorithm>
#include <omp.h>
#include <vector>

#include <sci/distributions.h>
#include <sci/xoshiro128.h>

#include "kernel_global.h"

using std::max;
using std::min;

template<typename Real>
Real ArrayMin(Real* arr, int nD)
{
    int iD;
    Real amin;
    amin = arr[1];
    if (nD <= 0)
    {
        return Real(0);
    }
    else
    {
        for (iD = 1; iD <= nD; iD++)
        {
            amin = min(amin, arr[iD]);
        }
    }
    return amin;
}

template<typename Real>
Real ArraySum(Real* arr, int nD)
{
    int iD;
    Real asum;
    asum = Real(0);
    if (nD <= 0)
    {
        return Real(0);
    }
    else
    {
        for (iD = 1; iD <= nD; iD++)
        {
            asum = asum + arr[iD];
        }
    }
    return asum;
}

template<typename Real>
void Cholvmm(Real* L, Real* x, int dim)
{
    /* L lower triangular matrix*/
    /* x=L*x */
    for (int i3 = dim; i3 >= 1; i3--)
    {
        Real csum = 0.0;
        for (int j1 = 1; j1 <= i3; j1++)
        {
            csum += L[i3 * (dim + 1) + j1] * x[j1];
        }
        x[i3] = csum;
    }
}

void pricer_kernel(
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
    real_t* V,
    real_t* dev)
{
    real_t payoffSum = 0.;
    real_t payoffSumE = 0.;

#pragma omp parallel reduction(+ : payoffSum, payoffSumE)
    {
        const real_t tfloor = real_t(0.00001);
        const real_t vfloor = real_t(0.00001);

        int c_num_threads = omp_get_num_threads();

        int iD, iObs, n;
        real_t AccruedBonus, discount, dt, dtCC, EarlyPayout, payoff, t, temp1,
            test, test1, tOldCC, WorstPerf;

        int tid = omp_get_thread_num();

        int Redeemed;
        sci::xoshiro128starstar rng;
        const unsigned int seed = 1;
        rng.seed(seed);
        rng.jump(tid + 1 + (seq - 1) * c_num_threads);

        std::vector<real_t> S(nD + 1);
        std::vector<real_t> v(nD + 1);
        std::vector<real_t> KI(nD + 1);
        std::vector<real_t> SOldCC(nD + 1);
        std::vector<real_t> vProt(nD + 1);
        std::vector<real_t> arr1(nD + 1);
        std::vector<real_t> Z(3 * nD + 1);

#pragma omp for
        for (int path = 1; path <= pmax; path++)
        {
            /* Initialize time */
            t = real_t(0);
            /* Initialize local loop counter */
            /* Initialize time-dependent equations */
            for (iD = 1; iD <= nD; iD++)
            {
                /* Initial value for S from IVEq1c. */
                S[iD] = Spot[iD];
                /* Initial value for v from IVEq2c. */
                v[iD] = vSpot[iD];
                KI[iD] = KI0[iD];
            }
            Redeemed = 0;
            AccruedBonus = AccruedBonus0;
            discount = real_t(0);
            EarlyPayout = real_t(0);
            WorstPerf = real_t(0);
            /* Begin processing discrete events. */
            /* Reset discrete indexes. */
            iObs = 1;
            tOldCC = t;
            for (iD = 1; iD <= nD; iD++)
            {
                SOldCC[iD] = Spot[iD];
            }
            n = 0;
            while (n <= miMax - 1)
            {
                dt = DEtimeTable[n + 1] - t;
                /* Take a time step. */
                for (iD = 1; iD <= nD; iD++)
                {
                    vProt[iD] = max(vfloor, v[iD]);
                }
                for (iD = 1; iD <= 3 * nD; iD++)
                {
                    sci::uniform_distribution_transform(rng.next32(), Z[iD]);
                }
                for (iD = 1; iD <= 2 * nD; iD++)
                {
                    Z[iD] = sci::normal_distribution_icdf(Z[iD]);
                }
                /* Computing Ztilda from equation Eq9c; formula is
                 * stubroutinevar2 == CholvmmN[Mcor, Ztilda]. */
                Cholvmm(Mcor, Z.data(), 2 * nD);
                temp1 = sqrt(dt);
                /* Computing v from equation Eqvc; formula is der[v, {t, 1}] ==
                   sigma*Sqrt[vProt]*Ztilda*Sqrt[delta[t]] + kappa*(theta -
                   vProt)*delta[t]. */
                for (iD = 1; iD <= nD; iD++)
                {
                    v[iD] += dt * kappa[iD] * (theta[iD] - vProt[iD]) +
                        temp1 * sigma[iD] * Z[iD + nD] * sqrt(vProt[iD]);
                }
                /* Computing discount from equation Eqrc; formula is
                 * der[discount, {t, 1}] == r. */
                discount += dt * (r[n] + r[n + 1]) * real_t(0.5);
                /* Computing S from equation EqSc; formula is der[S, {t, 1}] ==
                   S*(Sqrt[vProt]*Ztilda*Sqrt[delta[t]] + (-q + r)*delta[t]).
                 */
                for (iD = 1; iD <= nD; iD++)
                {
                    S[iD] = S[iD] *
                        exp(dt * (r[n] + vProt[iD] * (real_t)(-0.5) - q[iD]) +
                            temp1 * Z[iD] * sqrt(vProt[iD]));
                }
                /* General discrete event updates. */
                /* update for Path[function[KI==(if[Redeemed==0, (if[((S
                   SRef^-1) <= KIBarrier), 1, KI]), KI])], direction[KI],
                   tsample==BarDates, ContinuityCorrection] */
                if ((Redeemed == 0 && DEtable[(n + 1) * 3 + 1] == 1))
                {
                    dtCC = dt + t - tOldCC;
                    for (iD = 1; iD <= nD; iD++)
                    {
                        if (S[iD] / SRef[iD] <= KIBarrier)
                        {
                            KI[iD] = real_t(1);
                        }
                        else
                        {
                            if (KIBarrier > real_t(0))
                            {
                                real_t denom = dtCC * vProt[iD];
                                if (denom < tfloor)
                                    denom = tfloor;
                                test1 =
                                    exp((real_t(2) *
                                         log(S[iD] / (SRef[iD] * KIBarrier)) *
                                         log((KIBarrier * SRef[iD]) /
                                             SOldCC[iD])) /
                                        denom);
                                test = Z[iD + 2 * nD];
                                if ((test < test1 && test1 <= real_t(1.)))
                                {
                                    KI[iD] = real_t(1);
                                }
                            }
                        }
                        SOldCC[iD] = S[iD];
                    }
                    tOldCC = dt + t;
                }
                /* update for Path[direction[AccruedBonus],
                   (function[if[Redeemed==0, AccruedBonus==(AccruedBonus +
                   (BonusCoup DeltaT)); WorstPerf==ArrayMin[(S SRef^-1)];
                   EarlyPayout==(if[(WorstPerf > CallBarrier), (1 +
                   AccruedBonus), EarlyPayout]); Redeemed==(if[(WorstPerf >
                   CallBarrier), 1, Redeemed])]]), tsample==ObsDates] */
                if ((Redeemed == 0 && DEtable[(n + 1) * 3 + 2] == 1))
                {
                    AccruedBonus += BonusCoup * DeltaT[iObs];
                    for (iD = 1; iD <= nD; iD++)
                    {
                        arr1[iD] = S[iD] / SRef[iD];
                    }
                    WorstPerf = ArrayMin(arr1.data(), nD);
                    if (WorstPerf > CallBarrier[iObs])
                    {
                        EarlyPayout = (AccruedBonus + 1) * exp(-discount);
                        Redeemed = 1;
                    }
                }
                /* Update time variables  */
                /* Update time */
                t += dt;
                /* Update local loop counter */
                n++;
                /* Discrete event index updates. */
                if (DEtable[n * 3 + 2] == 1)
                {
                    iObs++;
                }
            }
            if (Redeemed)
            {
                payoff = EarlyPayout * Notional;
            }
            else
            {
                if (ArraySum(KI.data(), nD) >= KINum)
                {
                    for (iD = 1; iD <= nD; iD++)
                    {
                        arr1[iD] = S[iD] / SRef[iD];
                    }
                    payoff = ArrayMin(arr1.data(), nD) * Notional;
                }
                else
                {
                    payoff = (MatCoup + 1) * Notional;
                }
            }
            payoffSum += payoff;
            payoffSumE += payoff * payoff;
        }
    }
    payoffSum /= pmax;
    *V = payoffSum;
    payoffSumE /= pmax;
    *dev = (payoffSumE - payoffSum * payoffSum) / pmax;
    *dev = sqrt(*dev);
}
