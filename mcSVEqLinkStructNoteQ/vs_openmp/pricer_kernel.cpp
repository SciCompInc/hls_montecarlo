#include <algorithm>
#include <omp.h>
#include <vector>

#include <sci/brownian_bridge.h>
#include <sci/distributions.h>
#include <sci/sobol_joe_kuo.h>

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

    /* *** Key to program variables: *** */
    /* asum: accumulator for sum */
    /* iD: index variable for RhoSS */
    /* KI: solution variable */
    /* nD: array maximum for RhoSS, RhoSv, Rhovv, Spot, SRef, q, kappa, theta,
     * vSpot, sigma and KI0 */
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
void CholvmmMtx(Real* L, Real* x, int dim, int steps)
{
    /* L lower triangular matrix*/
    /* x=L*x */
    for (int n = 1; n <= steps; n++)
    {
        for (int i3 = dim; i3 >= 1; i3--)
        {
            Real csum = 0.0;
            for (int j1 = 1; j1 <= i3; j1++)
            {
                csum += L[i3 * (dim + 1) + j1] * x[j1 * (steps + 1) + n];
            }
            x[i3 * (steps + 1) + n] = csum;
        }
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
    int pMax,
    real_t* q,
    int series,
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
    real_t* V)
{
    real_t payoffSum = 0.;

#pragma omp parallel reduction(+ : payoffSum)
    {
        const real_t tfloor = real_t(0.00001);
        const real_t vfloor = real_t(0.00001);

        int iD, iObs, n;
        real_t AccruedBonus, discount, dt, dtCC, EarlyPayout, payoff, t, test,
            tOldCC, WorstPerf;

        int Redeemed;

        int tid = omp_get_thread_num();

        int nThreads = omp_get_num_threads();

        int ndim = miMax * 3 * nD;
        std::vector<unsigned int> state(ndim, 0);
        std::vector<real_t> zu(ndim);
        std::vector<real_t> zn((2 * nD + 1) * miMax);
        std::vector<real_t> zb((2 * nD + 1) * (miMax + 1));

        sci::sobol_joe_kuo qrng(dirnum, state.data(), ndim);
        qrng.skip(pMax * series + tid * pMax / nThreads);

        std::vector<real_t> S(nD + 1);
        std::vector<real_t> v(nD + 1);
        std::vector<real_t> KI(nD + 1);
        std::vector<real_t> SOldCC(nD + 1);
        std::vector<real_t> vProt(nD + 1);
        std::vector<real_t> arr1(nD + 1);
        std::vector<real_t> Z(3 * nD + 1);

#pragma omp for
        for (int path = 1; path <= pMax; path++)
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

            qrng.next();
            for (int i = 0; i < ndim; i++)
            {
                // state : (miMax) x (3*nD)
                // zu : (miMax) x (3*nD)
                sci::uniform_distribution_transform(state[i], zu[i]);
            }
            for (int iD = 1; iD <= 2 * nD; iD++)
            {
                for (int n = 0; n < miMax; n++)
                {
                    // zn: (2*nD + 1) x (miMax)
                    zn[iD * miMax + n] = sci::normal_distribution_icdf(
                        zu[n * (3 * nD) + (iD - 1)]);
                }
                // zb : (2*nD + 1) x (miMax + 1)
                sci::brownian_bridge(
                    &zn[iD * miMax],
                    &zb[iD * (miMax + 1)],
                    c_data,
                    l_data,
                    r_data,
                    qasave,
                    qbsave,
                    miMax);
            }
            CholvmmMtx(Mcor, zb.data(), 2 * nD, miMax);
            n = 0;
            while (n <= miMax - 1)
            {
                dt = DEtimeTable[n + 1] - t;
                /* Take a time step. */
                for (iD = 1; iD <= nD; iD++)
                {
                    vProt[iD] = max(vfloor, v[iD]);
                }
                /* Computing Ztilda from equation Eq9c; formula is
                 * stubroutinevar2 == CholvmmN[Mcor, Ztilda]. */
                /* Computing v from equation Eqvc; formula is der[v, {t, 1}] ==
                   sigma*Sqrt[vProt]*Ztilda*Sqrt[delta[t]] + kappa*(theta -
                   vProt)*delta[t]. */
                for (iD = 1; iD <= nD; iD++)
                {
                    v[iD] += dt * kappa[iD] * (theta[iD] - vProt[iD]) +
                        sigma[iD] * zb[(iD + nD) * (miMax + 1) + (n + 1)] *
                            sqrt(vProt[iD]);
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
                            zb[iD * (miMax + 1) + (n + 1)] * sqrt(vProt[iD]));
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
                                real_t denom = max(dtCC * vProt[iD], tfloor);
                                real_t test1 =
                                    exp((real_t(2) *
                                         log(S[iD] / (SRef[iD] * KIBarrier)) *
                                         log((KIBarrier * SRef[iD]) /
                                             SOldCC[iD])) /
                                        denom);
                                test = zu[n * (3 * nD) + 2 * nD + iD - 1];
                                if ((test < test1 && test1 <= real_t(1)))
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
        }
    }
    payoffSum /= real_t(pMax);
    *V = payoffSum;
}
