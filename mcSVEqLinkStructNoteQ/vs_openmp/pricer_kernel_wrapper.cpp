#include <omp.h>
#include <vector>

#include <sci/brownian_bridge_setup.h>
#include <sci/sobol_joe_kuo_dirnum.h>

#include "kernel_global.h"

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
    real_t* V);

void pricer_kernel_wrapper(
    const std::vector<double>& DeltaT,
    const std::vector<double>& r,
    const std::vector<int[3]>& DEtable,
    const std::vector<double>& DEtimeTable,
    const std::vector<std::vector<double>>& Mcor,
    double AccruedBonus0,
    double BonusCoup,
    const std::vector<double>& CallBarrier,
    const std::vector<double>& kappa,
    const std::vector<double>& KI0,
    double KIBarrier,
    double KINum,
    double MatCoup,
    double Notional,
    int pmax,
    const std::vector<double>& q,
    int seq,
    const std::vector<double>& sigma,
    const std::vector<double>& Spot,
    const std::vector<double>& SRef,
    const std::vector<double>& theta,
    const std::vector<double>& vSpot,
    double& devx,
    double& Vx,
    double& tk)
{
    real_t V;

    int nObs = CallBarrier.size();
    int nD = Spot.size();
    int miMax = r.size() - 1;

    // kernel transfer data

    std::vector<real_t> CallBarrier_(nObs + 1);
    for (int i = 0; i < nObs; i++)
    {
        CallBarrier_[i + 1] = (real_t)CallBarrier[i];
    }

    std::vector<real_t> kappa_(nD + 1);
    std::vector<real_t> KI0_(nD + 1);
    std::vector<real_t> q_(nD + 1);
    std::vector<real_t> sigma_(nD + 1);
    std::vector<real_t> Spot_(nD + 1);
    std::vector<real_t> SRef_(nD + 1);
    std::vector<real_t> theta_(nD + 1);
    std::vector<real_t> vSpot_(nD + 1);
    for (int i = 0; i < nD; i++)
    {
        kappa_[i + 1] = (real_t)kappa[i];
        KI0_[i + 1] = (real_t)KI0[i];
        q_[i + 1] = (real_t)q[i];
        sigma_[i + 1] = (real_t)sigma[i];
        Spot_[i + 1] = (real_t)Spot[i];
        SRef_[i + 1] = (real_t)SRef[i];
        theta_[i + 1] = (real_t)theta[i];
        vSpot_[i + 1] = (real_t)vSpot[i];
    }

    std::vector<real_t> DeltaT_(nObs + 1);
    for (int i = 1; i <= nObs; i++)
    {
        DeltaT_[i] = (real_t)DeltaT[i];
    }

    std::vector<real_t> r_(miMax + 1);
    for (int i = 0; i <= miMax; i++)
    {
        r_[i] = (real_t)r[i];
    }

    std::vector<real_t> DEtimeTable_(miMax + 1);
    for (int i = 0; i <= miMax; i++)
    {
        DEtimeTable_[i] = (real_t)DEtimeTable[i];
    }

    std::vector<int> DEtable_((miMax + 1) * 3);
    for (int i = 0; i <= miMax; i++)
    {
        for (int id = 0; id < 3; id++)
            DEtable_[i * 3 + id] = DEtable[i][id];
    }

    std::vector<real_t> Mcor_((2 * nD + 1) * (2 * nD + 1));
    for (int i = 1; i <= 2 * nD; i++)
    {
        for (int j = 1; j <= 2 * nD; j++)
        {
            Mcor_[i * (2 * nD + 1) + j] = (real_t)Mcor[i - 1][j - 1];
        }
    }

    const int nbit = sci::new_sobol_joe_kuo_6_21201_nbit;
    const int ndim = miMax * 3 * nD;
    std::vector<unsigned int> dirnum(ndim * nbit);
    std::vector<unsigned int> shift(ndim, 0);
    sci::new_sobol_joe_kuo_6_21201_dirnum(
        ndim, dirnum.data(), shift.data(), false);

    std::vector<int> c_data(miMax + 1);
    std::vector<int> l_data(miMax + 1);
    std::vector<int> r_data(miMax + 1);
    std::vector<real_t> qasave(miMax + 1);
    std::vector<real_t> qbsave(miMax + 1);

    sci::brownian_bridge_setup(
        miMax,
        DEtimeTable_.data(),
        c_data.data(),
        l_data.data(),
        r_data.data(),
        qasave.data(),
        qbsave.data());

    double t1 = omp_get_wtime();
    pricer_kernel(
        (real_t)AccruedBonus0,
        (real_t)BonusCoup,
        CallBarrier_.data(),
        kappa_.data(),
        KI0_.data(),
        (real_t)KIBarrier,
        (real_t)KINum,
        (real_t)MatCoup,
        nD,
        (real_t)Notional,
        pmax,
        q_.data(),
        seq,
        sigma_.data(),
        Spot_.data(),
        SRef_.data(),
        theta_.data(),
        vSpot_.data(),
        nObs,
        miMax,
        DEtimeTable_.data(),
        DEtable_.data(),
        DeltaT_.data(),
        Mcor_.data(),
        r_.data(),
        dirnum.data(),
        c_data.data(),
        l_data.data(),
        r_data.data(),
        qasave.data(),
        qbsave.data(),
        &V);

    double t2 = omp_get_wtime();
    //	std::cout << (t2 - t1) * 1000 << " ms" << std::endl;

    Vx = (double)V;
    devx = double(0);
    tk = (t2 - t1) * 1000;
}
