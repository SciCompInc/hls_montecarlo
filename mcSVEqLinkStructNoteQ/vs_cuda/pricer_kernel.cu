#include "sci/brownian_bridge_setup.h"
#include "sci/cuda/blas_util.h"
#include "sci/cuda/brownian_bridge.h"
#include "sci/cuda/cuda_common.h"
#include "sci/cuda/distributions.h"
#include "sci/cuda/sobol.h"
#include "sci/sobol_joe_kuo_dirnum.h"
#include <cfloat>
#include <vector>

#include <sci/num_utils.h>

#define SCIFIN_DOUBLE_PRECISION

#ifdef SCIFIN_DOUBLE_PRECISION
typedef double data_t;
#define FPTYPE(x) x
#else
typedef float data_t;
#define FPTYPE(x) x##f
#endif

namespace mcgpuSVEqLinkStructNote1Q
{
__constant__ data_t AccruedBonus0;
__constant__ data_t BonusCoup;
__constant__ data_t* dCallBarrier;
__constant__ data_t* dDeltaT;
__constant__ int* dDEtable;
__constant__ data_t* dDEtimeTable;
__constant__ unsigned int* div1;
__constant__ data_t* dkappa;
__constant__ data_t* dKI0;
__constant__ int* dlnsave;
__constant__ data_t* dMcor;
__constant__ int* dncsave;
__constant__ data_t* dq;
__constant__ data_t* dqasave;
__constant__ data_t* dqbsave;
__constant__ data_t* dr;
__constant__ int* drnsave;
__constant__ data_t* dsigma;
__constant__ data_t* dSpot;
__constant__ data_t* dSRef;
__constant__ data_t* dtheta;
__constant__ data_t* dvSpot;
__constant__ data_t KIBarrier;
__constant__ data_t KINum;
__constant__ data_t MatCoup;
__constant__ int miMax;
__constant__ int nBit;
__constant__ int nD;
__constant__ int nDim;
__constant__ int nMax;
__constant__ int nObs;
__constant__ data_t Notional;
__constant__ int pmax;
__constant__ int sskip;

void init()
{
    int devID = 0;
    SCI_CUDART_CALL(cudaSetDevice(0));
    cudaDeviceProp deviceProp;
    SCI_CUDART_CALL(cudaGetDeviceProperties(&deviceProp, devID));
    if (deviceProp.major < 3)
        throw std::runtime_error(
            "Device with Compute Capability 3.0 or higher is required");

    SCI_CUDART_CALL(cudaFree(0));
    SCI_CUDART_CALL(cudaDeviceSynchronize());
}

void finalize() { SCI_CUDART_CALL(cudaDeviceReset()); }

} // namespace mcgpuSVEqLinkStructNote1Q

using namespace sci;

namespace
{
template<typename Real>
__device__ Real ArrayMinSP(const Array1D<Real>& arr, int nD);
} // namespace

namespace
{
template<typename Real>
__device__ Real ArraySumSP(const Array1D<Real>& KI, int nD);
} // namespace

namespace mcgpuSVEqLinkStructNote1Q
{
template<typename Real>
__global__ void pricerKernel(
    Real* darr,
    Real* darr1,
    unsigned int* dix1,
    Real* dKI,
    Real* dpayoffSum,
    Real* dQRZ,
    Real* dS,
    Real* dSOldCC,
    Real* dv,
    Real* dvProt,
    Real* dZ);
} // namespace mcgpuSVEqLinkStructNote1Q

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
    data_t V;

    int nObs = CallBarrier.size();
    int nD = Spot.size();
    int miMax = r.size() - 1;

    int series = seq;

    std::vector<data_t> CallBarrier_(nObs + 1);
    for (int i = 0; i < nObs; i++)
    {
        CallBarrier_[i + 1] = (data_t)CallBarrier[i];
    }

    std::vector<data_t> kappa_(nD + 1);
    std::vector<data_t> KI0_(nD + 1);
    std::vector<data_t> q_(nD + 1);
    std::vector<data_t> sigma_(nD + 1);
    std::vector<data_t> Spot_(nD + 1);
    std::vector<data_t> SRef_(nD + 1);
    std::vector<data_t> theta_(nD + 1);
    std::vector<data_t> vSpot_(nD + 1);
    for (int i = 0; i < nD; i++)
    {
        kappa_[i + 1] = (data_t)kappa[i];
        KI0_[i + 1] = (data_t)KI0[i];
        q_[i + 1] = (data_t)q[i];
        sigma_[i + 1] = (data_t)sigma[i];
        Spot_[i + 1] = (data_t)Spot[i];
        SRef_[i + 1] = (data_t)SRef[i];
        theta_[i + 1] = (data_t)theta[i];
        vSpot_[i + 1] = (data_t)vSpot[i];
    }

    std::vector<data_t> DeltaT_(nObs + 1);
    for (int i = 1; i <= nObs; i++)
    {
        DeltaT_[i] = (data_t)DeltaT[i];
    }

    std::vector<data_t> r_(miMax + 1);
    for (int i = 0; i <= miMax; i++)
    {
        r_[i] = (data_t)r[i];
    }

    std::vector<data_t> DEtimeTable_(miMax + 1);
    for (int i = 0; i <= miMax; i++)
    {
        DEtimeTable_[i] = (data_t)DEtimeTable[i];
    }

    std::vector<int> DEtable_((miMax + 1) * 3);
    for (int i = 0; i <= miMax; i++)
    {
        for (int id = 0; id < 3; id++)
            DEtable_[i * 3 + id] = DEtable[i][id];
    }

    std::vector<data_t> Mcor_((2 * nD + 1) * (2 * nD + 1));
    for (int i = 1; i <= 2 * nD; i++)
    {
        for (int j = 1; j <= 2 * nD; j++)
        {
            Mcor_[i * (2 * nD + 1) + j] = (data_t)Mcor[i - 1][j - 1];
        }
    }
    int sskip = pmax * series;
    const int nBit = 32;
    const int nDim = miMax * 3 * nD;
    std::vector<unsigned int> dirnum_(nDim * nBit);
    std::vector<unsigned int> shift(nDim);
    sci::new_sobol_joe_kuo_6_21201_dirnum(nDim, dirnum_.data(), shift.data());

    // Convert to 1 - based:
    std::vector<unsigned int> dirnum((nDim + 1) * (nBit + 1));
    for (int i = 0; i < nDim; i++)
    {
        for (int j = 0; j < nBit; j++)
        {
            dirnum[(i + 1) * (nBit + 1) + (j + 1)] = dirnum_[i * nBit + j];
        }
    }

    std::vector<int> c_data(miMax + 1);
    std::vector<int> l_data(miMax + 1);
    std::vector<int> r_data(miMax + 1);
    std::vector<data_t> qasave(miMax + 1);
    std::vector<data_t> qbsave(miMax + 1);

    sci::brownian_bridge_setup(
        miMax,
        DEtimeTable_.data(),
        c_data.data(),
        l_data.data(),
        r_data.data(),
        qasave.data(),
        qbsave.data());

    SCI_CUDART_CALL(cudaSetDevice(0));

    //  Input device arrays

    DeviceArray<data_t> dCallBarrier(CallBarrier_);
    DeviceArray<data_t> dDeltaT(DeltaT_);
    DeviceArray<int> dDEtable(DEtable_);
    DeviceArray<data_t> dDEtimeTable(DEtimeTable_);
    DeviceArray<unsigned int> div1(dirnum);
    DeviceArray<data_t> dkappa(kappa_);
    DeviceArray<data_t> dKI0(KI0_);
    DeviceArray<int> dlnsave(l_data);
    DeviceArray<data_t> dMcor(Mcor_);
    DeviceArray<int> dncsave(c_data);
    DeviceArray<data_t> dq(q_);
    DeviceArray<data_t> dqasave(qasave);
    DeviceArray<data_t> dqbsave(qbsave);
    DeviceArray<data_t> dr(r_);
    DeviceArray<int> drnsave(r_data);
    DeviceArray<data_t> dsigma(sigma_);
    DeviceArray<data_t> dSpot(Spot_);
    DeviceArray<data_t> dSRef(SRef_);
    DeviceArray<data_t> dtheta(theta_);
    DeviceArray<data_t> dvSpot(vSpot_);

    /* Thread geometry */
    dim3 grid;
    dim3 block;

    block.x = 256;
    grid.x = (pmax + block.x - 1) / (block.x);
    if (grid.x < 1)
        grid.x = 1;
    sciCUDAOptimumGrid1D(grid.x, 0);

    /* Allocate host and device output arrays */
    std::vector<data_t> hpayoffSum(block.x * grid.x);
    DeviceArray<data_t> dpayoffSum(block.x * grid.x);

    //  Working device local arrays

    DeviceArray<data_t> darr(block.x * grid.x * (1 + nD));
    DeviceArray<data_t> darr1(block.x * grid.x * (1 + nD));
    DeviceArray<unsigned int> dix1(block.x * grid.x * (1 + nDim));
    DeviceArray<data_t> dKI(block.x * grid.x * (1 + nD));
    DeviceArray<data_t> dQRZ(block.x * grid.x * (1 + miMax) * (1 + 2 * nD));
    DeviceArray<data_t> dS(block.x * grid.x * (1 + nD));
    DeviceArray<data_t> dSOldCC(block.x * grid.x * (1 + nD));
    DeviceArray<data_t> dv(block.x * grid.x * (1 + nD));
    DeviceArray<data_t> dvProt(block.x * grid.x * (1 + nD));
    DeviceArray<data_t> dZ(block.x * grid.x * (1 + miMax) * (1 + 3 * nD));

    /*  Call kernel function */
    /*                                 */
    /* set up arguments for the kernel */
    SCI_CUDART_CALL(cudaMemcpyToSymbol(
        mcgpuSVEqLinkStructNote1Q::AccruedBonus0,
        &AccruedBonus0,
        sizeof(AccruedBonus0)));
    SCI_CUDART_CALL(cudaMemcpyToSymbol(
        mcgpuSVEqLinkStructNote1Q::BonusCoup, &BonusCoup, sizeof(BonusCoup)));
    data_t* dCallBarrierX = dCallBarrier.array();
    SCI_CUDART_CALL(cudaMemcpyToSymbol(
        mcgpuSVEqLinkStructNote1Q::dCallBarrier,
        &dCallBarrierX,
        sizeof(dCallBarrierX)));
    data_t* dDeltaTX = dDeltaT.array();
    SCI_CUDART_CALL(cudaMemcpyToSymbol(
        mcgpuSVEqLinkStructNote1Q::dDeltaT, &dDeltaTX, sizeof(dDeltaTX)));
    int* dDEtableX = dDEtable.array();
    SCI_CUDART_CALL(cudaMemcpyToSymbol(
        mcgpuSVEqLinkStructNote1Q::dDEtable, &dDEtableX, sizeof(dDEtableX)));
    data_t* dDEtimeTableX = dDEtimeTable.array();
    SCI_CUDART_CALL(cudaMemcpyToSymbol(
        mcgpuSVEqLinkStructNote1Q::dDEtimeTable,
        &dDEtimeTableX,
        sizeof(dDEtimeTableX)));
    unsigned int* div1X = div1.array();
    SCI_CUDART_CALL(cudaMemcpyToSymbol(
        mcgpuSVEqLinkStructNote1Q::div1, &div1X, sizeof(div1X)));
    data_t* dkappaX = dkappa.array();
    SCI_CUDART_CALL(cudaMemcpyToSymbol(
        mcgpuSVEqLinkStructNote1Q::dkappa, &dkappaX, sizeof(dkappaX)));
    data_t* dKI0X = dKI0.array();
    SCI_CUDART_CALL(cudaMemcpyToSymbol(
        mcgpuSVEqLinkStructNote1Q::dKI0, &dKI0X, sizeof(dKI0X)));
    int* dlnsaveX = dlnsave.array();
    SCI_CUDART_CALL(cudaMemcpyToSymbol(
        mcgpuSVEqLinkStructNote1Q::dlnsave, &dlnsaveX, sizeof(dlnsaveX)));
    data_t* dMcorX = dMcor.array();
    SCI_CUDART_CALL(cudaMemcpyToSymbol(
        mcgpuSVEqLinkStructNote1Q::dMcor, &dMcorX, sizeof(dMcorX)));
    int* dncsaveX = dncsave.array();
    SCI_CUDART_CALL(cudaMemcpyToSymbol(
        mcgpuSVEqLinkStructNote1Q::dncsave, &dncsaveX, sizeof(dncsaveX)));
    data_t* dqX = dq.array();
    SCI_CUDART_CALL(
        cudaMemcpyToSymbol(mcgpuSVEqLinkStructNote1Q::dq, &dqX, sizeof(dqX)));
    data_t* dqasaveX = dqasave.array();
    SCI_CUDART_CALL(cudaMemcpyToSymbol(
        mcgpuSVEqLinkStructNote1Q::dqasave, &dqasaveX, sizeof(dqasaveX)));
    data_t* dqbsaveX = dqbsave.array();
    SCI_CUDART_CALL(cudaMemcpyToSymbol(
        mcgpuSVEqLinkStructNote1Q::dqbsave, &dqbsaveX, sizeof(dqbsaveX)));
    data_t* drX = dr.array();
    SCI_CUDART_CALL(
        cudaMemcpyToSymbol(mcgpuSVEqLinkStructNote1Q::dr, &drX, sizeof(drX)));
    int* drnsaveX = drnsave.array();
    SCI_CUDART_CALL(cudaMemcpyToSymbol(
        mcgpuSVEqLinkStructNote1Q::drnsave, &drnsaveX, sizeof(drnsaveX)));
    data_t* dsigmaX = dsigma.array();
    SCI_CUDART_CALL(cudaMemcpyToSymbol(
        mcgpuSVEqLinkStructNote1Q::dsigma, &dsigmaX, sizeof(dsigmaX)));
    data_t* dSpotX = dSpot.array();
    SCI_CUDART_CALL(cudaMemcpyToSymbol(
        mcgpuSVEqLinkStructNote1Q::dSpot, &dSpotX, sizeof(dSpotX)));
    data_t* dSRefX = dSRef.array();
    SCI_CUDART_CALL(cudaMemcpyToSymbol(
        mcgpuSVEqLinkStructNote1Q::dSRef, &dSRefX, sizeof(dSRefX)));
    data_t* dthetaX = dtheta.array();
    SCI_CUDART_CALL(cudaMemcpyToSymbol(
        mcgpuSVEqLinkStructNote1Q::dtheta, &dthetaX, sizeof(dthetaX)));
    data_t* dvSpotX = dvSpot.array();
    SCI_CUDART_CALL(cudaMemcpyToSymbol(
        mcgpuSVEqLinkStructNote1Q::dvSpot, &dvSpotX, sizeof(dvSpotX)));
    SCI_CUDART_CALL(cudaMemcpyToSymbol(
        mcgpuSVEqLinkStructNote1Q::KIBarrier, &KIBarrier, sizeof(KIBarrier)));
    SCI_CUDART_CALL(cudaMemcpyToSymbol(
        mcgpuSVEqLinkStructNote1Q::KINum, &KINum, sizeof(KINum)));
    SCI_CUDART_CALL(cudaMemcpyToSymbol(
        mcgpuSVEqLinkStructNote1Q::MatCoup, &MatCoup, sizeof(MatCoup)));
    SCI_CUDART_CALL(cudaMemcpyToSymbol(
        mcgpuSVEqLinkStructNote1Q::miMax, &miMax, sizeof(miMax)));
    SCI_CUDART_CALL(cudaMemcpyToSymbol(
        mcgpuSVEqLinkStructNote1Q::nBit, &nBit, sizeof(nBit)));
    SCI_CUDART_CALL(
        cudaMemcpyToSymbol(mcgpuSVEqLinkStructNote1Q::nD, &nD, sizeof(nD)));
    SCI_CUDART_CALL(cudaMemcpyToSymbol(
        mcgpuSVEqLinkStructNote1Q::nDim, &nDim, sizeof(nDim)));
    SCI_CUDART_CALL(cudaMemcpyToSymbol(
        mcgpuSVEqLinkStructNote1Q::nObs, &nObs, sizeof(nObs)));
    SCI_CUDART_CALL(cudaMemcpyToSymbol(
        mcgpuSVEqLinkStructNote1Q::Notional, &Notional, sizeof(Notional)));
    SCI_CUDART_CALL(cudaMemcpyToSymbol(
        mcgpuSVEqLinkStructNote1Q::pmax, &pmax, sizeof(pmax)));
    SCI_CUDART_CALL(cudaMemcpyToSymbol(
        mcgpuSVEqLinkStructNote1Q::sskip, &sskip, sizeof(sskip)));
    /*                                 */
    SCI_CUDART_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
    mcgpuSVEqLinkStructNote1Q::pricerKernel KERNEL_ARGS2(grid, block)(
        darr.array(),
        darr1.array(),
        dix1.array(),
        dKI.array(),
        dpayoffSum.array(),
        dQRZ.array(),
        dS.array(),
        dSOldCC.array(),
        dv.array(),
        dvProt.array(),
        dZ.array());
    SCI_CUDART_CALL(cudaGetLastError());
    dpayoffSum.copy(hpayoffSum);
    double payoffSum = 0.;
    for (int thid = 0; thid <= grid.x * block.x - 1; thid++)
    {
        payoffSum += static_cast<double>(hpayoffSum[thid]);
    }
    payoffSum = payoffSum / (double)(pmax);
    V = payoffSum;
    Vx = V;
    devx = 0;
}

namespace
{

template<typename Real>
__device__ Real ArrayMinSP(const Array1D<Real>& arr, int nD)
{
    int iD;
    Real amin;

    /* *** Key to program variables: *** */
    /* amin: accumulator for min */
    /* arr: array-building temporary */
    /* iD: index variable for RhoSS */
    /* nD: array maximum for RhoSS, RhoSv, Rhovv, Spot, SRef, q, kappa, theta,
     * vSpot, sigma and KI0 */
    amin = arr(1);
    if (nD <= 0)
    {
        return Real(0);
    }
    else
    {
        for (iD = 1; iD <= nD; iD++)
        {
            amin = min(amin, arr(iD));
        }
    }
    return amin;
}

template<typename Real>
__device__ Real ArraySumSP(const Array1D<Real>& KI, int nD)
{
    int iD;
    Real asum;

    /* *** Key to program variables: *** */
    /* asum: accumulator for sum */
    /* iD: index variable for RhoSS */
    /* KI: solution variable */
    /* nD: array maximum for RhoSS, RhoSv, Rhovv, Spot, SRef, q, kappa, theta,
     * vSpot, sigma and KI0 */
    asum = Real(0.);
    if (nD <= 0)
    {
        return Real(0);
    }
    else
    {
        for (iD = 1; iD <= nD; iD++)
        {
            asum = asum + KI(iD);
        }
    }
    return asum;
}

} // namespace

namespace mcgpuSVEqLinkStructNote1Q
{

template<typename Real>
__global__ void pricerKernel(
    Real* darr,
    Real* darr1,
    unsigned int* dix1,
    Real* dKI,
    Real* dpayoffSum,
    Real* dQRZ,
    Real* dS,
    Real* dSOldCC,
    Real* dv,
    Real* dvProt,
    Real* dZ)
{
    double eps = DBL_EPSILON;
    int iObsmin = 1;
    double vfloor = 0.00001;
    int iD, iObs, n, nThreads, path, tid;
    Real AccruedBonus, discount, dt, dtCC, EarlyPayout, payoff, t, test, test1,
        tOldCC, WorstPerf;

    int Redeemed;

    /* *** Key to program variables: *** */
    /* AccruedBonus, EarlyPayout, iv1, ix1, KI, lnsave, ncsave, nDim, qasave,
       qbsave, Redeemed, rnsave, S, v, WorstPerf: solution variable */
    /* arr, arr1: array-building temporary */
    /* darr: CUDA temporary for arr */
    /* darr1: CUDA temporary for arr1 */
    /* dDeltaT: CUDA temporary for DeltaT */
    /* dDEtable: CUDA temporary for DEtable */
    /* dDEtimeTable: CUDA temporary for DEtimeTable */
    /* discount: total discount over time */
    /* dix1: CUDA temporary for ix1 */
    /* dKI: CUDA temporary for KI */
    /* dMcor: CUDA temporary for Mcor */
    /* dpayoffSum: CUDA temporary for payoffSum */
    /* dQRZ: CUDA temporary for QRZ */
    /* dr: CUDA temporary for r */
    /* dS: CUDA temporary for S */
    /* dSOldCC: CUDA temporary for SOldCC */
    /* dt: step size for t */
    /* dtCC, SOldCC, tOldCC: continuous barrier temporary */
    /* dv: CUDA temporary for v */
    /* dvProt: CUDA temporary for vProt */
    /* dZ: CUDA temporary for Z */
    /* iD: index variable for RhoSS */
    /* iObs: index variable for ObsDates */
    /* n: index variable for t */
    /* nD: array maximum for RhoSS, RhoSv, Rhovv, Spot, SRef, q, kappa, theta,
     * vSpot, sigma and KI0 */
    /* nThreads: parallel temporary */
    /* path: path index */
    /* payoff: payoff on a trial */
    /* payoffSum: payoff summed over all trials */
    /* t: time variable */
    /* test, test1: continuity correction temporary */
    /* tid: index variable */
    /* vProt: temporary for soft floor of v */
    /* Z: stochastic */
    /* Ztilda: correlated stochastic */
    tid = blockDim.x * blockIdx.x + threadIdx.x;
    nThreads = blockDim.x * gridDim.x;
    Array1D<Real> arr1(darr1, 1 + nD, tid, nThreads);
    Array1D<Real> arr(darr, 1 + nD, tid, nThreads);
    Array1D<Real> CallBarrier(dCallBarrier, 1 + nObs);
    Array1D<Real> DeltaT(dDeltaT, 1 + nObs);
    Array2D<int> DEtable(dDEtable, 1 + miMax, 3);
    Array1D<Real> DEtimeTable(dDEtimeTable, 1 + miMax);
    Array2D<unsigned int> iv1(div1, 1 + nDim, 1 + nBit);
    Array1D<unsigned int> ix1(dix1, 1 + nDim, tid, nThreads);
    Array1D<Real> kappa(dkappa, 1 + nD);
    Array1D<Real> KI0(dKI0, 1 + nD);
    Array1D<Real> KI(dKI, 1 + nD, tid, nThreads);
    Array1D<int> lnsave(dlnsave, 1 + miMax);
    Array2D<Real> Mcor(dMcor, 1 + 2 * nD, 1 + 2 * nD);
    Array1D<int> ncsave(dncsave, 1 + miMax);
    Array1D<Real> qasave(dqasave, 1 + miMax);
    Array1D<Real> qbsave(dqbsave, 1 + miMax);
    Array1D<Real> q(dq, 1 + nD);
    Array1D<Real> r(dr, 1 + miMax);
    Array1D<int> rnsave(drnsave, 1 + miMax);
    Array1D<Real> S(dS, 1 + nD, tid, nThreads);
    Array1D<Real> sigma(dsigma, 1 + nD);
    Array1D<Real> SOldCC(dSOldCC, 1 + nD, tid, nThreads);
    Array1D<Real> Spot(dSpot, 1 + nD);
    Array1D<Real> SRef(dSRef, 1 + nD);
    Array1D<Real> theta(dtheta, 1 + nD);
    Array1D<Real> v(dv, 1 + nD, tid, nThreads);
    Array1D<Real> vProt(dvProt, 1 + nD, tid, nThreads);
    Array1D<Real> vSpot(dvSpot, 1 + nD);
    Array2D<Real> Z(dZ, 1 + 3 * nD, 1 + miMax, tid, nThreads);
    Array2D<Real> QRZ(dQRZ, 1 + 2 * nD, 1 + miMax, tid, nThreads);
    /* Computing iv1 from equation Eq3c; formula is stubroutinevar6 ==
     * SciSobolInit[iv1, ix1, sskip]. */
    SciSobolSeqC<Real, Array1D, Array2D> QRSeqStruct(
        sskip, nDim, nBit, iv1, ix1, nThreads, tid);
    /*                       */
    /* Initialize current state */
    dpayoffSum[tid] = Real(0);
    for (path = 1 + tid; path <= pmax; path = path + nThreads)
    {
        QRSeqStruct.getMatrixNormalPartial(Z, 2 * nD, 1);
        BrownianBridge(Z, QRZ, ncsave, lnsave, rnsave, qasave, qbsave, 1);
        Cholvmm(Mcor, QRZ, 1);
        /* Initialize time */
        t = Real(0);
        n = 0;
        /* Initialize time-dependent equations */
        for (iD = 1; iD <= nD; iD++)
        {
            /* Initial value for S from IVEq1c. */
            S(iD) = Spot(iD);
            /* Initial value for v from IVEq2c. */
            v(iD) = vSpot(iD);
            KI(iD) = KI0(iD);
        }
        Redeemed = 0;
        AccruedBonus = AccruedBonus0;
        discount = Real(0);
        EarlyPayout = Real(0);
        WorstPerf = Real(0);
        /* Begin processing discrete events. */
        /* Reset discrete indexes. */
        iObs = iObsmin;
        tOldCC = t;
        for (iD = 1; iD <= nD; iD++)
        {
            SOldCC(iD) = Spot(iD);
        }
        while (n <= miMax - 1)
        {
            dt = DEtimeTable(n + 1) - t;
            /* Take a time step. */
            for (iD = 1; iD <= nD; iD++)
            {
                vProt(iD) = max(vfloor, v(iD));
                /* Computing v from equation Eqvc; formula is der[v, {t, 1}] ==
                   sigma*Sqrt[vProt]*Ztilda*Sqrt[delta[t]]
                   + kappa*(theta - vProt)*delta[t]. */
                v(iD) += dt * kappa(iD) * (theta(iD) - vProt(iD)) +
                    sigma(iD) * QRZ(iD + nD, n + 1) * FPTYPE(sqrt)(vProt(iD));
            }
            /* Computing discount from equation Eqrc; formula is der[discount,
             * {t, 1}] == r. */
            discount += dt * (r(n) + r(n + 1)) * Real(0.5);
            /* Computing S from equation EqSc; formula is der[S, {t, 1}] ==
               S*(Sqrt[vProt]*Ztilda*Sqrt[delta[t]] + (-q + r)*delta[t]). */
            for (iD = 1; iD <= nD; iD++)
            {
                S(iD) = S(iD) *
                    FPTYPE(exp)(
                            dt * (r(n) + vProt(iD) * (Real)(-0.5) - q(iD)) +
                            QRZ(iD, n + 1) * FPTYPE(sqrt)(vProt(iD)));
            }
            /* General discrete event updates. */
            /* update for Path[function[KI==(if[Redeemed==0, (if[((S SRef^-1)
               <= KIBarrier), 1, KI]), KI])], direction[KI], tsample==BarDates,
               ContinuityCorrection] */
            if ((Redeemed == 0 && DEtable(n + 1, 1) == 1))
            {
                dtCC = dt + t - tOldCC;
                for (iD = 1; iD <= nD; iD++)
                {
                    if (S(iD) / SRef(iD) <= KIBarrier)
                    {
                        KI(iD) = Real(1);
                    }
                    else
                    {
                        if (KIBarrier > Real(0))
                        {
                            test1 = FPTYPE(exp)(
                                (2 *
                                 FPTYPE(log)(S(iD) / (SRef(iD) * KIBarrier)) *
                                 FPTYPE(log)(
                                     (KIBarrier * SRef(iD)) / SOldCC(iD))) /
                                max(dtCC * vProt(iD), eps));
                            test = Z(iD + 2 * nD, n + 1);
                            if ((test < test1 && test1 <= Real(1.)))
                            {
                                KI(iD) = Real(1);
                            }
                        }
                    }
                    SOldCC(iD) = S(iD);
                }
                tOldCC = dt + t;
            }
            /* update for Path[direction[AccruedBonus],
               (function[if[Redeemed==0, AccruedBonus==(AccruedBonus +
               (BonusCoup DeltaT)); WorstPerf==ArrayMin[(S SRef^-1)];
               EarlyPayout==(if[(WorstPerf > CallBarrier), (1 + AccruedBonus),
               EarlyPayout]); Redeemed==(if[(WorstPerf > CallBarrier), 1,
               Redeemed])]]), tsample==ObsDates] */
            if ((Redeemed == 0 && DEtable(n + 1, 2) == 1))
            {
                AccruedBonus += BonusCoup * DeltaT(iObs);
                for (iD = 1; iD <= nD; iD++)
                {
                    arr1(iD) = S(iD) / SRef(iD);
                }
                WorstPerf = ArrayMinSP(arr1, nD);
                if (WorstPerf > CallBarrier(iObs))
                {
                    EarlyPayout = (AccruedBonus + 1) * FPTYPE(exp)(-discount);
                    Redeemed = 1;
                }
            }
            /* Update time variables  */
            /* Update time */
            t += dt;
            /* Update local loop counter */
            n++;
            /* Discrete event index updates. */
            if (DEtable(n, 2) == 1)
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
            if (ArraySumSP(KI, nD) >= KINum)
            {
                for (iD = 1; iD <= nD; iD++)
                {
                    arr(iD) = S(iD) / SRef(iD);
                }
                payoff = ArrayMinSP(arr, nD) * Notional;
            }
            else
            {
                payoff = (MatCoup + 1) * Notional;
            }
        }
        dpayoffSum[tid] += payoff;
    }
}

} // namespace mcgpuSVEqLinkStructNote1Q
