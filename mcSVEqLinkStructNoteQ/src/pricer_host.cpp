#include <vector>

#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>

#include <sci/num_utils.h>

#include "pricer_kernel_wrapper.h"

#include "pricer_host.h"

void mcSVEqLinkStructNote::calc(
    int pmax,
    int seq,
    int nsub,
    double& Vx,
    double& devx,
    double& tk)
{
    int miMax;
    double TMax;

    int nObs = ObsDates.size();
    int nD = Spot.size();

    std::vector<double> DeltaT(nObs + 1);
    DeltaT[1] = ObsDates[0] - PrevObsDate;
    for (int i = 1; i < nObs; i++)
    {
        DeltaT[i + 1] = ObsDates[i] - ObsDates[i - 1];
    }

    miMax = nObs * nsub;
    std::vector<int[3]> DEtable(miMax + 1);
    std::vector<double> DEtimeTable(miMax + 1, 0);

    TMax = ObsDates[nObs - 1];
    double dt_sub = TMax / (nObs * nsub);

    for (int n = 0; n <= miMax; n++)
    {
        DEtimeTable[n] = n * dt_sub;
        DEtable[n][0] = 0;
        DEtable[n][1] = 1;
        if (n > 0 && n % nsub == 0)
            DEtable[n][2] = 1;
    }

    std::vector<double> r(miMax + 1);

    /* Computing rho from equation Eq8c; formula is stubroutinevar1 ==
     * getCorrelation[rho, RhoSS, RhoSv, Rhovv, nD, nD].
     */
    std::vector<std::vector<double>> rho;
    sci::getCorrelation(rho, RhoSS, RhoSv, Rhovv);
    /* Computing Mcor from equation Eq1c; formula is Mcor == Choldecomp[rho,
     * Mcor]. */
    std::vector<std::vector<double>> Mcor;
    sci::CholDecomp(rho, Mcor);

    for (int n = 0; n <= miMax; n++)
    {
        r[n] = sci::LinearInterpolationUniform(
            ZeroDates, ZeroRates, DEtimeTable[n], dt_sub);
    }

    pricer_kernel_wrapper(
        DeltaT,
        r,
        DEtable,
        DEtimeTable,
        Mcor,
        AccruedBonus0,
        BonusCoup,
        CallBarrier,
        kappa,
        KI0,
        KIBarrier,
        KINum,
        MatCoup,
        Notional,
        pmax,
        q,
        seq,
        sigma,
        Spot,
        SRef,
        theta,
        vSpot,
        devx,
        Vx,
        tk);
}

void mcSVEqLinkStructNote::load_inputs()
{
    nlohmann::json j;
    // Read input JSON file specified via environment variable SCI_DATAFILE
    std::string input_file_path("input_data.json");
    const char* input_data_var = getenv("SCI_DATAFILE");
    if (input_data_var)
    {
        input_file_path = input_data_var;
    }
    std::ifstream input(input_file_path);
    if (!input.is_open())
    {
        throw std::runtime_error(
            std::string(input_file_path) + " is not found");
    }
    input >> j;

    AccruedBonus0 = j["AccruedBonus0"];
    BonusCoup = j["BonusCoup"];
    KIBarrier = j["KIBarrier"];
    KINum = j["KINum"];
    MatCoup = j["MatCoup"];
    Notional = j["Notional"];

    PrevObsDate = j["Call Barrier"]["PrevObsDate"];

    std::vector<double> ObsDates_ = j["Call Barrier"]["ObsDates"];
    ObsDates = ObsDates_;
    std::vector<double> CallBarrier_ = j["Call Barrier"]["CallBarrier"];
    CallBarrier = CallBarrier_;
    if (ObsDates.size() != CallBarrier.size())
        throw std::runtime_error("Call Barrier data mismatch");

    std::vector<std::vector<double>> RhoSS_ = j["Correlation Matrix"]["rho"];
    RhoSS = RhoSS_;
    std::vector<std::vector<double>> RhoSv_ = j["Correlation Matrix"]["rhoSv"];
    RhoSv = RhoSv_;
    std::vector<std::vector<double>> Rhovv_ = j["Correlation Matrix"]["rhovv"];
    Rhovv = Rhovv_;

    std::vector<double> Spot_ = j["Stocks"]["Spot"];
    Spot = Spot_;
    std::vector<double> SRef_ = j["Stocks"]["SRef"];
    SRef = SRef_;
    std::vector<double> q_ = j["Stocks"]["q"];
    q = q_;
    std::vector<double> kappa_ = j["Stocks"]["kappa"];
    kappa = kappa_;
    std::vector<double> theta_ = j["Stocks"]["theta"];
    theta = theta_;
    std::vector<double> vSpot_ = j["Stocks"]["vSpot"];
    vSpot = vSpot_;
    std::vector<double> sigma_ = j["Stocks"]["sigma"];
    sigma = sigma_;
    std::vector<double> KI0_ = j["Stocks"]["KI0"];
    KI0 = KI0_;

    std::vector<double> ZeroDates_ = j["Zero Curve"]["ZeroDates"];
    ZeroDates = ZeroDates_;
    std::vector<double> ZeroRates_ = j["Zero Curve"]["ZeroRates"];
    ZeroRates = ZeroRates_;
    if (ZeroDates.size() != ZeroRates.size())
        throw std::runtime_error("Call Barrier data mismatch");
}
