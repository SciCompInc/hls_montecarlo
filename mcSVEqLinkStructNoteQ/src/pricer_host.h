#pragma once

#include <vector>

struct mcSVEqLinkStructNote {

    void load_inputs();

    void
    calc(int pmax, int seq, int nsub, double& Vx, double& devx, double& tk);

    // input data
    double AccruedBonus0;
    double BonusCoup;
    std::vector<double> CallBarrier;
    std::vector<double> kappa;
    std::vector<double> KI0;
    double KIBarrier;
    double KINum;
    double MatCoup;
    double Notional;
    std::vector<double> ObsDates;
    double PrevObsDate;
    std::vector<double> q;
    std::vector<std::vector<double>> RhoSS;
    std::vector<std::vector<double>> RhoSv;
    std::vector<std::vector<double>> Rhovv;
    std::vector<double> sigma;
    std::vector<double> Spot;
    std::vector<double> SRef;
    std::vector<double> theta;
    std::vector<double> vSpot;
    std::vector<double> ZeroDates;
    std::vector<double> ZeroRates;
};
