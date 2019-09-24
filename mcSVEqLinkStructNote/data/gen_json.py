import numpy as np
from json_array_pretty import *

Notional = 10000.0
nD = 6
r = 0.05

sigma = 0.25*np.ones(nD)
q = 0.0*np.ones(nD)
Spot = 100.*np.ones(nD)
SRef = Spot
KI0 = 0*Spot

theta = sigma*sigma
kappa = 2. + 0*theta
vSpot = theta
sigma = 1*np.sqrt(2*kappa*theta)

ObsDates = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
CallBarrier = [0.95, 0.95, 0.9, 0.9, 0.85, 0.85]

nObs = len(ObsDates)
TMax = ObsDates[nObs-1]
PrevObsDate = 0

AccruedBonus0 = 0
BonusCoup = 0.15
MatCoup = 0.15
KIBarrier = 0.65
KINum = 3

# correlation matrix blocks
rho = np.ones((nD,nD))*0.5
np.fill_diagonal(rho,1.0)
rhoSv = np.eye(nD)*(-0.7)
rhovv = np.eye(nD)

ZeroDates = np.arange(0.0, 5.5, 0.5)
ZeroRates = 0.05 + 0*ZeroDates

input_data = {
    'Notional'      : Notional,
    'r'             : r,
    'TMax'          : TMax,
    'AccruedBonus0' : AccruedBonus0,
    'BonusCoup'     : BonusCoup,
    'MatCoup'       : MatCoup,
    'KIBarrier'     : KIBarrier,
    'KINum'         : KINum
}

input_data["Stocks"] = {
    'Spot' : Spot.tolist(),
    'SRef' : SRef.tolist(),
    'q'    : q.tolist(),
    'kappa': kappa.tolist(),
    'theta': theta.tolist(),
    'vSpot': vSpot.tolist(),
    'sigma': sigma.tolist(),
    'KI0'   : KI0.tolist()
}

input_data["Call Barrier"] = {
    'ObsDates'    : ObsDates,
    'CallBarrier' : CallBarrier,
    'PrevObsDate'   : PrevObsDate
}

input_data["Zero Curve"] = {
    'ZeroDates' : ZeroDates.tolist(),
    'ZeroRates' : ZeroRates.tolist(),
}

input_data["Correlation Matrix"] = {
    'rho': [NoIndent(elem) for elem in rho.tolist()],
    'rhoSv': [NoIndent(elem) for elem in rhoSv.tolist()],
    'rhovv': [NoIndent(elem) for elem in rhovv.tolist()]
}

print(json.dumps(input_data, cls=MyEncoder, indent=4))

# test custom JSONEncoder with json.dump()
with open('input_data.json', 'w') as fp:
    json.dump(input_data, fp, cls=MyEncoder, indent=4)
    fp.write('\n')
