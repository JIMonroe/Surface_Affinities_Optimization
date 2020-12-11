#!/usr/bin/env python

#Super simple script to just calculate free energies given files with columns 3-end as potential energies

import sys, os
import numpy as np
from pymbar import mbar
from pymbar import timeseries
import pickle

args = sys.argv[1:]

fName = args[0]
eqTime = int(args[1])
kB = float(args[2])
T = float(args[3])

kBT = kB*T

rawdat = np.loadtxt(fName)
lstates = rawdat[eqTime:,2]
Ukn = rawdat[eqTime:,3:-1]
PVdat = rawdat[eqTime:,-1]

#print(Ukn.shape)

gl = timeseries.statisticalInefficiency(lstates)
#print("Correlation time for lambda state: %f"%gl)

Nsamps = np.zeros(Ukn.shape[1])
for i in range(Ukn.shape[1]):
  gU = timeseries.statisticalInefficiency(Ukn[:,i])
  print("Correlation time if using state %i: %f"%(i, gU))
  Nsamps[i] = np.sum((lstates==i))
  print(Nsamps[i])
  Ukn[:,i] += PVdat

#Don't need to order them - it just makes things more complicated when computing averages
#neworder = np.argsort(lstates)
#Ukn = Ukn[neworder]

Ukn /= kBT

mbarObj = mbar.MBAR(Ukn.T, Nsamps)
dG, dGerr, thetaStuff = mbarObj.getFreeEnergyDifferences()

print(dG[0])
print(dGerr[0])

print("dGsolv = %f"%(-1.0*dG[0][-1]))
print("dGsolvError = %f"%(dGerr[0][-1]))

with open('mbar_object.pkl', 'w') as outfile:
  pickle.dump(mbarObj, outfile)

np.savetxt('alchemical_U.txt', Ukn, header='Potential energies at all interaction states (kBT)')


