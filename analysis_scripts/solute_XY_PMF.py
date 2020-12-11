#!/usr/bin/env python

import sys, os
import numpy as np
import waterlib as wl
import pickle
import matplotlib
showPlots = True
try:
  os.environ["DISPLAY"]
except KeyError:
  showPlots = False
  matplotlib.use('Agg')
import matplotlib.pyplot as plt

kB = 0.008314459848
T = 298.15
kBT = kB*T
beta = 1.0 / kBT

mbarObjFile = 'mbar_object.pkl'
energyFile = 'alchemical_U_noXYres.txt'
xyFile = 'solute_0_XY.txt'

#Load in files
with open(mbarObjFile, 'r') as infile:
  mbarObj = pickle.load(infile)

allPots = np.loadtxt(energyFile)
xyDat = np.loadtxt(xyFile)

#Define box dimensions
xyBox = np.array([29.8200, 34.4332]) #Hard coded... but won't change unless change system working with

xyPos = xyDat[:,:2] #Raw XY is first two columns, wrapped is last two
xyPos = wl.reimage(xyPos, np.array([0.0, 0.0]), xyBox) #wrap back INTO the box if not raw
#xyPos = xyDat[:,-2:] #Raw XY is first two columns, wrapped is last two

#Would also be nice to generate 2D PMF based on x and y coordinates
#To do this, can actually use computePMF from mbar, but need to map back into 2D

#First define bins
xWidth = xyBox[0] / 10.0
yWidth = xyBox[1] / 12.0
xBins = np.arange(-0.5*xyBox[0], 0.5*xyBox[0]+0.001, xWidth)
yBins = np.arange(-0.5*xyBox[1], 0.5*xyBox[1]+0.001, yWidth)

#And initiate PMFs and unweighted histograms
xyPMF0 = np.zeros((len(xBins)-1, len(yBins)-1))
xyPMF18 = np.zeros((len(xBins)-1, len(yBins)-1))
xyHist = np.zeros((len(xBins)-1, len(yBins)-1))

#And get X and Y bins for each sample
#Remember, already wrapped around the last united atom on each side of the first SAM chain as reference points
xInds = np.digitize(xyPos[:,0], xBins) - 1
yInds = np.digitize(xyPos[:,1], yBins) - 1

#Now need to map our 2D bins to 1D bins for mbar
#Keeping mapping simple, i.e. 1Dind = xInd*numberYbins + yInd
#To map back, xInd = 1Dind / numberYbins and yInd = 1Dind % numberYbins
#Will also histogram in 2D as we do this
mapInds = np.zeros(xyPos.shape[0])
for i in range(xyPos.shape[0]):
  mapInds[i] = xInds[i]*(len(yBins)-1) + yInds[i]
  xyHist[xInds[i], yInds[i]] += 1

#Now compute the PMF with mbar for both coupled and decoupled states
thisxyPMF0map, thisxyPMF0mapErr = mbarObj.computePMF(allPots[:,0], mapInds, (len(xBins)-1)*(len(yBins)-1))
thisxyPMF18map, thisxyPMF18mapErr = mbarObj.computePMF(allPots[:,-1], mapInds, (len(xBins)-1)*(len(yBins)-1))

#Now need to map back to 2D arrays
xyPMF0 = np.zeros((len(xBins)-1, len(yBins)-1))
xyPMF18 = np.zeros((len(xBins)-1, len(yBins)-1))
for i in range((len(xBins)-1)*(len(yBins)-1)):
  thisXind = int(i / (len(yBins)-1))
  thisYind = i % (len(yBins)-1)
  xyPMF0[thisXind, thisYind] = thisxyPMF0map[i]
  xyPMF18[thisXind, thisYind] = thisxyPMF18map[i]

#Now reference to PMF mins and show plots!
xyPMFsolv = xyPMF0 - xyPMF18 #Free energy difference to SOLVATE
#xyPMF0 -= np.min(xyPMF0)
#xyPMF18 -= np.min(xyPMF18)
#xyPMFsolv -= np.min(xyPMFsolv)

heatFig, heatAx = plt.subplots(4)
heatHist = heatAx[0].imshow(xyHist.T, cmap=plt.get_cmap('magma'), aspect=xyBox[1]/xyBox[0], origin='lower', 
                            extent=(-0.5*xyBox[0], 0.5*xyBox[0], -0.5*xyBox[1], 0.5*xyBox[1]),
                            interpolation='spline16')
heatPMF0 = heatAx[1].imshow(xyPMF0.T, cmap=plt.get_cmap('magma'), aspect=xyBox[1]/xyBox[0], origin='lower', 
                            extent=(-0.5*xyBox[0], 0.5*xyBox[0], -0.5*xyBox[1], 0.5*xyBox[1]),
                            interpolation='spline16')
heatPMF18 = heatAx[2].imshow(xyPMF18.T, cmap=plt.get_cmap('magma'), aspect=xyBox[1]/xyBox[0], origin='lower', 
                             extent=(-0.5*xyBox[0], 0.5*xyBox[0], -0.5*xyBox[1], 0.5*xyBox[1]),
                             interpolation='spline16')
heatPMFsolv = heatAx[3].imshow(xyPMFsolv.T, cmap=plt.get_cmap('magma'), aspect=xyBox[1]/xyBox[0], origin='lower', 
                               extent=(-0.5*xyBox[0], 0.5*xyBox[0], -0.5*xyBox[1], 0.5*xyBox[1]),
                               interpolation='spline16')
heatFig.subplots_adjust(right=0.8)
cbarax = heatFig.add_axes([0.85, 0.8, 0.05, 0.20])
heatFig.colorbar(heatHist, cax=cbarax)
cbarax.set_ylabel(r'Counts')
cbaraxPMF0 = heatFig.add_axes([0.85, 0.55, 0.05, 0.20])
heatFig.colorbar(heatPMF0, cax=cbaraxPMF0)
cbaraxPMF0.set_ylabel(r'$k_{B}T$')
cbaraxPMF18 = heatFig.add_axes([0.85, 0.30, 0.05, 0.20])
heatFig.colorbar(heatPMF18, cax=cbaraxPMF18)
cbaraxPMF18.set_ylabel(r'$k_{B}T$')
cbaraxPMFsolv = heatFig.add_axes([0.85, 0.05, 0.05, 0.20])
heatFig.colorbar(heatPMFsolv, cax=cbaraxPMFsolv)
cbaraxPMFsolv.set_ylabel(r'$k_{B}T$')

xyFig, xyAx = plt.subplots(1)
#xyAx.plot(xyPos[:,0], 'o', label='X')
#xyAx.plot(xyPos[:,1], 'o', label='Y')
xyAx.plot(xyPos[:,0], 'o', label='X')
xyAx.plot(xyPos[:,1], 'o', label='Y')
xyAx.legend()
xyAx.set_ylabel(r'Coordinate value')

plt.show()


