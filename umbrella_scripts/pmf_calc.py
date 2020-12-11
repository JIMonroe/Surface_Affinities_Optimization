#!/usr/bin/env python

import sys, os
import glob
import numpy as np
import pickle
from scipy.integrate import simps
from pymbar import timeseries
from pymbar import mbar
import matplotlib
try:
  os.environ["DISPLAY"]
except KeyError:
  showPlots = False
  matplotlib.use('Agg')
import matplotlib.pyplot as plt

kB = 8.314459848E-03 #kJ/mol*K for Gromacs
T = 298.15 #K
beta = 1.0 / (kB*T)

#Calculates a pmf given a set of directories containing files with energies,
#distances along a reaction coordinate, and references for said distances for each umbrella

#Builds matrix of potential energies and passes to MBAR to get unweighted pmf

umbPrefix = sys.argv[1]
umbDirs = glob.glob('%s*'%umbPrefix)
umbNums = [int(a.split('a')[1]) for a in umbDirs]
sortUmbs = np.argsort(umbNums)
umbDirs = [umbDirs[x] for x in sortUmbs]

print umbDirs

springConst = float(sys.argv[2])

#Create array to hold all energies without umbrella restraints
allU = np.array([])

#Also need array for all solute-surface distances
allDist = np.array([])

#Finally, also need all reference distances for each umbrella
allRef = np.zeros(len(umbDirs))

#And number of independent samples for each state
numSamples = np.zeros(len(umbDirs), dtype=int)

#And make plot of distance histograms to check overlap as go
histFig, histAx = plt.subplots(1)
#histAx.set_prop_cycle('color', [plt.cm.spectral(i) for i in range(len(allRef))])

#And decomposed energies to compute perturbed PMFs
#First column will be fully coupled, then no electrostatics, then just WCA, then decoupled
allDecompU = np.array([[]]*4).T

#Loop over directories
for i, adir in enumerate(umbDirs):
  
  #MUST have two files named correctly for this to work!
  #Can add ability to provide prefixes later to make general
  #thisEDat = np.loadtxt('%s/energy.xvg'%adir)
  #thisDistDat = np.loadtxt('%s/prod_pullx.xvg'%adir)
  thisEDat = np.loadtxt('%s/prod_out.txt'%adir)
  thisDistDat = np.loadtxt('%s/prod_restraint.txt'%adir)
  
  #allRef[i] = thisDistDat[0, 2]
  allRef[i] = thisDistDat[0, 1]

  #Need to adjust so have energies with same frequency as distances
  #Should also have restraint energies in file so can subtract
  #thisEnergy = thisEDat[::2,2] - thisEDat[::2,1] 
  #thisDist = thisDistDat[:,1]
  thisEnergy = thisEDat[:,2] - thisDistDat[:,3]
  thisDist = thisDistDat[:,2]

  #Only take uncorrelated samples...
  uncorrinds = timeseries.subsampleCorrelatedData(thisEnergy)
  #uncorrinds = np.arange(len(thisEnergy))
  numSamples[i] = len(uncorrinds)

  print "For %s, have %i independent samples." % (adir, len(uncorrinds))
    
  allU = np.hstack((allU, thisEnergy[uncorrinds]))
  allDist = np.hstack((allDist, thisDist[uncorrinds]))

  #Plot histogram with uncorrelated indices
  thisHist, thisBins = np.histogram(thisDist[uncorrinds], bins='auto', density=False)
  binCents = 0.5*(thisBins[:-1] + thisBins[1:])
  histAx.plot(binCents, thisHist)

  #Load potential energies decomposed into various components and pull out same indices as from total E
  decompDat = np.loadtxt('%s/pot_energy_decomp.txt'%adir)
  allDecompU = np.vstack((allDecompU, decompDat[uncorrinds, :]))

#Finish up and save histogram figure
histAx.set_xlabel(r'Surface-Solute distance ($nm$)')
histAx.set_ylabel(r'Histogram counts')
histFig.savefig('distance_hists.png')

#Now need to construct matrix
#For each umbrella, need energy of all sampled configurations in that umbrella
Umat = np.zeros((len(umbDirs), len(allU)))

#Make potential energies dimensionless
allU *= beta

#Set reference for potential energies
#allU -= np.min(allU)

for i, aref in enumerate(allRef):
  
  Umat[i,:] = allU + beta*((0.5*springConst) * ((allDist - aref)**2))

#And use mbar to get pmf!
mbarobj = mbar.MBAR(Umat, numSamples)

deltaGs, deltaGerr, thetaStuff = mbarobj.getFreeEnergyDifferences()

print "\nFree energies between states:"
print deltaGs[0]
print "\nwith uncertainties:"
print deltaGerr[0]

#Now also calculate pmf
binsize = 0.05 #nm
pmfBins = np.arange(np.min(allDist)-(1E-06), np.max(allDist), binsize)
pmfBinInds = np.digitize(allDist, pmfBins) - 1
pmfBinCents = 0.5*(pmfBins[:-1] + pmfBins[1:])
nBins = len(pmfBinCents)

pmfVals, pmfErr = mbarobj.computePMF(allU, pmfBinInds, nBins)

print ''
print pmfVals
print pmfErr

#Also compute PMFs in other ensembles, such as no electrostatics, WCA, decoupled, etc.
#Can accomplish by plugging in potential energies for each state of interest
pmfTot, pmfTotErr = mbarobj.computePMF(allDecompU[:,0], pmfBinInds, nBins)
pmfLJ, pmfLJErr = mbarobj.computePMF(allDecompU[:,1], pmfBinInds, nBins)
pmfWCA, pmfWCAErr = mbarobj.computePMF(allDecompU[:,2], pmfBinInds, nBins)
pmfDecoupled, pmfDecoupledErr = mbarobj.computePMF(allDecompU[:,3], pmfBinInds, nBins)

#Quickly plot and save pmf
pmfFig, pmfAx = plt.subplots(1)
pmfAx.errorbar(pmfBinCents, pmfVals, yerr=pmfErr, fmt='-', linewidth=1.0, 
               elinewidth=1.0, capsize=1.5, capthick=0.5)
pmfAx.set_xlabel(r'Surface-Solute distance ($nm$)')
pmfAx.set_ylabel(r'Free energy ($k_{B}T$)')
pmfFig.savefig('pmf.png')

#Saving PMF in fully coupled ensemble and PMFs based on perturbations
np.savetxt('pmf.txt', np.vstack((pmfBinCents, pmfVals, pmfErr, pmfTot, pmfTotErr, pmfLJ, 
                                 pmfLJErr, pmfWCA, pmfWCAErr, pmfDecoupled, pmfDecoupledErr)).T, 
           header='#Distance    PMF     PMFerr     PMF fully coupled (and error)     PMF_LJ (and error)     PMF_WCA (and error)     PMF_decoupled (and error)')

#Also compute potential energies as a function of distance
#Do this by following computePMF method in pymbar
log_w_n = mbarobj._computeUnnormalizedLogWeights(allU)
maxLogW = np.max(log_w_n)
pmfUvals = np.zeros(nBins)
#AND compute various decompositions of the potential energy as a function of distance
#Will do both in only the fully coupled ensemble - can use the total solute-solvent energy to back out Srel
#First column will be total difference (coupled-decoupled)
#Next column will be electrostatic (coupled-noQ)
#Next will be attractive LJ part (noQ-WCA)
#And last will have WCA part (WCA-decoupled)
pmfDecompUvals = np.zeros((nBins, 4))
for i in range(nBins):
  thisInds = np.where(pmfBinInds == i)
  thisWeight = np.sum(np.exp(log_w_n[thisInds] - maxLogW))
  pmfUvals[i] = np.sum(allU[thisInds]*np.exp(log_w_n[thisInds] - maxLogW)) / thisWeight
  pmfDecompUvals[i,0] = np.sum((allDecompU[thisInds,0] - allDecompU[thisInds,3])*np.exp(log_w_n[thisInds] - maxLogW)) / thisWeight
  pmfDecompUvals[i,1] = np.sum((allDecompU[thisInds,0] - allDecompU[thisInds,1])*np.exp(log_w_n[thisInds] - maxLogW)) / thisWeight
  pmfDecompUvals[i,2] = np.sum((allDecompU[thisInds,1] - allDecompU[thisInds,2])*np.exp(log_w_n[thisInds] - maxLogW)) / thisWeight
  pmfDecompUvals[i,3] = np.sum((allDecompU[thisInds,2] - allDecompU[thisInds,3])*np.exp(log_w_n[thisInds] - maxLogW)) / thisWeight

#Feel like should not shift potential energies - can do when plot/present data if want
#pmfUvals -= pmfUvals[np.where(pmfVals == 0.0)[0]]
print ''
print pmfUvals

#for i in range(pmfDecompUvals.shape[1]):
#  pmfDecompUvals[:,i] -= pmfDecompUvals[np.where(pmfVals == 0.0)[0]]

np.savetxt('pmf_U.txt', np.vstack((pmfBinCents, pmfUvals, pmfDecompUvals.T)).T, 
           header='#Distance    Average potential energy     <U_tot>     <U_Q>     <U_LJ_attr>     <U_WCA>')

#Compute binding affinity to surface at various distances from surface
#Where plot levels out, can use this as Keq
KeqVals = np.zeros(nBins-2)
for i in range(2, nBins):
  KeqVals[i-2] = simps(np.exp(-pmfVals[:i]), pmfBinCents[:i])

#Plot Keq versus cut-off distance as well as dGbind associated with each Keq
bindFig, bindAx = plt.subplots(2, sharex=True)
bindAx[0].plot(pmfBinCents[2:], KeqVals, linewidth=4.0)
bindAx[1].plot(pmfBinCents[2:], np.log(KeqVals), linewidth=4.0) 

bindAx[0].tick_params(labelbottom=False)
bindAx[1].set_xlabel(r'Cutoff distance from interface ($nm$)')
bindAx[0].set_ylabel(r'$K_{bind}$')
bindAx[1].set_ylabel(r'$\Delta G_{bind}$ ($k_{B}T$)')
bindFig.tight_layout()
bindFig.subplots_adjust(hspace=0)
bindFig.savefig('Kbind_dGbind.png')

#Finally, pickle the MBAR object so we can re-evaluate free energies, etc.
with open('mbar_object.pkl', 'w') as outfile:
  pickle.dump(mbarobj, outfile)

#And save potential energies with no umbrella restraints (so don't have to recompute)
np.savetxt('U_noRestraint.txt', allU, header='Potential energies for each configuration without restraints (kBT)')


