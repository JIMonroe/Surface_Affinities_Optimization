#!/usr/bin/env python

#THIS IS AN OLDER VERSION OF THIS CODE! EITHER USE calcdGsolv IN genetic_lib OR WRAPPER FOR THAT IN check_dGsolv.py

#Script to calculate dG for a solute near a surface
#Requires the full trajectory, actually, because need solute coordinates to get rid of biases
#This will all be hard-coded to match my file naming conventions
#Can generalize later if need be, but make sure to change spring constants etc. if change in other scripts
#Also, should eventually make this a function in a library-ish file so can use more easily with genetic algorithm

import sys, os
import copy
import numpy as np
import parmed as pmd
import pytraj as pt
from pymbar import mbar
from pymbar import timeseries
import waterlib as wl
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

numStates = 19

args = sys.argv[1:]

Usage="""calcdG_surface.py
\tThis script runs a free energy calculation for inserting a particle on a surface using simulations with different restraints.
\tTo run this script, must provide the following information for each simulation.
\t\t1) Directory housing the sol_surf.top, prod.nc, and alchemical_output.txt files
\t\t2) Spring constant for the x-y position restraint and the reference x and y positions and reference distances
\t\t3) Optionally, the number of frames to read, i.e. the "end time" or time to stop at. -1 is until end.
\tSo the command line input is...
\t\tpython calcdG_surface.py simdir1 kxy1 refx1 refy1 distx1 disty1 simdir2 kxy2 refx2 refy2 distx2 disty2... endTime
\tSeems like a lot, but provides flexibility, and can hard code if really want to.
\tNote that the spring constants and reference distances, etc. should be given in kJ/mol*A^2 and Angstroms, respectively.
\tThis is because by default the NetCDF trajectory writer uses Angstroms and so does pytraj.
"""

#First make sure we have the right number of inputs provided
if (len(args)-1)%6 != 0:
  print('Wrong number of inputs provided.')
  print(Usage)
  sys.exit(2)

simDirs = args[0:-1:6]
kXY = [float(x) for x in args[1:-1:6]]
refX = [float(x) for x in args[2:-1:6]]
refY = [float(x) for x in args[3:-1:6]]
distRefX = [float(x) for x in args[4:-1:6]]
distRefY = [float(x) for x in args[5:-1:6]]
try:
  endTime = int(args[-1])
except IndexError:
  endTime = -1

#print(args)
#print(simDirs)
#print(kXY)
#print(refX)
#print(refY)
#print(distRefX)
#print(distRefY)
#print(endTime)

#Want to loop over all trajectories provided, storing solute position information to calculate restraints
xyPos = None #X and Y coordinates of first heavy atom for all solutes - will get shape after find solutes
wrapxyPos = None #Wrapped X and Y coordinates of solutes relative to reference on surface
nSamps = np.zeros((len(simDirs), numStates)) #Have as many x-y restraints as sims and then the same number of lambda states for each 
allPots = np.array([[]]*numStates).T #Potential energies, EXCLUDING RESTRAINT, for each simulation frame and lambda state
xyBox = np.zeros(2)

for i, adir in enumerate(simDirs):

  topFile = "%s/../sol_surf.top"%adir
  trajFile = "%s/prod.nc"%adir
  alchemicalFile = "%s/alchemical_output.txt"%adir

  #First load in topology and get atom indices 
  top = pmd.load_file(topFile)

  #Get solute heavy atoms for each solute
  #Also get indices of surface atoms to use as references later
  #Only taking last united atoms of the first SAM molecule we find
  heavyIndices = []
  cuIndices = []
  for res in top.residues:
    if res.name not in ['OTM', 'CTM', 'SOL']:
      thisheavyinds = []
      for atom in res.atoms:
        if not atom.name[0] == 'H':
          thisheavyinds.append(atom.idx)
      heavyIndices.append(thisheavyinds)
    else:
      if len(cuIndices) == 0:
        for atom in res.atoms:
          if atom.name in ['U10', 'U20']:
            cuIndices.append(atom.idx)

  #Make into arrays for easier referencing
  heavyIndices = np.array(heavyIndices)
  cuIndices = np.array(cuIndices)

  #Load in the potential energies, INCLUDING RESTRAINT, at all states for this simulation to figure out frames to skip
  alcDat = np.loadtxt(alchemicalFile)
  startTime = alcDat[0, 1]
  startFrame = int(startTime) - 1 #Be careful here... need write frequency in alchemical file to match exactly with positions
                                  #AND assuming that have written in 1 ps increments... 
                                  #Also, first frame in trajectory is NOT at time zero, so subtract 1
  if endTime == -1:
    thisPot = alcDat[:, 3:-1]
  else:
    thisPot = alcDat[:endTime, 3:-1]
  thisg = timeseries.statisticalInefficiencyMultiple(thisPot)
  print("Statistical inefficiency for this set of potential energies: %f"%thisg)

  #print(startTime)
  #print(startFrame)
  #print(thisPot.shape)

  #Next load in the trajectory and get all solute coordinates that matter
  top.rb_torsions = pmd.TrackedList([])
  top = pt.load_parmed(top, traj=False)
  if endTime == -1:
    traj = pt.iterload(trajFile, top, frame_slice=(startFrame, -1))
  else:
    traj = pt.iterload(trajFile, top, frame_slice=(startFrame, startFrame+endTime))
  nFrames = len(traj)
  xyBox = np.array(traj[0].box.values)[:2] #A little lazy, but all boxes should be same and fixed in X and Y dimensions

  #print(nFrames)

  #To correctly map solute positions on both sides of surface, need fixed reference on each surface face
  #Will take first SAM residue and use its furthest out united atoms on each face
  #Will have one reference for each solute, with the chosen reference being the closest atom by z distance
  refIndices = [0]*len(heavyIndices)
  for k in range(len(heavyIndices)):
    solZpos = traj[0][heavyIndices[k][0], 2]
    cuZpos = traj[0][cuIndices, 2] 
    zdists = abs(solZpos - cuZpos)
    closeind = np.argmin(zdists)
    refIndices[k] = cuIndices[closeind]
  refIndices = np.array(refIndices)

  thisxyPos = np.zeros((nFrames, len(heavyIndices), 2))
  thiswrapxyPos = np.zeros((nFrames, len(heavyIndices), 2))
  thisnSamps = np.zeros(numStates)

  #Reference x and y coordinates for this restraint
  thisRefXY = np.array([refX[i], refY[i]])

  for j, frame in enumerate(traj):

    thisPos = np.array(frame.xyz)
    thisXY = thisPos[heavyIndices[:,0]][:, :2] #Takes XY coords for first heavy atom from each solute
    thisxyPos[j] = thisXY
    thisnSamps[int(alcDat[j, 2])] += 1 #Lambda states must be indexed starting at 0

    #Also get wrapped positions relative to each reference face
    #AND calculate xy restraint energy to remove by adding this for each solute
    xyEnergy = 0.0
    for k in range(len(heavyIndices)):
      xy = thisXY[k]
      #First reimage around the reference surface atom for this solute
      cuxyPos = thisPos[refIndices[k]][:2]
      thiswrapxyPos[j, k, :] = wl.reimage([xy], cuxyPos, xyBox)[0] - cuxyPos
      #Then separately reimage around the restraint reference positions to calculate energy
      xy = wl.reimage([xy], thisRefXY, xyBox)[0] - thisRefXY
      xyEnergy += (  0.5*kXY[i]*(0.5*(np.sign(xy[0] - distRefX[i]) + 1))*((xy[0] - distRefX[i])**2)
                   + 0.5*kXY[i]*(0.5*(np.sign(xy[1] - distRefY[i]) + 1))*((xy[1] - distRefY[i])**2) )

    #Remove the restraint energy (only for x-y restraint... z is the same in all simulations)
    thisPot[j,:] -= (xyEnergy / kBT)

  #Add to other things we're keeping track of
  if xyPos is None:
    xyPos = copy.deepcopy(thisxyPos)
    wrapxyPos = copy.deepcopy(thiswrapxyPos)
  else:
    xyPos = np.vstack((xyPos, thisxyPos))
    wrapxyPos = np.vstack((wrapxyPos, thiswrapxyPos))
  nSamps[i,:] = thisnSamps
  allPots = np.vstack((allPots, thisPot))

  #if showPlots:
  #  plt.plot(thisxyPos[:,0], label='X')
  #  plt.plot(thisxyPos[:,1], label='Y')
  #  plt.legend()
  #  plt.ylabel(r'Coordinate value')
  #  plt.show()
  #
  #  thiswrapxy = wl.reimage(thisxyPos, thisRefXY, xyBox) - thisRefXY
  #  plt.plot(thiswrapxy[:,0] - distRefX[i], label='X')
  #  plt.plot(thiswrapxy[:,1] - distRefY[i], label='Y')
  #  plt.legend()
  #  plt.ylabel(r'Minimum image distance')
  #  plt.show()

#Now should have all the information we need
#Next, put it into the format that MBAR wants, adding energies as needed
Ukn = np.zeros((len(simDirs)*numStates, int(np.sum(nSamps))))

for i in range(len(simDirs)):

  #First get energy of ith type of x-y restraint for all x-y positions
  thisRefXY = np.array([refX[i], refY[i]])
  #Must do by looping over each solute
  xyEnergy = np.zeros(xyPos.shape[0])
  for k in range(len(heavyIndices)):
    xy = wl.reimage(xyPos[:,k,:], thisRefXY, xyBox) - thisRefXY
    xyEnergy += (  0.5*kXY[i]*(0.5*(np.sign(xy[:,0] - distRefX[i]) + 1))*((xy[:,0] - distRefX[i])**2)
                 + 0.5*kXY[i]*(0.5*(np.sign(xy[:,1] - distRefY[i]) + 1))*((xy[:,1] - distRefY[i])**2) )
  
  #Loop over alchemical states with this restraint and add energy
  for j in range(numStates):
    
    Ukn[i*numStates+j, :] = allPots[:,j] + (xyEnergy / kBT)

#Print some info about numbers of samples and effective numbers of samples  
print('\n')
print(np.sum(nSamps, axis=1))
print(nSamps.tolist())
print('  ')

#Now should be set to run MBAR
#Note we ignore the pV term, as it won't matter here because all states have same V for given configuration
#Also note that we haven't actually sampled the states we're interested in (i.e. unrestrained)
#So we have to reweight, or us perturbation to get the free energies we're interested in
mbarObj = mbar.MBAR(Ukn, nSamps.flatten())
dG, dGerr = mbarObj.computePerturbedFreeEnergies(allPots.T)
print(dG[0].tolist())
print('\n')

#Would also be nice to generate 2D PMF based on x and y coordinates
#To do this, can actually use computePMF from mbar, but need to map back into 2D
#Actually, have to do for each solute separately, then combine via Boltzmann weights of bins

#First define bins
xWidth = xyBox[0] / 10.0
yWidth = xyBox[1] / 10.0
xBins = np.arange(-0.5*xyBox[0], 0.5*xyBox[0]+0.01, xWidth)
yBins = np.arange(-0.5*xyBox[1], 0.5*xyBox[1]+0.01, yWidth)

#And initiate PMFs and unweighted histograms
xyPMF0 = np.zeros((len(xBins)-1, len(yBins)-1))
xyPMF18 = np.zeros((len(xBins)-1, len(yBins)-1))
xyHist = np.zeros((len(xBins)-1, len(yBins)-1))

#Now loop over the solutes
for k in range(len(heavyIndices)):

  #And get X and Y bins for each sample
  #Remember, already wrapped around the last united atom on each side of the first SAM chain as reference points
  xInds = np.digitize(wrapxyPos[:,k,0], xBins) - 1
  yInds = np.digitize(wrapxyPos[:,k,1], yBins) - 1

  #Now need to map our 2D bins to 1D bins for mbar
  #Keeping mapping simple, i.e. 1Dind = xInd*numberYbins + yInd
  #To map back, xInd = 1Dind / numberYbins and yInd = 1Dind % numberYbins
  #Will also histogram in 2D as we do this
  mapInds = np.zeros(wrapxyPos.shape[0])
  for i in range(wrapxyPos.shape[0]):
    mapInds[i] = xInds[i]*(len(yBins)-1) + yInds[i]
    xyHist[xInds[i], yInds[i]] += 1

  #Now compute the PMF with mbar for both coupled and decoupled states
  thisxyPMF0map, thisxyPMF0mapErr = mbarObj.computePMF(allPots[:,0], mapInds, (len(xBins)-1)*(len(yBins)-1))
  thisxyPMF18map, thisxyPMF18mapErr = mbarObj.computePMF(allPots[:,-1], mapInds, (len(xBins)-1)*(len(yBins)-1))

  #Now need to map back to 2D arrays
  thisxyPMF0 = np.zeros((len(xBins)-1, len(yBins)-1))
  thisxyPMF18 = np.zeros((len(xBins)-1, len(yBins)-1))
  for i in range((len(xBins)-1)*(len(yBins)-1)):
    thisXind = int(i / (len(yBins)-1))
    thisYind = i % (len(yBins)-1)
    thisxyPMF0[thisXind, thisYind] = thisxyPMF0map[i]
    thisxyPMF18[thisXind, thisYind] = thisxyPMF18map[i]

  #Add to Boltzmann sum for combining across solutes
  xyPMF0 += np.exp(-thisxyPMF0)
  xyPMF18 += np.exp(-thisxyPMF18)

#Now take logs, reference to PMF mins and show plots!
xyPMF0 = -np.log(xyPMF0)
xyPMF18 = -np.log(xyPMF18)
xyPMFsolv = xyPMF0 - xyPMF18 #Free energy difference to SOLVATE
xyPMF0 -= np.min(xyPMF0)
xyPMF18 -= np.min(xyPMF18)
xyPMFsolv -= np.min(xyPMFsolv)

heatFig, heatAx = plt.subplots(4)
heatHist = heatAx[0].imshow(xyHist, cmap=plt.get_cmap('magma'), aspect='auto')
heatPMF0 = heatAx[1].imshow(xyPMF0, cmap=plt.get_cmap('magma'), aspect='auto')
heatPMF18 = heatAx[2].imshow(xyPMF18, cmap=plt.get_cmap('magma'), aspect='auto')
heatPMFsolv = heatAx[3].imshow(xyPMFsolv, cmap=plt.get_cmap('magma'), aspect='auto')
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
xyAx.plot(wrapxyPos[:,0], 'o', label='X')
xyAx.plot(wrapxyPos[:,1], 'o', label='Y')
xyAx.legend()
xyAx.set_ylabel(r'Coordinate value')

if showPlots:
  plt.show()



