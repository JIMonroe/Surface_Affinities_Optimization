#!/usr/bin/env python

import sys, os
import numpy as np
import parmed as pmd
import pytraj as pt
import waterlib as wl
import matplotlib
showPlots = True
try:
  os.environ["DISPLAY"]
except KeyError:
  showPlots = False
  matplotlib.use('Agg')
import matplotlib.pyplot as plt

#Examines the 2D sampling of the solute near the surface
#Must specify topology and trajectory files

#Read in topology file and trajectory file
topFile = sys.argv[1]
trajFile = sys.argv[2]

#And look for file detailing which state the system is in on each step
try:
  stateInfoFile = sys.argv[3]
except IndexError:
  stateInfoFile = None

try:
  skipframes = int(sys.argv[4])
except IndexError:
  skipframes = 0

#Read in topology with parmed
top = pmd.load_file(topFile)

#Just in case have Ryckaert-Bellemans torsions (do for SAM system), remove so can convert
#Don't need this info for just generating configurations
top.rb_torsions = pmd.TrackedList([])
top = pt.load_parmed(top, traj=False)

#Read trajectory in and link to topology
traj = pt.iterload(trajFile, top, frame_slice=(skipframes,-1))

#Find number of frames 
nFrames = traj.n_frames 

#Get box dimensions
boxDims = traj.unitcells[0, 0:3]

#Get indices of solute
solInds = top.select('!(:OTM,CTM,SOL)')

#Also get which thermodynamic state is sampled each frame
if stateInfoFile is not None:
  states = np.loadtxt(stateInfoFile)[:,2]

#Want to histogram in 2D, but first get all x and y coordinates of solute at all times
solXY = np.zeros((nFrames, 2))

for t, frame in enumerate(traj):
  coords = np.array(frame.xyz)
  solcoords = coords[solInds]
  solcoords = wl.reimage(solcoords, 0.5*boxDims, boxDims)
  #solXY[t] = np.average(solcoords, axis=0)[:2]
  solXY[t] = solcoords[0][:2]

#Define histogram bins
binSizeX = boxDims[0]/10.0
binSizeY = boxDims[1]/10.0
binsX = np.arange(0.0, boxDims[0]+0.01, binSizeX) #Will exclude a little bit on one edge, but that's ok
binsY = np.arange(0.0, boxDims[1]+0.01, binSizeY) 

#If have multiple thermodynamic states, create histogram for each to check individual sampling
if stateInfoFile is not None:
  print("Doing all states.\n")
  for astate in range(int(np.max(states))+1):
    #Find frames where system is in desired state
    thisXY = solXY[(states == astate)]
    thisHist, thisX, thisY = np.histogram2d(thisXY[:,0], thisXY[:,1], bins=[binsX, binsY], normed=False)
    thisFig, thisAx = plt.subplots(1)
    thisHeat = thisAx.imshow(thisHist, cmap=plt.get_cmap('magma'), aspect='auto')
    thisFig.subplots_adjust(right=0.8)
    thiscbar = thisFig.add_axes([0.85, 0.15, 0.05, 0.7])
    thisFig.colorbar(thisHeat, cax=thiscbar)
    thiscbar.set_ylabel(r'Counts')
    thisFig.savefig('xyhist_state%i.png'%astate)
    if showPlots:
      plt.show()
    #plt.close(thisFig)

#If we know we should be sampling evenly, looking for a small difference between the max and min count bins
#So want to plot this parity as a function of simulation time
simTimes = np.arange(int(nFrames/10), nFrames+int(nFrames/100), int(nFrames/100)) 
histMinMax = np.zeros((len(simTimes),2))

for i, t in enumerate(simTimes):

  #Get histogram up to this amount of simulation time
  xyHist, xEdges, yEdges = np.histogram2d(solXY[:t,0], solXY[:t,1], bins=[binsX, binsY], normed=False)

  #Compute min and max counts
  histMinMax[i,0] = np.min(xyHist)
  histMinMax[i,1] = np.max(xyHist)

  if i == len(simTimes)-1:
    heatFig, heatAx = plt.subplots(1)
    heatHist = heatAx.imshow(xyHist, cmap=plt.get_cmap('magma'), aspect='auto')
    heatFig.subplots_adjust(right=0.8)
    cbarax = heatFig.add_axes([0.85, 0.15, 0.05, 0.7])
    heatFig.colorbar(heatHist, cax=cbarax)
    cbarax.set_ylabel(r'Counts')
    heatFig.savefig('xy_sampling.png')
    print np.sum(xyHist)

sampFig, sampAx = plt.subplots(1)
sampAx.plot(simTimes, histMinMax[:,0], label='Min', linewidth=2.0)
sampAx.plot(simTimes, histMinMax[:,1], label='Max', linewidth=2.0)
sampAx.plot(simTimes, histMinMax[:,1] - histMinMax[:,0], label='Max-Min', linewidth=2.0)
sampAx.set_ylabel(r'Counts')
sampAx.set_xlabel(r'Simulation steps')
sampAx.legend()
sampFig.tight_layout()
sampFig.savefig('sampling_parity.png')

if showPlots:
  plt.show()

