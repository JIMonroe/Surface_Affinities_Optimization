#!/usr/bin/env python

import sys, os
import copy
import time
import numpy as np
from skimage import measure as skmeasure
import waterlib as wl
import water_properties as wp
from netCDF4 import Dataset
import parmed as pmd
import pytraj as pt
from pymbar import mbar


Usage="""solute_water_structure topFile inBulk doReweight
  Computes various properties of water in the hydration shell of a solute. This 
  is intended to be used with expanded-ensemble simulation inserting a molecule
  into either bulk solution or at an interface. This requires that all configurations
  be reweighted to the appropriate ensemble, which includes removing restraints
  that keep the solute in a specific part of the interface. Two functions related
  to that which computes the solvation free energy, etc. are provided to help with
  this. For simulations in bulk, inBulk should be True to use the appropriate
  reweighting calculation, while if inBulk is False it reweights for a solute at
  an interface - if this is the case, you must have directories named Quad*
  that restrain the solute to different quadrants on the surface. Must be run
  from the directory containing either the four directories with simulations
  at each quadrant restraint or the bulk simulation directory with a single 
  trajectory.
  Inputs:
    topFile - topology file associated with trajectories
    inBulk - (default False) flag to say if simulation in bulk or at an interface
    doReweight - (default True) flag for whether or not need to reweight (i.e. in
                                expanded ensemble or not)
  Outputs (all files, no returns):
    
"""


def getConfigWeightsSurf(kB=0.008314459848, T=298.15):
  """Computes and returns the configuration weights for simulations with a solute
   at an interface.
   Mostly replicates calcdGsolv in genetic_lib, but returns config weights in both
   the fully coupled and decoupled states (also includes pV term - won't matter very
   much for free energy differences, but maybe matters for weighting configurations, 
   even though also probably not too much).
  """
  #First define directory structure, spring constants, etc.
  simDirs = ['Quad_0.25X_0.25Y', 'Quad_0.25X_0.75Y', 'Quad_0.75X_0.25Y', 'Quad_0.75X_0.75Y']
  kXY = [10.0, 10.0, 10.0, 10.0] #spring constant in kJ/mol*A^2
  refX = [7.4550, 7.4550, 22.3650, 22.3650]
  refY = [8.6083, 25.8249, 8.6083, 25.8249]
  distRefX = [7.4550, 7.4550, 7.4550, 7.4550]
  distRefY = [8.6083, 8.6083, 8.6083, 8.6083]
  numStates = 19

  #And some constants
  kBT = kB*T
  beta = 1.0 / kBT

  #First make sure all the input arrays have the same dimensions
  numSims = len(simDirs)
  allLens = np.array([len(a) for a in [kXY, refX, refY, distRefX, distRefY]])

  #Want to loop over all trajectories provided, storing solute position information to calculate restraints
  xyPos = None #X and Y coordinates of first heavy atom for all solutes - get shape later
  nSamps = np.zeros((len(simDirs), numStates), dtype=int) #Have as many x-y restraints as sims and same number of lambda states for each 
  allPots = np.array([[]]*numStates).T #Potential energies, EXCLUDING RESTRAINT, for each simulation frame and lambda state
                                       #Will also include pV term because may matter for configurations
  xyBox = np.zeros(2)

  for i, adir in enumerate(simDirs):

    topFile = "%s/../sol_surf.top"%adir
    trajFile = "%s/prod.nc"%adir
    alchemicalFile = "%s/alchemical_output.txt"%adir

    #First load in topology and get atom indices 
    top = pmd.load_file(topFile)

    #Get solute heavy atoms for each solute
    #Also get indices of surface atoms to use as references later
    #Only taking last united atoms of first SAM molecule we find
    heavyIndices = []
    for res in top.residues:
      if res.name not in ['OTM', 'CTM', 'STM', 'NTM', 'SOL']: #Assumes working with SAM surface...
        thisheavyinds = []
        for atom in res.atoms:
          if not atom.name[0] == 'H':
            thisheavyinds.append(atom.idx)
        heavyIndices.append(thisheavyinds)

    #Make into arrays for easier referencing
    heavyIndices = np.array(heavyIndices)

    #Load in the potential energies, INCLUDING RESTRAINT, at all states for this simulation to figure out frames to skip
    alcDat = np.loadtxt(alchemicalFile)
    startTime = alcDat[0, 1]
    startFrame = int(startTime) - 1 #Be careful here... need write frequency in alchemical file to match exactly with positions
                                    #AND assuming that have written in 1 ps increments... 
                                    #Also, first frame in trajectory is NOT at time zero, so subtract 1
    thisPot = alcDat[:, 3:-1]
    thispV = alcDat[:, -1]

    #Next load in the trajectory and get all solute coordinates that matter
    top.rb_torsions = pmd.TrackedList([])
    top = pt.load_parmed(top, traj=False)
    traj = pt.iterload(trajFile, top, frame_slice=(startFrame, -1))
    nFrames = len(traj)
    xyBox = np.array(traj[0].box.values)[:2] #A little lazy, but all boxes should be same and fixed in X and Y dimensions

    thisxyPos = np.zeros((nFrames, len(heavyIndices), 2))
    thisnSamps = np.zeros(numStates, dtype=int)

    #Reference x and y coordinates for this restraint
    thisRefXY = np.array([refX[i], refY[i]])

    for j, frame in enumerate(traj):

      thisPos = np.array(frame.xyz)
      thisXY = thisPos[heavyIndices[:,0]][:, :2] #Takes XY coords for first heavy atom from each solute
      thisxyPos[j,:] = thisXY 
      thisnSamps[int(alcDat[j, 2])] += 1 #Lambda states must be indexed starting at 0

      #Also get wrapped positions relative to each reference face
      #AND calculate xy restraint energy to remove by adding this for each solute
      xyEnergy = 0.0
      for k in range(len(heavyIndices)):
        xy = thisXY[k]
        #Then separately reimage around the restraint reference positions to calculate energy
        xy = wl.reimage([xy], thisRefXY, xyBox)[0] - thisRefXY
        xyEnergy += (  0.5*kXY[i]*(0.5*(np.sign(xy[0] - distRefX[i]) + 1))*((xy[0] - distRefX[i])**2)
                     + 0.5*kXY[i]*(0.5*(np.sign(xy[1] - distRefY[i]) + 1))*((xy[1] - distRefY[i])**2) )

      #Remove the restraint energy (only for x-y restraint... z is the same in all simulations)
      thisPot[j,:] -= (xyEnergy / kBT)

      #And also add in pV contribution
      thisPot[j,:] += thispV[j]

    #Add to other things we're keeping track of
    if xyPos is None:
      xyPos = copy.deepcopy(thisxyPos)
    else:
      xyPos = np.vstack((xyPos, thisxyPos))
    nSamps[i,:] = thisnSamps
    allPots = np.vstack((allPots, thisPot))

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
  
  #Now should be set to run MBAR
  mbarObj = mbar.MBAR(Ukn, nSamps.flatten())

  #Following computePMF in MBAR to get configuration weights with desired potential of interest
  logwCoupled = mbarObj._computeUnnormalizedLogWeights(allPots[:,0])
  logwDecoupled = mbarObj._computeUnnormalizedLogWeights(allPots[:,-1])

  #Also report average solute-system LJ and coulombic potential energies in the fully coupled ensemble
  #(with restraints removed)
  #Just printing these values
  avgQ, stdQ = mbarObj.computeExpectations(allPots[:,0] - allPots[:,4], allPots[:,0])
  avgLJ, stdLJ = mbarObj.computeExpectations(allPots[:,4] - allPots[:,-1], allPots[:,0])
  print("\nAverage solute-system electrostatic potential energy: %f +/- %f"%(avgQ, stdQ))
  print("Average solute-system LJ potential energy: %f +/- %f\n"%(avgLJ, stdLJ))

  #Also print information that can be used to break free energy into components
  #Start by just printing all of the free energies between states
  alldGs, alldGerr = mbarObj.computePerturbedFreeEnergies(allPots.T)
  print("\nAll free energies relative to first (coupled) state:")
  print(alldGs.tolist())
  print(alldGerr.tolist())
  #And the free energy changes associated with just turning on LJ and elctrostatics separately
  dGq = alldGs[0][0] - alldGs[0][4]
  dGqErr = np.sqrt((alldGerr[0][0]**2) + (alldGerr[0][4])**2)
  print("\nElectrostatic dG (with LJ on): %f +/- %f"%(dGq, dGqErr))
  dGlj = alldGs[4][4] - alldGs[4][-1]
  dGljErr = np.sqrt((alldGerr[4][4]**2) + (alldGerr[4][-1])**2)
  print("\nLJ dG (no charges): %f +/- %f"%(dGlj, dGljErr))
  #Now calculate average potential energy differences needed for computing relative entropies
  dUq, dUqErr = mbarObj.computeExpectations(allPots[:,0] - allPots[:,4], allPots[:,0])
  print("\nAverage electrostatic potential energy in fully coupled state: %f +/- %f"%(dUq, dUqErr))
  dUlj, dUljErr = mbarObj.computeExpectations(allPots[:,4] - allPots[:,-1], allPots[:,4])
  print("\nAverage LJ potential energy (no charges) in uncharged state: %f +/- %f"%(dUlj, dUljErr))

  #And return weights after exponentiating log weights and normalizing
  wCoupled = np.exp(logwCoupled)
  wCoupled /= np.sum(wCoupled)
  wDecoupled = np.exp(logwDecoupled)
  wDecoupled /= np.sum(wDecoupled)

  return wCoupled, wDecoupled


def getConfigWeightsBulk(alchfile='alchemical_output.txt', kB=0.008314459848, T=298.15):
  """Given an alchemical output file, computes and returns configuration weights in
   both the fully coupled and decoupled ensembles of the solute.
  """
  rawdat = np.loadtxt(alchfile)
  lstates = rawdat[:,2]
  Ukn = rawdat[:,3:-1] 
  pV = rawdat[:,-1] #pV term is in last column
  #pV term doesn't matter for free energy differences
  #But does matter for configuration weights (even though it's a small contribution)

  Nsamps = np.zeros(Ukn.shape[1], dtype=int)

  for i in range(Ukn.shape[1]):
    Nsamps[i] = int(np.sum((lstates==i)))
    Ukn[:,i] += pV

  #neworder = np.argsort(lstates)
  #Ukn = Ukn[neworder]

  Ukn /= (kB*T)

  mbarObj = mbar.MBAR(Ukn.T, Nsamps)

  #Following computePMF in MBAR to get configuration weights with desired potential of interest
  logwCoupled = mbarObj._computeUnnormalizedLogWeights(Ukn[:,0])
  logwDecoupled = mbarObj._computeUnnormalizedLogWeights(Ukn[:,-1])

  #Also report average solute-system LJ and coulombic potential energies in the fully coupled ensemble
  #(with restraints removed)
  #Just printing these values
  avgQ, stdQ = mbarObj.computeExpectations(Ukn[:,0] - Ukn[:,4], Ukn[:,0])
  avgLJ, stdLJ = mbarObj.computeExpectations(Ukn[:,4] - Ukn[:,-1], Ukn[:,0])
  print("\nAverage solute-water electrostatic potential energy: %f +/- %f"%(avgQ, stdQ))
  print("Average solute-water LJ potential energy: %f +/- %f\n"%(avgLJ, stdLJ))

  #Also print information that can be used to break free energy into components
  #Start by just printing all of the free energies between states
  alldGs, alldGerr = mbarObj.computePerturbedFreeEnergies(Ukn.T)
  print("\nAll free energies relative to first (coupled) state:")
  print(alldGs.tolist())
  print(alldGerr.tolist())
  #And the free energy changes associated with just turning on LJ and elctrostatics separately
  dGq = alldGs[0][0] - alldGs[0][4]
  dGqErr = np.sqrt((alldGerr[0][0]**2) + (alldGerr[0][4])**2)
  print("\nElectrostatic dG (with LJ on): %f +/- %f"%(dGq, dGqErr))
  dGlj = alldGs[4][4] - alldGs[4][-1]
  dGljErr = np.sqrt((alldGerr[4][4]**2) + (alldGerr[4][-1])**2)
  print("\nLJ dG (no charges): %f +/- %f"%(dGlj, dGljErr))
  #Now calculate average potential energy differences needed for computing relative entropies
  dUq, dUqErr = mbarObj.computeExpectations(Ukn[:,0] - Ukn[:,4], Ukn[:,0])
  print("\nAverage electrostatic potential energy in fully coupled state: %f +/- %f"%(dUq, dUqErr))
  dUlj, dUljErr = mbarObj.computeExpectations(Ukn[:,4] - Ukn[:,-1], Ukn[:,4])
  print("\nAverage LJ potential energy (no charges) in uncharged state: %f +/- %f"%(dUlj, dUljErr))

  #And return weights after exponentiating log weights
  wCoupled = np.exp(logwCoupled)
  wCoupled /= np.sum(wCoupled)
  wDecoupled = np.exp(logwDecoupled)
  wDecoupled /= np.sum(wDecoupled)

  return wCoupled, wDecoupled


def main(args):

  print time.ctime(time.time())

  #Get topology file we're working with
  topFile = args[0]

  #And figure out if we're dealing with solute at surface or in bulk
  if (args[1] == 'True'):
    inBulk = True
  else:
    inBulk = False

  if (args[2] == 'True'):
    doReweight = True
  else:
    doReweight = False

  #Read in topology file now to get information on solute atoms
  top = pmd.load_file(topFile)
  soluteInds = []
  for res in top.residues:
    if res.name not in ['OTM', 'CTM', 'STM', 'NTM', 'SOL']:
      for atom in res.atoms:
        soluteInds.append(atom.idx)

  #Now define how we compute three-body angles with bins and cut-off
  #Shell cut-off
  shellCut = 3.32 #1st minimum distance for TIP4P-Ew water at 298.15 K and 1 bar
  #Number of angle bins
  nAngBins = 100 #500
  #Define bin centers (should be nBins equally spaced between 0 and 180)
  angBinCents = 0.5 * (np.arange(0.0, 180.001, 180.0/nAngBins)[:-1] + np.arange(0.0, 180.001, 180.0/nAngBins)[1:])

  #And distance bins for local oxygen-oxygen RDF calculation 
  #(really distance histograms from central oxygens - can normalize however we want, really)
  distBinWidth = 0.05
  nDistBins = int(shellCut / distBinWidth)
  distBins = np.arange(0.0, nDistBins*distBinWidth+0.00001, distBinWidth)
  distBinCents = 0.5 * (distBins[:-1] + distBins[1:])

  #Define the size of the probes used for assessing density fluctuations near solute
  probeRadius = 3.3 # radius in Angstroms; the DIAMETER of a methane so assumes other atoms methane-sized

  #And bins for numbers of waters in probes
  probeBins = np.arange(0.0, 21.00001, 1.0)
  nProbeBins = len(probeBins) - 1 #Will use np.histogram, which includes left edge in bin (so if want up to 20, go to 21)

  #And will record number waters in each solvation shell (histograms of)
  shellBins = np.arange(0.0, 251.00001, 1.0) #probably way too many bins, but don't want to run out
  nShellBins = len(shellBins) - 1

  #Should we do 2D histogram for angles and distances?
  #Or also do 2D histogram for number of waters in probe and three-body angle?
  #Interesting if make probe radius same size as three-body angle cutoff?

  #Finally, also define the bins for computing RDFs of waters near all solute atoms
  #Will use to define 1st and 2nd solvation shells
  rdfBinWidth = 0.2
  rdfMax = 12.00
  rdfBins = np.arange(0.0, rdfMax+0.000001, rdfBinWidth)
  rdfBinCents = 0.5 * (rdfBins[:-1] + rdfBins[1:])
  nRDFBins = len(rdfBinCents)
  rdfBinVols = (4.0*np.pi/3.0)*(rdfBins[1:]**3 - rdfBins[:-1]**3)
  bulkDens = 0.0332 #Roughly right for TIP4P-EW at 298.15 K and 1 bar in inverse Angstroms cubed

  #Need to create a variety of arrays to hold the data we're interested in
  #Will record distributions in both 1st and 2nd solvation shells
  shellCountsCoupled = np.zeros((nShellBins, 2)) #Histograms for numbers of waters in hydration shells of solutes
  probeHistsCoupled = np.zeros((nProbeBins, 2)) #Histograms for numbers waters in probes in 1st and 2nd hydration shells 
  angHistsCoupled = np.zeros((nAngBins, 2)) #Histograms of three-body angles for water oxygens within solvation shells
  distHistsCoupled = np.zeros((nDistBins, 2)) #Histograms of distances to water oxygens from central oxygens
  solRDFsCoupled = np.zeros((nRDFBins, len(soluteInds))) #RDFs between each solute atom and water oxygens
  shellCountsDecoupled = np.zeros((nShellBins, 2)) #Same as above, but decoupled state, not coupled state 
  probeHistsDecoupled = np.zeros((nProbeBins, 2)) 
  angHistsDecoupled = np.zeros((nAngBins, 2)) 
  distHistsDecoupled = np.zeros((nDistBins, 2)) 
  solRDFsDecoupled = np.zeros((nRDFBins, len(soluteInds))) 

  #First need configuration weights to use in computing average quantities
  #But only do if we're using a simuation with an expanded ensemble
  if doReweight:
    if inBulk:
      weightsCoupled, weightsDecoupled = getConfigWeightsBulk(kB=1.0, T=1.0)
      #Using 1 for kB and T because alchemical_output.txt should already have potential energies in kBT
      simDirs = ['.']
    else:
      weightsCoupled, weightsDecoupled = getConfigWeightsSurf()
      simDirs = ['Quad_0.25X_0.25Y', 'Quad_0.25X_0.75Y', 'Quad_0.75X_0.25Y', 'Quad_0.75X_0.75Y']
  else:
    weightsCoupled = np.array([])
    weightsDecoupled = np.array([])
    simDirs = ['.']

  #To correctly match weights up to configurations, need to count frames from all trajectories
  countFrames = 0

  #Next, want to loop over all trajectories and compute RDFs from solute atoms to water oxygens
  #Will use this to define solvation shells for finding other properties
  #Actually, having looked at RDFs, just use 5.5 for first shell and 8.5 for second shell...
  #AND use all atoms, including hydrogens, which have LJ interactions in GAFF2, to define shells
  #Actually now only using heavy atoms... but when look at RDFs, examine all atoms 
  for adir in simDirs:

    if doReweight:
      #Before loading trajectory, figure out how many frames to exclude due to weight equilibration
      alcDat = np.loadtxt(adir+'/alchemical_output.txt')
      startTime = alcDat[0, 1]
      startFrame = int(startTime) - 1
    else:
      startFrame = 0
 
    top = pmd.load_file(topFile)
    top.rb_torsions = pmd.TrackedList([]) #This is just for SAM systems so that it doesn't break pytraj
    top = pt.load_parmed(top, traj=False)
    traj = pt.iterload(adir+'/prod.nc', top, frame_slice=(startFrame, -1))

    if not doReweight:
      weightsCoupled = np.hstack((weightsCoupled, np.ones(len(traj))))
      weightsDecoupled = np.hstack((weightsDecoupled, np.ones(len(traj))))
  
    print("\nTopology and trajectory loaded from directory %s" % adir)

    owInds = top.select('@OW')
    soluteInds = top.select('!(:OTM,CTM,STM,NTM,SOL)')

    print("\n\tFound %i water oxygens" % len(owInds))
    print("\tFound %i solute atoms" % len(soluteInds))

    for i, frame in enumerate(traj):
      
      if i%1000 == 0:
        print "On frame %i" % i
    
      boxDims = np.array(frame.box.values[:3])

      currCoords = np.array(frame.xyz)

      #Wrap based on soluate atom center of geometry and get coordinates of interest
      wrapCOM = np.average(currCoords[soluteInds], axis=0)
      currCoords = wl.reimage(currCoords, wrapCOM, boxDims) - wrapCOM
      owCoords = currCoords[owInds]
      solCoords = currCoords[soluteInds]

      #Loop over solute atoms and find pair-distance histograms with water oxygens
      for j, acoord in enumerate(solCoords):
        solRDFsCoupled[:,j] += (weightsCoupled[countFrames+i]
                                * wl.pairdistancehistogram(np.array([acoord]), owCoords, rdfBinWidth, nRDFBins, boxDims))
        solRDFsDecoupled[:,j] += (weightsDecoupled[countFrames+i]
                                  * wl.pairdistancehistogram(np.array([acoord]), owCoords, rdfBinWidth, nRDFBins, boxDims))
        #Note that pairdistancehistogram is right-edge inclusive, NOT left-edge inclusive
        #In practice, not a big difference

    countFrames += len(traj)

  #Finish by normalizing RDFs properly
  for j in range(len(soluteInds)):
    solRDFsCoupled[:,j] /= rdfBinVols #bulkDens*rdfBinVols
    solRDFsDecoupled[:,j] /= rdfBinVols #bulkDens*rdfBinVols
  if not doReweight:
    solRDFsCoupled /= float(countFrames)
    solRDFsDecoupled /= float(countFrames)

  #And save to file
  np.savetxt('solute-OW_RDFs_coupled.txt', np.hstack((np.array([rdfBinCents]).T, solRDFsCoupled)), 
             header='RDF bins (A)    solute atom-OW RDF for solute atom indices %s'%(str(soluteInds)))
  np.savetxt('solute-OW_RDFs_decoupled.txt', np.hstack((np.array([rdfBinCents]).T, solRDFsDecoupled)), 
             header='RDF bins (A)    solute atom-OW RDF for solute atom indices %s'%(str(soluteInds)))

  print("\tFound RDFs for water oxygens from solute indices.")

  solShell1Cut = 5.5 #Angstroms from all solute atoms (including hydrogens)
  solShell2Cut = 8.5

  #And now that we know how many frames, we can assign real weights if not reweighting
  if not doReweight:
    weightsCoupled /= float(countFrames)
    weightsDecoupled /= float(countFrames)

  #Reset countFrames so get weights right
  countFrames = 0

  #Repeat looping over trajectories to calculate water properties in solute solvation shell
  for adir in simDirs:

    if doReweight:
      #Before loading trajectory, figure out how many frames to exclude due to weight equilibration
      alcDat = np.loadtxt(adir+'/alchemical_output.txt')
      startTime = alcDat[0, 1]
      startFrame = int(startTime) - 1
    else:
      startFrame = 0
 
    top = pmd.load_file(topFile)
    top.rb_torsions = pmd.TrackedList([]) #This is just for SAM systems so that it doesn't break pytraj
    top = pt.load_parmed(top, traj=False)
    traj = pt.iterload(adir+'/prod.nc', top, frame_slice=(startFrame, -1))
  
    print("\nTopology and trajectory loaded from directory %s" % adir)

    owInds = top.select('@OW')
    soluteInds = top.select('!(:OTM,CTM,STM,NTM,SOL)&!(@H=)')
    surfInds = top.select('(:OTM,CTM,STM,NTM)&!(@H=)') #For probe insertions, also include solute and surface heavy atoms

    print("\n\tFound %i water oxygens" % len(owInds))
    print("\tFound %i solute heavy atoms" % len(soluteInds))
    print("\tFound %i non-hydrogen surface atoms" % len(surfInds))

    if len(surfInds) == 0:
      surfInds.dtype=int

    for i, frame in enumerate(traj):
  
      #if i%10 == 0:
      #  print "On frame %i" % i
    
      boxDims = np.array(frame.box.values[:3])
  
      currCoords = np.array(frame.xyz)
  
      #Wrap based on soluate atom center of geometry and get coordinates of interest
      wrapCOM = np.average(currCoords[soluteInds], axis=0)
      currCoords = wl.reimage(currCoords, wrapCOM, boxDims) - wrapCOM
      owCoords = currCoords[owInds]
      solCoords = currCoords[soluteInds]
      surfCoords = currCoords[surfInds]

      #Now get solvent shells around solute
      shell1BoolMat = wl.nearneighbors(solCoords, owCoords, boxDims, 0.0, solShell1Cut)
      shell1Bool = np.array(np.sum(shell1BoolMat, axis=0), dtype=bool)

      shell2BoolMat = wl.nearneighbors(solCoords, owCoords, boxDims, solShell1Cut, solShell2Cut)
      shell2Bool = np.array(np.sum(shell2BoolMat, axis=0), dtype=bool)

      #And add weight to histogram for numbers of waters in shells
      thisCount1 = int(np.sum(shell1Bool))
      shellCountsCoupled[thisCount1, 0] += weightsCoupled[countFrames+i]
      shellCountsDecoupled[thisCount1, 0] += weightsDecoupled[countFrames+i]

      thisCount2 = int(np.sum(shell2Bool))
      shellCountsCoupled[thisCount2, 1] += weightsCoupled[countFrames+i]
      shellCountsDecoupled[thisCount2, 1] += weightsDecoupled[countFrames+i]

      #And compute water properties of solvent shells, first 3-body angles
      thisAngs1, thisNumAngs1 = wp.getCosAngs(owCoords[shell1Bool], owCoords, boxDims, highCut=shellCut)
      thisAngHist1, thisAngBins1 = np.histogram(thisAngs1, bins=nAngBins, range=[0.0, 180.0], density=False)
      angHistsCoupled[:,0] += weightsCoupled[countFrames+i] * thisAngHist1
      angHistsDecoupled[:,0] += weightsDecoupled[countFrames+i] * thisAngHist1

      thisAngs2, thisNumAngs2 = wp.getCosAngs(owCoords[shell2Bool], owCoords, boxDims, highCut=shellCut)
      thisAngHist2, thisAngBins2 = np.histogram(thisAngs2, bins=nAngBins, range=[0.0, 180.0], density=False)
      angHistsCoupled[:,1] += weightsCoupled[countFrames+i] * thisAngHist2
      angHistsDecoupled[:,1] += weightsDecoupled[countFrames+i] * thisAngHist2

      #And ow-ow pair distance histograms in both shells as well
      thisDistHist1 = wl.pairdistancehistogram(owCoords[shell1Bool], owCoords, distBinWidth, nDistBins, boxDims)
      distHistsCoupled[:,0] += weightsCoupled[countFrames+i] * thisDistHist1
      distHistsDecoupled[:,0] += weightsDecoupled[countFrames+i] * thisDistHist1

      thisDistHist2 = wl.pairdistancehistogram(owCoords[shell2Bool], owCoords, distBinWidth, nDistBins, boxDims)
      distHistsCoupled[:,1] += weightsCoupled[countFrames+i] * thisDistHist2
      distHistsDecoupled[:,1] += weightsDecoupled[countFrames+i] * thisDistHist2

      #Next compute distributions of numbers of waters in probes centered within each shell
      #To do this, create random grid of points in SQUARE that encompasses both shells
      #Then only keep points within each shell based on distance
      #Square will be based on shell cutoffs and min and max coordinates in each dimension of solute
      minSolX = np.min(solCoords[:,0]) - solShell2Cut
      maxSolX = np.max(solCoords[:,0]) + solShell2Cut
      minSolY = np.min(solCoords[:,1]) - solShell2Cut
      maxSolY = np.max(solCoords[:,1]) + solShell2Cut
      minSolZ = np.min(solCoords[:,2]) - solShell2Cut
      maxSolZ = np.max(solCoords[:,2]) + solShell2Cut
      thisGridX = minSolX + np.random.random(500)*(maxSolX - minSolX)
      thisGridY = minSolY + np.random.random(500)*(maxSolY - minSolY)
      thisGridZ = minSolZ + np.random.random(500)*(maxSolZ - minSolZ)
      thisGrid = np.vstack((thisGridX, thisGridY, thisGridZ)).T

      gridBoolMat1 = wl.nearneighbors(solCoords, thisGrid, boxDims, 0.0, solShell1Cut)
      gridBool1 = np.array(np.sum(gridBoolMat1, axis=0), dtype=bool)
      thisNum1 = wl.probegrid(np.vstack((owCoords, surfCoords, solCoords)), thisGrid[gridBool1], probeRadius, boxDims)
      thisProbeHist1, thisProbeBins1 = np.histogram(thisNum1, bins=probeBins, density=False)
      probeHistsCoupled[:,0] += weightsCoupled[countFrames+i] * thisProbeHist1
      probeHistsDecoupled[:,0] += weightsDecoupled[countFrames+i] * thisProbeHist1

      gridBoolMat2 = wl.nearneighbors(solCoords, thisGrid, boxDims, solShell1Cut, solShell2Cut)
      gridBool2 = np.array(np.sum(gridBoolMat2, axis=0), dtype=bool)
      thisNum2 = wl.probegrid(np.vstack((owCoords, surfCoords, solCoords)), thisGrid[gridBool2], probeRadius, boxDims)
      thisProbeHist2, thisProbeBins2 = np.histogram(thisNum2, bins=probeBins, density=False)
      probeHistsCoupled[:,1] += weightsCoupled[countFrames+i] * thisProbeHist2
      probeHistsDecoupled[:,1] += weightsDecoupled[countFrames+i] * thisProbeHist2

    countFrames += len(traj)

  #Should have everything we need, so save to text files
  np.savetxt('solute_shell_hists.txt', 
             np.hstack((np.array([shellBins[:-1]]).T, shellCountsCoupled, shellCountsDecoupled)),
             header='Histograms of numbers of waters in first and second solute solvation shells with solvent in coupled (columns 2, 3) and decoupled (columns 4, 5) states')
  np.savetxt('solute_probe_hists.txt', 
             np.hstack((np.array([probeBins[:-1]]).T, probeHistsCoupled, probeHistsDecoupled)),
             header='Number waters in probe histograms in first and second solute solvation shells with solvent in coupled (columns 2, 3) and decoupled (columns 4, 5) states')
  np.savetxt('solute_ang_hists.txt', 
             np.hstack((np.array([angBinCents]).T, angHistsCoupled, angHistsDecoupled)),
             header='3-body angle histograms in first and second solute solvation shells with solvent in coupled (columns 2, 3) and decoupled (columns 4, 5) states')
  np.savetxt('solute_pair_hists.txt', 
             np.hstack((np.array([distBinCents]).T, distHistsCoupled, distHistsDecoupled)),
             header='O-O pair-distance histograms in first and second solute solvation shells with solvent in coupled (columns 2, 3) and decoupled (columns 4, 5) states')

  print time.ctime(time.time())
 

if __name__ == "__main__":
  main(sys.argv[1:])


