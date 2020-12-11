#!/usr/bin/env python

import sys, os
import copy
import numpy as np
from scipy.integrate import simps
import parmed as pmd
import pytraj as pt
import waterlib as wl

Usage="""python solute_insertions.py topFile trajFile

Computes hard-sphere radii for solute atoms with all other non-solute atoms. Also reports
average decoupled (gas phase) solute volume based on HS radii with solute atoms and water 
oxygens, which is most relevant to the solute size. More importantly, computes and saves
a histogram for the number of non-solute atoms overlapping with solute atoms, as calculated
by the HS radii from turning the LJ potential between the atom types into a WCA potential
and integrating 1-exp(-beta*U). 
  Inputs:
          topFile - topology file (must have parameter information)
          trajFile - trajectory file (can be in bulk or at interface)
  Outputs:
          HS-solute_overlap_hist.txt - text file of histograms of overlapping non-solute
"""

def main(args):

  print('\nReading in files and obtaining LJ information...')

  #First read in topology and trajectory
  topFile = args[0]
  trajFile = args[1]

  top = pmd.load_file(topFile)
  trajtop = copy.deepcopy(top)
  trajtop.rb_torsions = pmd.TrackedList([]) #Necessary for SAM systems so doesn't break pytraj
  trajtop = pt.load_parmed(trajtop, traj=False)
  traj =  pt.iterload(trajFile, top=trajtop)

  #Next use parmed to get LJ parameters for all atoms in the solute, as well as water oxygens and surface atoms
  #While go, also collect dictionaries of atomic indices associated with each atom type
  #Will have to check separately when looking at overlaps with solute
  soluteLJ = []
  soluteInds = []
  dictOtherLJ = {}
  dictOtherInds = {}
  for res in top.residues:
    if res.name not in ['OTM', 'CTM', 'STM', 'NTM', 'SOL']:
      for atom in res.atoms:
        soluteInds.append(atom.idx)
        soluteLJ.append([atom.sigma, atom.epsilon])
    else:
      for atom in res.atoms:
        if not atom.type in dictOtherInds:
          dictOtherInds[atom.type] = [atom.idx]
        else:
          dictOtherInds[atom.type].append(atom.idx)
        if not atom.type in dictOtherLJ:
          dictOtherLJ[atom.type] = np.array([atom.sigma, atom.epsilon])
      
  soluteLJ = np.array(soluteLJ)

  print(soluteLJ)

  #Use Lorentz-Berthelot combining rules to get LJ parameters between each solute atom and a water oxygen
  dictMixLJ = {}
  for i in range(soluteLJ.shape[0]):
    for akey in dictOtherLJ:
      dictMixLJ['%i_%s'%(i, akey)] = np.array([0.5*(soluteLJ[i,0] + dictOtherLJ[akey][0]), 
                                               np.sqrt(soluteLJ[i,1]*dictOtherLJ[akey][1])])

  for key, val in dictMixLJ.iteritems():
    print("%s, %s"%(key, str(val.tolist())))

  print('\nDetermining hard-sphere radii for all combinations of solute and other system atoms...')

  #Next compute hard-sphere radii by integrating according to Barker and Hendersen, Weeks, etc.
  #In order to this right, technically using WCA potential, not LJ
  hsRadii = {}
  rvals = np.arange(0.0, 50.005, 0.005)
  betaLJ = 1.0 / ((1.9872036E-03)*298.15)
  for i in range(soluteLJ.shape[0]):
    for akey in dictOtherLJ:
      [thisSig, thisEps] = dictMixLJ['%i_%s'%(i, akey)]
      if thisEps == 0.0:
        hsRadii['%i_%s'%(i, akey)] = 0.0
        continue
      thisRmin = thisSig * (2.0**(1.0/6.0))
      thisSigdRmin6 = (thisSig/thisRmin)**6
      thisPotRmin = 4.0*thisEps*((thisSigdRmin6**2) - thisSigdRmin6)
      thisSigdR6 = (thisSig/rvals)**6
      thisPotVals = 4.0*thisEps*((thisSigdR6**2) - thisSigdR6) - thisPotRmin
      thisPotVals[np.where(rvals >= thisRmin)] = 0.0
      thisPotVals *= betaLJ
      thisIntegrand = 1.0 - np.exp(-thisPotVals)
      thisIntegrand[0] = 1.0
      hsRadii['%i_%s'%(i, akey)] = simps(thisIntegrand, rvals) 
      #Need to multiply by two because will use atom centers to check overlap? Is Rhs distance between centers?

  for key, val in hsRadii.iteritems():
    print("%s, %f"%(key, val))

  #Keep track of hard-sphere radii with water oxygens specially
  solOWhsRadii = np.zeros(soluteLJ.shape[0])
  for i in range(soluteLJ.shape[0]):
    solOWhsRadii[i] = hsRadii['%i_OW_tip4pew'%i] #Only using TIP4P/EW here

  print('\nStarting loop over trajectory...')
  
  #Now loop over trajectory and check if solute is overlapping with any waters OR surface atoms
  #Will create a distribution of overlapping atoms
  numOverlap = np.arange(101.0)
  countOverlap = np.zeros(len(numOverlap))

  #And also track average solute volume that we're trying to insert
  solVol = 0.0

  countFrames = 0
  print('')

  for frame in traj:

    if countFrames%100 == 0:
      print("On frame %i"%countFrames)

    countFrames += 1
    
    boxDims = np.array(frame.box.values[:3])
  
    currCoords = np.array(frame.xyz)

    #Get solute coordinates and make sure solute is whole
    #Then get solute coordinates relative to first solute atom
    #Don't need any other wrapping because wl.nearneighbors will do its own wrapping
    solCoords = currCoords[soluteInds]
    solRefCoords = wl.reimage(solCoords, solCoords[0], boxDims) - solCoords[0]

    #While we have the coordinates nice, compute solute volume
    #Do this based on hard-sphere radii to water oxygens, which is the most likely case anyway
    solVol += np.sum(wl.spherevolumes(solRefCoords, solOWhsRadii, 0.1))

    #Will shift the solute first atom (and others too) to a number of random locations
    #Since applying restraint (if have surface) Z is drawn from the distribution we want
    #X and Y can be drawn more easily from uniform distributions, so do this to randomize solute position
    numRandXY = 1000
    randX = np.random.random(numRandXY) * boxDims[0]
    randY = np.random.random(numRandXY) * boxDims[1]
   
    #And will keep track of number of overlapping atoms for each solute random position
    thisTotOverlap = np.zeros(numRandXY, dtype=int)

    #Loop over all non-solute atoms in the system by atom type
    for akey, val in dictOtherInds.iteritems():

      #For this specific atom type, need to keep track of WHICH neighbors overlapping
      #And need to do for EACH solute atom
      #Don't want to double-count if two solute atoms both overlap the same other atom
      overlapBool = np.zeros((numRandXY, len(val)), dtype=int)

      #Now loop over each solute atom
      for i, coord in enumerate(solRefCoords):

        thisRadius = hsRadii['%i_%s'%(i, akey)]
        if thisRadius == 0.0:
          continue

        #Define coordinates of the current solute atom we're working with
        #So setting first atom to random XY position, then shifting by distance to first atom
        hsCoords = np.zeros((numRandXY, 3))
        hsCoords[:,0] = coord[0] + randX
        hsCoords[:,1] = coord[1] + randY
        hsCoords[:,2] = coord[2] + solCoords[0,2]

        #Identify boolean for overlapping atoms and add to overall boolean for overlap
        #Note that want OR operation, so adding boolean arrays
        overlapBool += wl.nearneighbors(hsCoords, currCoords[val], boxDims, 0.0, thisRadius)
      
      #For this non-solute atom type, add number of atoms overlapping with ANY solute atom 
      thisTotOverlap += np.sum(np.array(overlapBool, dtype=bool), axis=1)

    thisBins = np.arange(np.max(thisTotOverlap) + 1)
    countOverlap[thisBins] += np.bincount(thisTotOverlap)

  print(countOverlap.tolist())
  print('Hard-sphere solute insertion probability: %f'%(-np.log(countOverlap[0]/np.sum(countOverlap))))

  #Save the distribution to file
  np.savetxt('HS-solute_overlap_hist.txt', np.vstack((numOverlap, countOverlap)).T,
             header='Number of non-solute atoms overlapping           Histogram count')

  solVol /= float(len(traj))
  print('Average solute hard-sphere volume (based on water oxygen LJ params): %f'%(solVol))


if __name__ == "__main__":
  main(sys.argv[1:])
 
