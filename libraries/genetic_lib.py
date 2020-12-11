#! /usr/bin/env python

import sys, os
import shutil
import subprocess
from datetime import datetime
import time
import pickle
import glob
import copy
import numpy as np
import scipy.stats as stats
import scipy.optimize as optimize
import pytraj as pt
import parmed as pmd
import waterlib as wl
from pymbar import mbar

Usage = """A library of classes and functions to be used in optimizing arbitrary metrics on quartz and SAM surfaces.
"""

class SurfStructure:
  """A class to hold information about silanol surface structures.
     Attributes:
                 topFile - topology file for this surface
                 strucFile - structure file for this surface
                 OHPairs - all eligible pairs of hydroxyls on this
                           surface (even those that have been turned
                           into hydroxyls); note that this should be a
                           Nx2x2 array, with each pair of pairs a set of 
                           corresponding pairs on the upper and lower 
                           surfaces of the slab; N is the number of  
                           total oxygens on one side of the slab
                 exclPairs - Nx2 list of pairs that cannot be simultaneously used
                             for a given eligible pair in OHpairs (since contain
                             the same hydroxyl oxygen index, so can't condense 
                             that hydroxyl twice)
                 condPairs - boolean array that specifies which of the sites in
                             OHPairs have been condensed; this is of length N
                 gen - the generation in which this surface was generated
                 metric - the metric (to be optimized by the GA) associated with this surface
                 face - the crystallographic face used (001 or 101 or amorph)
  """
  def __init__(self, topFile="topology.top", strucFile="structure.gro", OHPairs=[], exclPairs=[], condPairs=[], gen=0, metric=None, face='101'):
    self.topFile = topFile
    self.strucFile = strucFile
    self.OHPairs = np.array(OHPairs) 
    self.exclPairs = np.array(exclPairs)
    #Check to make sure all lists in exclPairs converted to arrays...
    #may not be the case if had list of different length lists, like for amorphous surface
    for ind, alist in enumerate(self.exclPairs):
      if isinstance(alist, list):
        self.exclPairs[ind] = np.array(alist)
    if len(condPairs) == 0 and len(OHPairs) > 0:
      condPairs = np.zeros(len(OHPairs), dtype=bool)
    self.condPairs = np.array(condPairs)
    #Boolean array mask over all oxygen pairs eligible to condense
    self.gen = gen #Keeps track of generation this structure belongs to
    self.metric = None #Can also associate arbitrary metric with this structure!
    self.face = face

  def makeCartoon(self, rows=7, cols=8):
    """Creates a cartoon (2D array of ones and zeros) representation of 
       the surface.
       Squares that are ones are hydroxyls, and blank squares (zeros) are 
       hydroxyls that have taken part in condensation.
       Note that this is done based on manual assignment of hydroxyl positions
       within the grid.
       This can easily be used to generate a heat-map, as done in analyzeOpt.
       Additionally, a bond cartoon is returned to show which hydroxyls have
       "bonded" or condensed to form bridging oxygens. Just a list of index
       pairs within the grid.
       rows - (default 8) number of rows for hydroxyls represented in cartoon
       cols - (default 14) number of columns for hydroxyls represented in cartoon
    """
    #Define order of hydroxyl oxygen indices and dictionary to map to locations on grid
    if self.face == '001':
      rows = 8
      cols = 14
      cartoonVec = np.array([1729, 1730, 1775, 1776, 1817, 1818, 1554, 1555, 1597, 1598, 1641, 1642, 1686, 1687,
                             760, 761, 804, 805, 849, 850, 892, 918, 630, 631, 674, 675, 716, 717,
                             1718, 1719, 1764, 1765, 1806, 1807, 1542, 1543, 1586, 1587, 1630, 1631, 1676, 1677,
                             749, 750, 793, 794, 837, 838, 884, 915, 619, 620, 663, 664, 705, 706,
                             1752, 1753, 1795, 1796, 1839, 1840, 1576, 1577, 1619, 1620, 1664, 1665, 1707, 1708,
                             783, 784, 826, 827, 872, 873, 908, 923, 652, 653, 695, 696, 738, 739,
                             1741, 1742, 1785, 1786, 1828, 1829, 1564, 1565, 1608, 1609, 1652, 1653, 1696, 1697,
                             771, 772, 815, 816, 860, 861, 900, 920, 641, 642, 685, 686, 727, 728], dtype=int)
    elif self.face == '101':
      rows = 7
      cols = 8
      cartoonVec = np.array([127, 85, 84, 99, 98, 113, 112, 120,
                             126, 83, 82, 97, 96, 111, 110, 119,
                             125, 81, 80, 95, 94, 109, 108, 118,
                             124, 79, 78, 93, 92, 107, 106, 117,
                             123, 77, 76, 91, 90, 105, 104, 116,
                             122, 75, 74, 89, 88, 103, 102, 115,
                             121, 73, 72, 87, 86, 101, 100, 114], dtype=int)
    cartoonDict = {}
    rowCount = 0
    columnCount = 0
    for k, ind in enumerate(cartoonVec):
      cartoonDict[str(ind)] = [rowCount, columnCount]
      if columnCount == cols-1:
        rowCount += 1
        columnCount = 0
      else:
        columnCount += 1

    cartoonPlot = np.ones((rows, cols))
    bondCartoon = []
    for indpair in self.OHPairs[self.condPairs][:,0]:
      for ind in indpair:
        rowcolinds = cartoonDict[str(ind)]
        cartoonPlot[rowcolinds[0], rowcolinds[1]] = 0
      inds1 = cartoonDict[str(indpair[0])]
      inds2 = cartoonDict[str(indpair[1])]
      #Enforce periodicity in bonds so don't reach across box
      if abs(inds1[0] - inds2[0]) > 0.5*rows:
        if inds1[0] < inds2[0]:
          inds2[0] = inds2[0] - rows
        else:
          inds1[0] = inds1[0] - rows
      if abs(inds1[1] - inds2[1]) > 0.5*cols:
        if inds1[1] > inds2[1]:
          inds2[1] = inds2[1] + cols
        else:
          inds1[1] = inds1[1] + cols
      bondCartoon.append(np.array([cartoonDict[str(indpair[0])], cartoonDict[str(indpair[1])]]))

    bondCartoon = np.array(bondCartoon)

    return cartoonPlot, bondCartoon


class SAMStructure:
  """A class to hold information about SAM surface structures.
     Attributes:
                 topFile - topology file for this surface
                 strucFile - structure file for this surface
                 chainTypes - integer array that specifies the type of each site 
                              on the SAM surface lattice.
                              Typically, 0 will be CH3, though this is arbitrary.
                              Other integers could represent OH or + or - charged headgroup.
                              To map back to lattice positions of chains, use
                              (0,0)->0, (0,1)->1, (0,2)->2, ... (i,j)->i*M+j ... (N,M)->N*M+M
                              where the grid or lattice is of dimension NxM
                 gen - the generation in which this surface was generated
                 metric - the metric (to be optimized) associated with this surface
                 metricpredicted - boolean for whether or not a model has been used (instead of MD) to predict the metric
  """
  def __init__(self, topFile="topology.top", strucFile="structure.gro", chainTypes=[], gen=0, metric=None, metricpredicted=False):
    self.topFile = topFile
    self.strucFile = strucFile
    self.chainTypes = np.array(chainTypes)
    self.gen = gen #Keeps track of generation this structure belongs to
    self.metric = None #Can also associate arbitrary metric with this structure!
    self.metricpredicted = metricpredicted

  def makeCartoon(self, rows=8, cols=9):
    """Creates a cartoon (2D array of ones and zeros) representation of 
       the surface.
       Squares that are ones are hydroxyls, and blank squares (zeros) are 
       CH3 terminated SAM chains.
       This can easily be used to generate a heat-map.
       rows - (default 8) number of rows for chains represented in cartoon
       cols - (default 9) number of columns for chains represented in cartoon
    """
    cartoonPlot = np.zeros((rows, cols))
    for i in range(rows):
      for j in range(cols):
        cartoonPlot[i,j] = self.chainTypes[i*cols+j]

    return cartoonPlot


class ToyStructure:
  """A class to hold information about toy (LJ) surface structures.
     Attributes:
                 topFile - topology file for this surface
                 strucFile - structure file for this surface
                 LJTypes - boolean array that specifies which of the sites on the
                           outer layer of the surface lattice are attractive or 
                           repulsive LJ particles (epsilon or a multiple of epsilon).
                           A 0 or False is repulsive, 1 or True is attractive.
                           To map back to lattice positions, use
                           (0,0)->0, (0,1)->1, (0,2)->2, ... (i,j)->i*M+j ... (N,M)->N*M+M
                           where the grid or lattice is of dimension NxM.
                 gen - the generation in which this surface was generated
                 metric - the metric (to be optimized) associated with this surface
  """
  def __init__(self, topFile="topology.top", strucFile="structure.gro", LJTypes=[], gen=0, metric=None):
    self.topFile = topFile
    self.strucFile = strucFile
    self.LJTypes = np.array(LJTypes)
    self.gen = gen #Keeps track of generation this structure belongs to
    self.metric = None #Can also arbitrary metric with this structure!

  def makeCartoon(self, rows=8, cols=9):
    """Creates a cartoon (2D array of ones and zeros) representation of 
       the surface.
       Squares that are ones are attractive, and blank squares (zeros) are 
       repulsive LJ particles.
       This can easily be used to generate a heat-map.
       rows - (default 8) number of rows for chains represented in cartoon
       cols - (default 9) number of columns for chains represented in cartoon
    """
    cartoonPlot = np.zeros((rows, cols))
    for i in range(rows):
      for j in range(cols):
        if self.LJTypes[i*cols+j]:
          cartoonPlot[i,j] = 1.0

    return cartoonPlot


def combineLibraries(libList, libDirDict):
  """Combines structure libraries from different optimization runs, assigning unique structure
     and generation numbers to all individuals. WARNING: This is a utility function to be used
     outside of the main genetic algorithm code. The files and directories used here should be
     moved outside the working directory for the genetic algorithm optimization before being
     combined in any way. For safety, this code will refuse to overwrite anything.
     Inputs:
             libList - a list that orders that structure library files (when get keys, will be
                       randomly ordered); note that this should contain all the same strings as
                       libDirDict.keys(), just in a different order!
             libDirDict - a dictionary for which the keys are structure library (.pkl) files that
                          contain pickled lists of SurfStructure objects, and the definitions are 
                          the absolute paths to the directories associated with those libraries
     Outputs:
             newLibName - the absolute path of the newly-created library file
             newStrucDir - the absolute path of the newly-created structures directory
  """
  #Initiate list to hold all structures
  fullLib = []
  
  #Check to make sure won't overwrite anything
  newLibName = os.getcwd()+'/structure_library.pkl'
  if not os.path.isfile(newLibName):
    pass
  else:
    print "There is already a structure_library.pkl' file in the current location. Please move this safely to somewhere else and run this code again. Will not overwrite."
    sys.exit(2)

  #Create a new directory to hold the structures added to the new library
  newStrucDir = os.getcwd() + '/structures'
  try:
    os.mkdir(newStrucDir)
  except OSError:
    print "Directory already exists, will not overwrite, so exiting."
    sys.exit(2)

  #Start keeping track of which generation we're on... this is how we will uniquely name
  totGen = 0

  for aLib in libList:

    #Open the library
    with open(aLib, 'r') as inFile:
      thisLib = pickle.load(inFile)
    
    #Get the starting and ending generation for this library
    thisMinGen = min([aStruct.gen for aStruct in thisLib])
    thisMaxGen = max([aStruct.gen for aStruct in thisLib])

    #Loop over structures and copy files over with new names, updating the SurfStructure objects as go
    for aStruct in thisLib:
      
      #Create new object with new generation and add to list
      aGen = aStruct.gen
      newGen = totGen + (aGen - thisMinGen)
      newStruct = copy.deepcopy(aStruct)
      newStruct.gen = newGen
      newStruct.topFile = aStruct.topFile.replace('gen'+str(aGen), 'gen'+str(newGen))
      newStruct.strucFile = aStruct.topFile.replace('gen'+str(aGen), 'gen'+str(newGen))
      fullLib.append(newStruct)
      
      #Now copy over files to new names in new directory
      oldPrefix = aStruct.topFile.split('.top')[0]
      oldFiles = glob.glob(libDirDict[aLib]+'/'+oldPrefix+'*')
      for aFile in oldFiles:
        oldFileName = aFile.split('/')[-1]
        newFileName = oldFileName.replace('gen'+str(aGen), 'gen'+str(newGen))
        shutil.copy(aFile, newStrucDir+'/'+newFileName)

    totGen += (1 + thisMaxGen - thisMinGen) #Need the +1 to move to the next generation for the next library
    
  #At end, save fullLib as a new .pkl file, again checking to make sure don't overwrite
  newLibName = os.getcwd()+'/structure_library.pkl'
  if not os.path.isfile(newLibName):
    with open(newLibName, 'w') as outFile:
      pickle.dump(fullLib, outFile)
  else:
    print "There is already a structure_library.pkl' file in the current location. Please move this safely to somewhere else and run this code again. Will not overwrite."
    sys.exit(2)

  return newLibName, newStrucDir


def energyMinSurf(topFile, groFile, gromppFile):
  """Given a topology file and structure file, generates GROMACS energy minimized structure.
     Inputs:
             topFile - gromacs topology file (path from cwd)
             groFile - gromacs structure file (path from cwd)
             gromppFile - gromacs run-input file (path from cwd or absolute)
  """

  surfMinText = """
gmx -quiet grompp -f %s -c %s -p %s -po min_steep.mdp -o min_steep.tpr
gmx -quiet mdrun -nt 8 -nb cpu -pin on -deffnm min_steep
cp min_steep.gro %s
rm min_steep.*
            """ % (gromppFile, groFile, topFile, groFile)

  #Execute above bash commands to perform minimization
  subprocess.call(surfMinText, shell=True)


def mdRelaxSurf(topFile, groFile, gromppFile):
  """Given a topology file and structure file, relaxes structure with 50 ps of NVT simulation.
     Inputs:
             topFile - gromacs topology file (path from cwd)
             groFile - gromacs structure file (path from cwd)
             gromppFile - gromacs run-input file (path from cwd or absolute)
  """

  surfMinText = """
gmx -quiet grompp -f %s -c %s -p %s -po min_relax.mdp -o min_relax.tpr
gmx -quiet mdrun -nt 8 -nb cpu -pin on -deffnm min_relax
cp min_relax.gro %s
rm min_relax.*
            """ % (gromppFile, groFile, topFile, groFile)

  #Execute above bash commands to perform minimization
  subprocess.call(surfMinText, shell=True)


def mapIndices(topObj):
  """Given a topology object, finds all silanol oxygen indices 
     and indices of Si and hydrogen atoms bonded to these 
     oxygens.
     Inputs: 
             topObj - parmed topology object
     Outputs:
             atomsO - list of silanol oxygen atom objects (use for classifying silanols)
             indsO - list of silanol oxygen indices
             indsSi - list of silanol silica indices
             indsH - list of silanol hydrogen indices
             atomsOdict - dictionary mapping absolute oxygen indices to indices in indsO
  """

  atomsO = np.array([a for a in topObj.atoms if a.type == '19'])
  indsO = []
  atomsOdict = {} #Also make dictionary to convert from absolute indices in indsO to index in atomsO and indsO
  indsH = []
  indsSi = []

  for a in atomsO:

    indsO.append(a.idx)
    atomsOdict[a.idx] = len(indsO) - 1

    partners = a.bond_partners

    for part in partners:

      if part.type == '04':  #Have a hydrogen
        indsH.append(part.idx)

      elif part.type == '52':  #Have a silicon
        indsSi.append(part.idx)

  return atomsO, indsO, indsSi, indsH, atomsOdict


def classifySilanol(topObj, face='101'):
  """Classifies SiOH as single (isolated), geminal, or vicinal. Also outputs
     the density of each on the surface by calculating surface area available
     to krypton atoms. Usefully also returns lists of certain types of pairs
     of oxygens. This includes a list of hydrogen bonded pairs, silanol pairs 
     with Si-Si and O-O distances of less than 5.5 and 4.5 Angstroms (see 
     Ewing, et al, 2014), and a list of unique top two nearest neighbor pairs.
     Inputs:
             topObj - parmed topology object (surface slab) to classify
             face - (default '101') can be '101' (Q3, cristabolite) or '001' (Q2, quartz)
                    or 'amorph' (amorphous) which behaves the same as '001'
                        only matters in determining rules for finding nearest
                        pairs
     Outputs:
             isoDens - density of isolated OH
             vicDens - density of vicinal OH
             gemDens - density of geminal OH
             hbPairs - hydrogen-bonded silanol oxygen pairs
             distPairs - silanol oxygen pairs within criteria described above
             uniqueNear - unique silanol oxygen pairs for two nearest neighbors
  """

  #Get coordinates to work  and radii and boxdim
  coords = topObj.coordinates
  radii = np.array([a.rmin for a in topObj.atoms[:]])
  boxdim = topObj.box[0:3]

  #And map indices
  oxAtoms, oxInds, siInds, hInds, oxDict = mapIndices(topObj)

  #Define classification array - will use 'I', 'G', or 'V'
  #Assume all isolated until classified as vicinal or geminal
  silType = np.array(['I']*len(oxInds))

  #Also keep track of H-bond pairs (vicinal pairs)
  #Will be array with first column as the OVERALL atom index of the first partner
  #oxygen in the H-bonded pair, and second column as the second oxygen involved
  hbPairs = []

  #And also keep track of OH that are close together but not H-bonded...
  #According to Ewing, et al, 2014, should consider pairs with Si-Si separation
  #less than 5.5 and O-O separation less than 4.5.
  distPairs = []

  #AND also keep track of two nearest oxygens not on the same Si
  #May give more consistent assignment of condensation partners
  nearDists = np.ones(2*len(oxInds))*100.0
  nearPairs = np.ones((2*len(oxInds),2), dtype=int)*(-1)

  countGemPasses = 0
  countSiPasses = 0
  countAngPasses = 0

  #To classify as vicinal, figure out which are H-bonded
  #Will define vicinal silanols just like Ewing, et al, 2014, which uses hydrogen-bond
  #criteria: O-O distance of < 3.5 A and H-O-H(? must mean O-H-O) angle > 110 degrees.
  #Pretty permissive, but ok.
  for i in range(len(oxInds)):

    posox1 = coords[oxInds[i]]

    for j in range(len(oxInds)):

      posox2 = coords[oxInds[j]]

      #Need to image coordinates!
      distvec = wl.reimage([posox2], posox1, boxdim)[0] - posox1
      dist = np.linalg.norm(distvec)

      #Make sure not same oxygen and i and j not on same silicon atom (i.e. geminal)
      if j != i:
        if siInds[j] != siInds[i]:
          countSiPasses += 1
          #Also, to simplify things, make sure these two silanol Si atoms are not
          #connected via a single Si atom - this would be weird
          #Means cannot share any angle partners - also double-checks not geminal partners
          #But only check for this if worried about geminals!  For alpha-cristabolite, don't care!
          siAngPartners1 = topObj.atoms[siInds[i]].angle_partners
          siAngPartners2 = topObj.atoms[siInds[j]].angle_partners
          if (set(siAngPartners1).isdisjoint(siAngPartners2)) or (face == '101'):
            countAngPasses += 1 
            #Additionally, need to check if other hydroxyl on this geminal silanol (i) is closer to this 
            #partner (j). If it is, ignore this partner.
            oxbondsi1 = topObj.atoms[siInds[i]].bond_partners
            otherpos1 = np.zeros(3)
            for anox in oxbondsi1:
              if anox.type == '19' and anox.idx != oxInds[i]:
                otherpos1[:] = coords[anox.idx]
            #Make sure found a geminal partner, i.e. otherpos1 isn't still all zeros
            if np.any(otherpos1):
              otherdist1 = np.linalg.norm(wl.reimage([posox2], otherpos1, boxdim)[0] - otherpos1)
            else:
              #If no geminal partner, make otherdist1 really large so always include pair
              #Means that doesn't have to compete
              otherdist1 = 1E+24
            if dist < otherdist1:
              #Finally, need to make sure that geminal partner of j is not closer to i
              #than j itself... if this is true, also ignore this pairing
              oxbondsi2 = topObj.atoms[siInds[j]].bond_partners
              otherpos2 = np.zeros(3)
              for anox in oxbondsi2:
                if anox.type == '19' and anox.idx != oxInds[j]:
                  otherpos2[:] = coords[anox.idx]
              #Again make sure found a geminal partner
              if np.any(otherpos2):
                otherdist2 = np.linalg.norm(wl.reimage([otherpos2], posox1, boxdim)[0] - posox1)
              else:
                #If no geminal partner, doesn't compete so always include this pair
                otherdist2 = 1E+24
              if dist < otherdist2:
                countGemPasses += 1
                if dist < nearDists[2*i]:
                  nearDists[(2*i)+1] = nearDists[2*i]
                  nearPairs[(2*i)+1,:] = nearPairs[2*i,:]
                  nearDists[2*i] = dist
                  nearPairs[2*i,:] = np.array([oxInds[i], oxInds[j]])
                elif dist < nearDists[(2*i)+1]:
                  nearDists[(2*i)+1] = dist
                  nearPairs[(2*i)+1,:] = np.array([oxInds[i], oxInds[j]])

      if j > i:

        if dist < 3.5:
  
          #Now check O-H-O angle for each hydrogen
          angcut = np.cos(110.0*np.pi/180.0)
  
          posh1 = coords[hInds[i]]
  
          vecOh1 = wl.reimage([posh1], posox1, boxdim)[0] - posox1
          vecOh1 = vecOh1 / np.linalg.norm(vecOh1)
          vecHo1 = wl.reimage([posh1], posox2, boxdim)[0] - posox2
          vecHo1 = vecHo1 / np.linalg.norm(vecHo1)
  
          if np.dot(vecOh1, vecHo1) < angcut:
  
            #Have hydrogen bond
            silType[i] = 'V'
            silType[j] = 'V'
            hbPairs.append([oxInds[i], oxInds[j]])
  
          else:
  
            #Repeat with other hydrogen only if no bond found
            posh2 = coords[hInds[j]]
  
            vecOh2 = wl.reimage([posh2], posox2, boxdim)[0] - posox2
            vecOh2 = vecOh2 / np.linalg.norm(vecOh2)
            vecHo2 = wl.reimage([posh2], posox1, boxdim)[0] - posox1
            vecHo2 = vecHo2 / np.linalg.norm(vecHo2)
  
            if np.dot(vecOh2, vecHo2) < angcut:
  
              #Have hydrogen bond
              silType[i] = 'V'
              silType[j] = 'V'
              hbPairs.append([oxInds[i], oxInds[j]])
  
            else:
  
              #Check if Si-Si distance satisfies cutoff
              vecSiSi = wl.reimage([coords[siInds[j]]], coords[siInds[i]], boxdim)[0] - coords[siInds[i]]
  
              if (np.linalg.norm(vecSiSi) < 5.5) and (siInds[j] != siInds[i]):
                #Not H-bonded, but eligible pair to remove
                distPairs.append([oxInds[i], oxInds[j]])
  
        #Maybe not H-bonded, but use other distance criteria...
        elif dist < 4.5:
  
          #Check if Si-Si distance satisfies cutoff
          vecSiSi = wl.reimage([coords[siInds[j]]], coords[siInds[i]], boxdim)[0] - coords[siInds[i]]
  
          if (np.linalg.norm(vecSiSi) < 5.5) and (siInds[j] != siInds[i]):
            #Not H-bonded, but eligible pair to remove
            distPairs.append([oxInds[i], oxInds[j]])
    
  #Now that all vicinal classified, can differentiate remaining as geminal or isolated
  for (i, atom1) in enumerate(oxAtoms):

    if silType[i] != 'V':

      partners1 = atom1.bond_partners

      for (j, atom2) in enumerate(oxAtoms):

        if j != i:

          partners2 = atom2.bond_partners

          #If shares a bonded partner and isn't already vicinal, must be geminal
          #Note that if partner is vicinal, still count as geminal
          #I think this makes sense considering how spectroscopy works
          if not set(partners1).isdisjoint(partners2):
            silType[i] = 'G'
 
  #At some point, need to check for disparity between opposite faces of slab
  #Use below as cutoff for top and bottom of slab... as long as far from wrapping, works
  avgZ = np.average(coords[:,2]) #Z-coord center of geometry

  #Now compute surface area to compute densities
  #Note using krypton atom vdW radius to define accessible area
  #This matches with experiments, though to satisfy bonds on surfaces, used area
  #accessible to water!
  points = wl.spherepoints(1000)
  SASAper, surfAtomsBool = wl.spheresurfaceareas(coords, radii+2.02, points, 10, boxdim)
  totSASA = np.sum(SASAper) / 100.0 #Converted to nm^2
  hiSASA = np.sum([aper for (k, aper) in enumerate(SASAper) if coords[k,2] > avgZ ]) / 100.0 
  loSASA = np.sum([aper for (k, aper) in enumerate(SASAper) if coords[k,2] < avgZ ]) / 100.0

  isoInds = np.array([oxInds[i] for i in range(len(silType)) if silType[i] == 'I'])
  vicInds = np.array([oxInds[i] for i in range(len(silType)) if silType[i] == 'V'])
  gemInds = np.array([oxInds[i] for i in range(len(silType)) if silType[i] == 'G'])

  isoDens = len(isoInds) / totSASA
  vicDens = len(vicInds) / totSASA
  gemDens = len(gemInds) / totSASA

  print "Overall silanol densities for both faces of slab:"
  print "Isolated: %f " % isoDens
  print " Vicinal: %f " % vicDens
  print " Geminal: %f " % gemDens
  print "   Total: %f OH/nm^2" % (isoDens+vicDens+gemDens)
  print "Assumed total surface area of: %f nm^2\n" % totSASA

  hiisoInds = np.array([ind for ind in isoInds if coords[ind,2] > avgZ])
  loisoInds = np.array([ind for ind in isoInds if coords[ind,2] < avgZ])
  hivicInds = np.array([ind for ind in vicInds if coords[ind,2] > avgZ])
  lovicInds = np.array([ind for ind in vicInds if coords[ind,2] < avgZ])
  higemInds = np.array([ind for ind in gemInds if coords[ind,2] > avgZ])
  logemInds = np.array([ind for ind in gemInds if coords[ind,2] < avgZ])

  print "Upper face densities (atom above average z-coord):"
  print "Isolated: %f " % (len(hiisoInds)/hiSASA)
  print " Vicinal: %f " % (len(hivicInds)/hiSASA)
  print " Geminal: %f " % (len(higemInds)/hiSASA)
  print "   Total: %f OH/nm^2" % ((len(hiisoInds)+len(hivicInds)+len(higemInds))/hiSASA)
  print "Assumed total surface area of: %f nm^2\n" % hiSASA

  print "Lower face densities (atom below average z-coord):"
  print "Isolated: %f " % (len(loisoInds)/loSASA)
  print " Vicinal: %f " % (len(lovicInds)/loSASA)
  print " Geminal: %f " % (len(logemInds)/loSASA)
  print "   Total: %f OH/nm^2" % ((len(loisoInds)+len(lovicInds)+len(logemInds))/loSASA)
  print "Assumed total surface area of: %f nm^2\n" % loSASA

  hbPairs = np.array(hbPairs)

  distPairs = np.array(distPairs)

  #Want to make nearPairs unique so no pairs repeat themselves
  uniqueNear = []
  for i in range(len(nearPairs)):
      thisPair = np.sort(nearPairs[i])
      inunique = False
      for otherpair in uniqueNear:
        if np.array_equal(thisPair, np.sort(otherpair)):
          inunique = True
      if not inunique:
        uniqueNear.append(thisPair)
       
  uniqueNear = np.array(uniqueNear)

  return isoDens, vicDens, gemDens, hbPairs, distPairs, uniqueNear


def sortSpatialOH(pairList, topObj, face='101'):
  """Spatially sorts OH pairs as top/bottom surface and orders spatially.
     Note that spatial ordering is based off a grid, moving along x, then 
     y in the mapping.
     Inputs:
             pairList - list of pairs of indices for OH pairs
             topObj - parmed topology object corresponding to provided indices
             face - (default '101') can be '101' (Q3, cristabolite) or '001' (Q2, quartz)
                    or 'amorph' (amorphous)
     Outputs:
             sortedPairs - an array of sorted pairs that is of dimension Nx2x2;
                           have N corresponding pairs (N each for top and bottom), 
                           and 2 indices per pair for each top/bottom pair; NOTE
                           that to match up top and bottom pairs, need to map x,y
                           coordinates of average x,y coords of upper hydroxyl
                           pair to -x,y coords of lower hydroxyl pairs since the
                           lower surface is reflected compared to the upper.
             excList - an exclusion list between pairs showing which pairs cannot
                       simultaneously exist (i.e. because they use the same hydroxyl)
  """

  #Get box information for reimaging
  boxdim = topObj.box[0:3]

  #Get coordinates to associate with each pair, centering to make easier to work with
  coords = topObj.coordinates
  coords = coords - np.average(coords, axis=0)
  avgCoords = np.zeros((len(pairList),3))

  #Find average coordinate for each pair to use as location of this pair
  #As go, classify as upper or lower and store x,y coordinates for upper, -x,y for lower
  topinds = []
  topavg = []
  botinds = []
  botavg = []
  for k, pair in enumerate(pairList):
    #To work right, need to image!
    pos1 = coords[pair[0]]
    pos2 = coords[pair[1]]
    pos2 = wl.reimage([pos2], pos1, boxdim)[0]
    thisAvg = np.average(np.vstack((pos1, pos2)), axis=0)
    avgCoords[k,:] = thisAvg
    if thisAvg[2] > 0.0:
      topinds.append(k)
      topavg.append(thisAvg[:2])
    else:
      botinds.append(k)
      botavg.append(thisAvg[:2])

  topavg = np.array(topavg)
  botavg = np.array(botavg)
  #Coordinate mapping to match top and bottom different for different faces
  if face == '101':
    botavg[:,1] = botavg[:,1] + 2.5 #The surfaces are shifted by an offset...
    botavg[:,0] = (-1)*botavg[:,0] + 2.5  #And flipped AND shifted in x
  elif face == '001':
    botavg[:,0] = (-1)*botavg[:,0]
  elif face == 'amorph':
    pass
  else:
    print "Cyrstal face %s not recognized, please use \'101\' or \'001\' or \'amorph\'." % face
    sys.exit(2)

  #import matplotlib.pyplot as plt
  #plt.plot(topavg[:,0], topavg[:,1], 'b.')
  #plt.plot(botavg[:,0], botavg[:,1], 'r.')
  #plt.show()

  #Now pair up upper and lower by find lower -x,y coordinates that 
  #have lowest distance to upper x,y coordinates for each pair
  sortedPairs = []
  for i, toppos in enumerate(topavg):
    mindist = 100.0
    minind = -1
    for j, botpos in enumerate(botavg):
      imbotpos = wl.reimage([botpos], toppos, boxdim[:2])[0]
      dist = np.linalg.norm(toppos - imbotpos)
      if dist < mindist:
        mindist = dist
        minind = j
    sortedPairs.append(np.array([pairList[topinds[i]], pairList[botinds[minind]]]))

  #Now need to provide spatial ordering... won't use quadrants, just map 2D grid 
  #to a 1D array to get an order in which adjacent elements are spatially near to
  #each other
  binsx = np.arange(-boxdim[0]/2.0-2.0, boxdim[0]/2.0+2.0, 0.1)
  binsy = np.arange(-boxdim[1]/2.0-2.0, boxdim[1]/2.0+2.0, 1.0)
  mapxy = {}
  mapcount = 0
  for i in range(len(binsy)):
    for j in range(len(binsx)):
      mapxy['%i,%i' % (i,j)] = mapcount
      mapcount += 1
  #Just use coordinates of upper pairs
  sortBy = np.zeros(len(topavg))
  for k, pos in enumerate(topavg):
    xind = np.digitize(pos[0], binsx, right=True)
    yind = np.digitize(pos[1], binsy, right=True)
    sortBy[k] = mapxy['%i,%i' % (yind, xind)]

  #And sort
  sortedPairs = np.array([x for (y,x) in sorted(zip(sortBy.tolist(), sortedPairs))])

  #Now create Nx2 list of indices in sortedPairs show for each pair, which other pair is 
  #disallowed. Do for top and bottom surfaces and then check for consistency.
  topExc = np.array([[]]*len(sortedPairs)).tolist()
  botExc = np.array([[]]*len(sortedPairs)).tolist()

  for i, pairpair in enumerate(sortedPairs):
    for j in range(i+1, len(sortedPairs), 1):
      if not set(sortedPairs[j][0]).isdisjoint(pairpair[0]):
        topExc[i].append(j)
        topExc[j].append(i)
      if not set(sortedPairs[j][1]).isdisjoint(pairpair[1]):
        botExc[i].append(j)
        botExc[j].append(i)

  for k in range(len(topExc)):
    if set(topExc[k]).isdisjoint(botExc[k]):
      print "Top and bottom exclusions don't match:"
      print topExc[k]
      print botExc[k]
      print sortedPairs[k]
      print sortedPairs[topExc[k][:]]
      print sortedPairs[botExc[k][:]]

  #Will assume that above test for identical top and bottom exclusion lists passed...
  #It really should! At least it did for me.
  excList = topExc

  return sortedPairs, excList


def reduceSiOHDens(topObj, pairs, verbose=False):
  """Reduces silanol density by removing one silanol group and the
     hydrogen of a nearby silanol group, then bonding the oxygen to
     both Si that it should bridge.
     Inputs:
             topObj - parmed topology object to modify
             pairs - list of pairs of oxygen indices in hydroxyls to condense
             verbose - (default False) turn on/off verbose output
     Outputs:
             topObj - returns a modified topology object
  """

  #Need to find indices of silica and hydrogen atoms attached to each oxygen
  oxAtoms, oxInds, siInds, hInds, oxDict = mapIndices(topObj)

  toStrip = '@' #Construct AMBER-like atom selection mask

  bridgeList = [] #Will keep track of oxygens that should be newly bridging

  #Need to define atom_type of bulk oxygen atom, so find first on in topObj
  bulkOxType = None

  for a in topObj.atoms:

    if a.type == '18':

      bulkOxType = a.atom_type
      break

  #Also need 1-4 NonbondedException object type for O-Si-O-Si and O-Si-O-H cases
  #But note that must also include case for silanol O to Si in case one Si is geminal
  nonbondOSitype = None
  nonbondOHtype = None
  nonbondOhSitype = None

  for nbe in topObj.adjusts:

    if nonbondOSitype is None:

      if nbe.atom1.type == '18' and nbe.atom2.type == '52':
        nonbondOSitype = nbe.type

      elif nbe.atom1.type == '52' and nbe.atom2.type == '18':
        nonbondOSitype = nbe.type

    if nonbondOHtype is None:

      if nbe.atom1.type == '18' and nbe.atom2.type == '04':
        nonbondOHtype = nbe.type

      elif nbe.atom1.type == '04' and nbe.atom2.type == '18':
        nonbondOHtype = nbe.type

    if nonbondOhSitype is None:

      if nbe.atom1.type == '19' and nbe.atom2.type == '52':
        nonbondOhSitype = nbe.type

      elif nbe.atom1.type == '52' and nbe.atom2.type == '19':
        nonbondOhSitype = nbe.type

  #Loop over pairs and build list of atoms to delete
  #Will add bonds as go, then delete atoms at end
  for apair in pairs:

    #Get list indices of both oxygens in pair
    ind1 = oxDict[apair[0]]
    ind2 = oxDict[apair[1]]
    
    #Will make first oxygen in pair into bridge, delete the other
    #Add hydrogen to strip string - note that atom number is index + 1
    toStrip = toStrip + '%i,' % (hInds[ind1] + 1)

    bridgeList.append(topObj.atoms[apair[0]])

    #Change charge of this oxygen
    pmd.tools.change(topObj, 'CHARGE', '@%i'%(apair[0] + 1), -0.55).execute()
    
    #Change atom type of this oxygen
    #Have to set type and atom_type separately!
    pmd.tools.change(topObj, 'ATOM_TYPE', '@%i'%(apair[0] + 1), '18').execute()
    topObj.atoms[apair[0]].atom_type = bulkOxType

    #Create bond to silicon currently attached to second oxygen
    pmd.tools.setBond(topObj, '@%i'%(apair[0] + 1), '@%i'%(siInds[ind2] + 1), 285.000, 1.680).execute()

    #And new Si-O-Si angle
    pmd.tools.setAngle(topObj, '@%i'%(siInds[ind1] + 1), '@%i'%(apair[0] + 1), 
                                             '@%i'%(siInds[ind2] + 1), 100, 149.0).execute()

    #And loop through bond partners of already attached Si and add as dihedrals to newly-attached Si
    for sipartner in topObj.atoms[siInds[ind1]].bond_partners:

      if ( not (sipartner is topObj.atoms[apair[0]]) and
           not (sipartner is topObj.atoms[apair[1]]) ):

        pmd.tools.addDihedral(topObj, '@%i'%(siInds[ind2] + 1), '@%i'%(apair[0] + 1), 
                                         '@%i'%(siInds[ind1] + 1),  '@%i'%(sipartner.idx + 1), 
                                                                  0.0, 0.0, 0.0, 1.0, 1.0).execute()

        if sipartner.type == '18':
          topObj.adjusts.append(pmd.topologyobjects.NonbondedException(topObj.atoms[siInds[ind2]],
                                                                             sipartner, type=nonbondOSitype))

        elif sipartner.type == '19':
          topObj.adjusts.append(pmd.topologyobjects.NonbondedException(topObj.atoms[siInds[ind2]],
                                                                             sipartner, type=nonbondOhSitype))

        else:
          print "Seems that Si atom %i has 1-4 partner that is type %s (index %i); weird." % (siInds[ind2], sipartner.type, sipartner.idx)

    #When create new bond, turns out also need to define new angles and dihedrals...
    #parmed does not do this automatically!
    #Will follow bond graph of second Si
    for sipartner in topObj.atoms[siInds[ind2]].bond_partners:

      if ( not (sipartner is topObj.atoms[apair[0]]) and 
           not (sipartner is topObj.atoms[apair[1]]) ):

        #Since defining new angle for O bonded to Si, know must be O-Si-O angle
        #Si isn't bonded to anything else!
        pmd.tools.setAngle(topObj, '@%i'%(apair[0] + 1), '@%i'%(siInds[ind2] + 1), 
                                                '@%i'%(sipartner.idx + 1), 100, 109.5).execute()

        #Need to add all atoms forming angle with bridging O to list of dihedrals with initially bonded Si
        pmd.tools.addDihedral(topObj, '@%i'%(siInds[ind1] + 1), '@%i'%(apair[0] + 1), 
                                         '@%i'%(siInds[ind2] + 1),  '@%i'%(sipartner.idx + 1), 
                                                                  0.0, 0.0, 0.0, 1.0, 1.0).execute()

        if sipartner.type == '18':
          topObj.adjusts.append(pmd.topologyobjects.NonbondedException(topObj.atoms[siInds[ind1]],
                                                                             sipartner, type=nonbondOSitype))

        elif sipartner.type == '19':
          topObj.adjusts.append(pmd.topologyobjects.NonbondedException(topObj.atoms[siInds[ind1]],
                                                                             sipartner, type=nonbondOhSitype))

        else:
          print "Seems that Si atom %i has 1-4 partner that is type %s (index %i); weird." % (siInds[ind1], sipartner.type, sipartner.idx)

        topObj.adjusts.append(pmd.topologyobjects.NonbondedException(topObj.atoms[siInds[ind1]],
                                                                             sipartner, type=nonbondOSitype))

        #And also loop through this atoms neighbors and add dihedral definitions
        #Necessary to get 1-4 interactions defined correctly! 
        for otherpartner in topObj.atoms[sipartner.idx].bond_partners:

          if not (otherpartner is topObj.atoms[siInds[ind2]]):

            #All dihedral terms zero in this force field
            pmd.tools.addDihedral(topObj, '@%i'%(apair[0] + 1), '@%i'%(siInds[ind2] + 1),  
                                         '@%i'%(sipartner.idx + 1), '@%i'%(otherpartner.idx + 1), 
                                                                     0.0, 0.0, 0.0, 1.0, 1.0).execute()

            #And add to 1-4 pair list
            if otherpartner.type == '52':
              topObj.adjusts.append(pmd.topologyobjects.NonbondedException(topObj.atoms[apair[0]], 
                                                                             otherpartner, type=nonbondOSitype))

            elif otherpartner.type == '04':
              topObj.adjusts.append(pmd.topologyobjects.NonbondedException(topObj.atoms[apair[0]], 
                                                                             otherpartner, type=nonbondOHtype))

            else:
              print "Somehow O atom %i has 1-4 partner of %i (%s) that is not Si or H... weird." % (apair[0], otherpartner.idx, otherpartner.type)

    #Add second oxygen in pair and its hydrogen to strip string
    toStrip = toStrip + '%i,' % (oxInds[ind2] + 1)
    toStrip = toStrip + '%i,' % (hInds[ind2] + 1)

  toStrip = toStrip[:-1] #Get rid of trailing comma

  print toStrip

  #Finally, strip unwanted atoms
  pmd.tools.strip(topObj, toStrip).execute()

  if verbose:
    #And get new information on added bonds and adjusted atoms...
    print "Information on newly bridging oxygen atoms:"

    for (i, a) in enumerate(bridgeList):

      print "  Atom %i of type %s/%s, name %s, and charge %f:" % (a.idx, a.type, a.atom_type, a.name, a.charge)
      print "    Bonds    : %s" % (str(a.bond_partners))
      print "    Angles   : %s" % (str(a.angle_partners))
      print "    Dihedrals: %s" % (str(a.dihedral_partners))
      print "    1-4 Pairs: %s" % (str(a.nonbonded_exclusions(only_greater=False, index_from=0)))

  return topObj


def createQuartzSurf(fullTopObj, OHPairs, exclPairs, condPairs, filePrefix, face='101'):
  """Given a surface that is fully hydroxylated, generates a new surface with specified
     silanol pairs replaced with bridging oxygens. Saves new topology and 
     structure files.
     Inputs:
             fullTopObj - parmed topology object for fully hydroxylated surface
             OHPairs - array of arrays of eligible OH pair atom indices
             exclPairs - array of OHPairs indices that exclude each other 
             condPairs - same first two dimensions OHPairs, but a boolean type
                         array that describes which pairs should be condensed
             filePrefix - prefix used to name .top and .gro files that are saved
             face - (default '101') crystallographic face, either '101' (crystabolite Q3) 
                    or '001' (quartz Q2), or 'amorph' (amorphous)
     Outputs:
             newStruct - returns a SurfStructure object
  """

  topObj = fullTopObj.copy(pmd.gromacs.GromacsTopologyFile)

  #Need to generate set of pairs that can easily be used by reduceSiOHDens
  #Should be just a list of pairs, but alternate corresponding upper and lower
  #And want only those selected, so do this before alternating top and bottom
  OHPairsFlat = copy.deepcopy(OHPairs)

  condInds = np.arange(len(OHPairsFlat))
  condInds = condInds[condPairs]

  OHPairsFlat = np.vstack((OHPairsFlat[condInds]))
 
  #And remove the desired silanol pairs
  topObj = reduceSiOHDens(topObj, OHPairsFlat, verbose=False)

  #Save topology and structure files
  topObj.save(filePrefix+'.top', combine='all')
  topObj.save(filePrefix+'.gro')

  #If it's an amorphous surface, need to add appropriate position restraints
  #The below works because I know exactly where the surface is placed in the box for the
  #reference fully hydroxylated surface... otherwise would be harder
  if face == 'amorph':
    coords = topObj.coordinates
    restbool = (coords[:,2] >= -5.0) * (coords[:,2] <= 5.0)
    restatoms = np.arange(len(restbool))[restbool] + 1 #Add one to index correctly in top file
    topfile = open(filePrefix+'.top', 'r')
    toplines = topfile.readlines()
    topfile.close()
    insertline = toplines.index('[ system ]\n')
    toplines.insert(insertline, ' \n')
    for arest in restatoms:
      toplines.insert(insertline, '  %4i  1      10000.0      10000.0     10000.0 \n' % arest)
    toplines.insert(insertline, '; ai    funct   fc\n')
    toplines.insert(insertline, '[ position_restraints ]\n')
    topfile = open(filePrefix+'.top', 'w')
    topfile.write("".join(toplines))
    topfile.close()

  #Perform quick energy minimization of just surface
  energyMinSurf(filePrefix+'.top', filePrefix+'.gro', '/home/jmonroe/Silanol_Density_Project/diff_optimization/inputMD/grompp_steep.mdp')

  #Get generation from file prefix (make naming redundant)
  gen = int(filePrefix.split('gen')[1].split('_')[0])

  newStruct = SurfStructure(topFile=filePrefix+'.top', strucFile=filePrefix+'.gro', 
                                  OHPairs=OHPairs, exclPairs=exclPairs, condPairs=condPairs, gen=gen, face=face)

  return newStruct


def createSAMSurf(structList, chainTypeList, filePrefix, chainsX=8, chainsY=9, latticea=4.97, doMinRelax=False):
  """Creates a surface composed of two chain types according to boolean array.
     Inputs:
             structList - arbitrary length list of parmed topology objects for the chain types
                          (for consistency with past results, put methane-terminated at zero position
                          and hydroxyl-terminated at one position)
                          Generally, order in structList (i.e. indexing) MUST match non-negative integers
                          in chainTypeList, i.e. if have a 2 in chainTypeList, should match structList[2]
             chainTypeList - integer array specifiying methane-terminated with 0, hydroxyl with 1
                             (assumes mapping of 0->(0,0); 1->(0,1); 2->(0,2); ... ;N*M+M->(N,M), etc.
                             With updates, can now handle an arbitrary number of chain types numbered 0
                             through the number of unique headgroup chemistries
             filePrefix - prefix used to name .top and .gro files that are saved
             chainsX - (default 8) number of chains in the x dimension
             chainsY - (default 9) number of chains in the y dimension
             latticea - (default 4.97) lattice spacing in Angstroms
             doMinRelax - (default False) whether or not to perform energy min and MD relaxation with GROMACS
     Outputs:
             newStructObj - new (optionally) energy minimized and MD-relaxed SAMStructure object
  """
  if len(structList) != np.max(chainTypeList)+1:
    print "To make surface, list of structure objects must match number of chain types."
    sys.exit(2)

  if len(chainTypeList) != chainsX*chainsY:
    print "Length of boolean array must match total number of chains!"
    sys.exit(2)

  #Get the names of the chains we're working with
  chainNames = ['']*len(structList)
  for k in range(len(structList)):
    chainNames[k] = structList[k].residues[0].name

  #Set box z dimension
  boxZdim = 60.0

  #Create new topology object to add chains to
  newSurf = copy.deepcopy(structList[chainTypeList[0]])

  #Make sure this first chain has a sulfur at the origin
  sulfInd = [a.idx for a in newSurf.atoms if a.name == 'SU'][0]
  sulfPos = newSurf.coordinates[sulfInd]
  newSurf.coordinates = newSurf.coordinates - sulfPos

  #And shift in z-direction towards center of box
  newSurf.coordinates = newSurf.coordinates + np.array([0.0, 0.0, boxZdim/2.0])

  #Loop over chains and specify positions for each
  for i in range(chainsX):
    for j in range(chainsY):
      if i == 0 and j == 0:
        continue
      tempChain = copy.deepcopy(structList[chainTypeList[i*chainsY+j]])
      sulfInd = [a.idx for a in tempChain.atoms if a.name == 'SU'][0]
      sulfPos = tempChain.coordinates[sulfInd]
      tempChain.coordinates = tempChain.coordinates - sulfPos
      #Change x-spacing if even or odd row (in y-dimension)
      if j%2 == 0:
        tempX = latticea*i
      else:
        tempX = latticea*np.cos(np.pi/3.0) + latticea*i
      tempY = latticea*np.sin(np.pi/3.0)*j
      tempMoveBy = np.array([tempX, tempY, boxZdim/2.0])
      tempChain.coordinates = tempChain.coordinates + tempMoveBy
      newSurf += tempChain

  #Set box correctly
  newSurf.box = [chainsX*latticea, chainsY*latticea*np.sin(np.pi/3.0), boxZdim, 90.0, 90.0, 90.0]

  #Save the new .gro and .top files
  #newSurf.save(filePrefix+'.top')#, combine='all', parameters='inline')
  newSurf.save(filePrefix+'.gro')

  #Due to bug in parmed, need to fill in scale factors in [ pairs ] section
  #subprocess.call("sed -i -e \"s/\([0-9]\)     1$/\\1     1    0.00e+00    0.00e+00 /\" %s.top"%filePrefix, shell=True)

  #Due to a limitation of parmed, also need to add in the constaints and position restraint information
  #Below takes care of position restraints, but constraints vary with each type of chain due to atom numbering
  #subprocess.call("sed -i \'/\[ angles/i \
#\[ position_restraints \] \
#; ai   funct   fc \
#  1     1      1000.0     1000.0     1000.0 ; restrains SU to a point \
#     \' %s.top"%filePrefix, shell=True)

  #Won't be stand-alone and quite as descriptive, but easier to just use template...
  #And below is the template used for paper optimizing surface affinities
  #This is super lazy coding, but it's easy enough to fix if you want to
  topTemplate = """;
;
;   File created manually with the createSAMSurf() function of genetic_lib.py
;
;

[ defaults ]
; nbfunc        comb-rule       gen-pairs       fudgeLJ fudgeQQ
1               2               yes             0.5     0.833333

; Use force field parameters 
#include "/home/jmonroe/Surface_Affinities_Project/FFfiles/ffnonbonded.itp"
#include "/home/jmonroe/Surface_Affinities_Project/FFfiles/ffbonded.itp"

; Include methyl-terminated chain (CTM)
#include "/home/jmonroe/Surface_Affinities_Project/FFfiles/ctmSAM.itp"

; Include hydroxyl-terminated chain (OTM)
#include "/home/jmonroe/Surface_Affinities_Project/FFfiles/otmSAM.itp"

; Include sulfonate-terminated chain (STM)
#include "/home/jmonroe/Surface_Affinities_Project/FFfiles/stmSAM.itp"

; Include quaternary-ammonium-terminated chain (NTM)
#include "/home/jmonroe/Surface_Affinities_Project/FFfiles/ntmSAM.itp"

; Include all solutes of interest
#include "/home/jmonroe/Surface_Affinities_Project/FFfiles/solutes.itp"

; Use TIP4P-Ew water model 
#include "/home/jmonroe/Surface_Affinities_Project/FFfiles/tip4pew.itp" 

[ system ]
; Name
Generic title

[ molecules ]
; Compound       #mols"""

  #Now need to loop over the chainTypeList to add order of chains at end
  for val in chainTypeList:
    topTemplate = topTemplate+"\n%s                 1"%(chainNames[val])

  topTemplate = topTemplate+"\n"

  #And save new topology file
  with open(filePrefix+'.top', 'w') as outTop:
    outTop.write(topTemplate)

  #Perform energy minimization and MD relaxation if desired (with GROMACS)
  if doMinRelax:
    energyMinSurf(filePrefix+'.top', filePrefix+'.gro', '/home/jmonroe/Silanol_Density_Project/diff_optimization/inputMD/grompp_steep.mdp')

    #And also perform brief MD relaxation (50 ps in NVT) to allow chains to relax and tilt
    mdRelaxSurf(filePrefix+'.top', filePrefix+'.gro', '/home/jmonroe/Silanol_Density_Project/diff_optimization/inputMD/grompp_relax.mdp')

  #Get generation from file prefix (make naming redundant)
  gen = int(filePrefix.split('gen')[1].split('_')[0])

  newStructObj = SAMStructure(topFile=filePrefix+'.top', strucFile=filePrefix+'.gro', 
                                                            chainTypes=chainTypeList, gen=gen)

  return newStructObj


def createToySurf(LJTypeList, filePrefix, templateStructObj, templateTopStr, oldType='Mr', oldRes='MR', newType='Ma', newRes='MA'):
  """Creates a surface composed of two LJ atom types according to boolean array.
     Inputs:
             LJTypeList - boolean array specifiying repulsive with 0, attractive LJ with 1
                          (assumes mapping of 0->(0,0); 1->(0,1); 2->(0,2); ... ;N*M+M->(N,M), etc.
             filePrefix - prefix used to name .top and .gro files that are saved
             templateStructObj - template stucture parmed object that will have atom types replaced according to LJTypeList
             templateTopStr - template topology file string that will have LJTypeList sequence added to it
             oldType - (default 'Mr') the atom type in the template to be replaced
             oldRes - (default 'MR') the residue type in the template to be replaced
             newType - (default 'Ma') the new atom type associated with a 1 in LJTypeList
             newRes - (default 'MA') the new residue type associated with a 1 in LJTypeList
     Outputs:
             newStructObj - new ToyStructure object with associated topology and structure files written
  """
  #First load the template structure with parmed
  tempStruct = copy.deepcopy(templateStructObj)

  #Also read in the topology file template as a standard file
  tempTop = copy.deepcopy(templateTopStr)

  #Replace old atom types and residue types with new atom type and residue type at the true positions in LJTypeList
  #At the same time, add to what will go at the end of the topology file to correctly specify the order
  topOrder = ''
  for ind, val in enumerate(LJTypeList):
    if val:
      #Do upper layer of surface
      tempStruct.atoms[ind].name = newType
      tempStruct.residues[ind].name = newRes
      #And also do bottom layer of surface the same way
      tempStruct.atoms[-(len(LJTypeList) - ind)].name = newType
      tempStruct.residues[-(len(LJTypeList) - ind)].name = newRes
      topOrder = topOrder+"%s                  1\n"%newRes
    else:
      topOrder = topOrder+"%s                  1\n"%oldRes
  
  #Finish off the topology file
  tempTop = tempTop+topOrder
  tempTop = tempTop+"%s                %i\n"%(oldRes, len(tempStruct.atoms) - 2*len(LJTypeList))
  tempTop = tempTop+topOrder

  #Save the new .gro file
  tempStruct.save(filePrefix+'.gro')

  #And save new topology file
  with open(filePrefix+'.top', 'w') as outTop:
    outTop.write(tempTop)

  #Get generation from file prefix (make naming redundant)
  gen = int(filePrefix.split('gen')[1].split('_')[0])

  newStructObj = ToyStructure(topFile=filePrefix+'.top', strucFile=filePrefix+'.gro', 
                                                            LJTypes=LJTypeList, gen=gen)

  return newStructObj


def genMutationsQuartz(Nmod, oldSites, structInds, exclList):
  """Generates random mutations (i.e. picks hydroxyls to condense)
     Inputs:
             Nmod - number of mutations to make
             oldSites - old pairs that are currently condensed
             structInds - all possible pair indices for the structure to mutate
             exclList - the excluded pair list for the structure to mutate
     Outputs:
             newSites - new boolean array for tagging which hydroxyls to condense
  """
  newSites = copy.deepcopy(oldSites)

  #Check if is fully hydroxylated surface or has some hydroxyls condensed
  if np.any(newSites):

    #First pick Nmod site that are currently condensed to become hydroxylated
    #Can pick without restriction because assume that every time do condensation
    #step, pick a unique set of hydroxyl pairs
    hydSites = np.random.choice(structInds[newSites], size=Nmod, replace=False)
    newSites[hydSites] = False

  #If no condensations yet, have fully-hydroxylated surface
  #Either way, perform Nmod hydroxylations now

  #Now perform hydroxylation step by picking unique set of pairs
  #Note that could end up with no mutations, technically (but unlikely)
  allCondSites = np.random.choice(structInds[np.invert(newSites)], 
                                     size=np.sum(np.invert(newSites)), replace=False)
  condExcl = exclList[newSites].flatten()
  condSites = []
  condCount = 0
  for ind in allCondSites:
    if ind not in condExcl:
      condSites.append(ind)
      #If condense this site, add its exclusions to exclusion list
      condExcl = np.hstack((condExcl, exclList[ind].flatten()))
      condCount += 1
    if condCount >= Nmod:
      break
  newSites[condSites] = True

  return newSites


def genMutationsSAM(Nmut, oldChains, numChains):
  """Generates mutations randomly for SAM surfaces
     Inputs:
             Nmut - array of number of mutations to make for each NON-ZERO chain type
                    Type zero chains will be treated as "background" with
                    mutations only made between these and other types, not
                    between other chains types represented by larger integers.
                    Note that if Nmut is only a single number but there are more
                    than one non-zero chain type, only the first non-zero chain type
                    will be mutated!
             oldChains - old chainTypes array, or an array of length zero
                         If length zero, generates chainTypeList array from scratch
             numChains - total number of chains in surface 
     Outputs:
             newChains - new integer array specifying chain types
  """
  #Caste to array just in case only one chain type and so given int or float
  #Have to flatten in case given list or array
  #Mainly maintains back-compatibility
  Nmut = np.array([Nmut]).flatten() 

  if len(oldChains) == 0:
    #Want to generate chainTypeList from scratch
    newChains = np.zeros(numChains, dtype=int)
    for k in range(len(Nmut)):
      thisZeroInds = np.where(newChains == 0)[0]
      thisInds = np.random.choice(thisZeroInds, size=Nmut[k], replace=False)
      newChains[thisInds] = k+1

  else:
    #Start with copy of original
    newChains = copy.deepcopy(oldChains)
    #Loop over non-zero chain types and mutate with zero chain types
    #Make sure to update which chains are zero each time!
    for k in range(len(Nmut)):
      #Pick Nmut[k] of type k+1 to turn into type zero
      currNonInds = np.where(newChains == (k+1))[0]
      changeToZeroInds = np.random.choice(currNonInds, size=Nmut[k], replace=False)
      newChains[changeToZeroInds] = 0
      #Then Nmut[k] zeros to type k+1
      currZeroInds = np.where(newChains == 0)[0]
      changeToNonInds = np.random.choice(currZeroInds, size=Nmut[k], replace=False)
      newChains[changeToNonInds] = k+1

  return newChains 


def genMutationsToy(Nmut, oldLJTypes, numLJ):
  """Generates mutations randomly for toy LJ surfaces
     Inputs:
             Nmut - number of mutations to make
             oldLJTypes - old LJTypes array, or an array of length zero
                         If length zero, generates random array from scratch
             numLJ - total number of LJ particles on surface
     Outputs:
             newLJTypes - new boolean array for showing which particles are attractive
  """
  #Create index array for particles
  LJIndRange = np.arange(numLJ)

  newLJTypes = []

  if len(oldLJTypes) == 0:
    thisInds = np.random.choice(LJIndRange, size=Nmut, replace=False)
    newLJTypes = np.zeros(numLJ, dtype=bool)
    newLJTypes[thisInds] = True
   
  else:
    #Pick Nmut attractive particles to switch to repulsive and vice versa
    currAttrInds = LJIndRange[oldLJTypes]
    changeToRepInds = np.random.choice(currAttrInds, size=Nmut, replace=False)
    newLJTypes = copy.deepcopy(oldLJTypes)
    newLJTypes[changeToRepInds] = False
    currRepInds = LJIndRange[np.invert(newLJTypes)]
    changeToAttrInds = np.random.choice(currRepInds, size=Nmut, replace=False)
    newLJTypes[changeToAttrInds] = True
 
  return newLJTypes 


def genSurfsQuartz(fullOHTop, Nsurf, Nmod, structList, generation, numCond, doMerge=True, face='101'):
  """Generates random surfaces given a set of input structure files.
     structList may either be the same length as Nsurf, or of length
     1. If it is of length 1, the single surface provided is mutated
     Nsurf times with Nmod mutations each time. numCond only matters 
     if the Q2 surface is provided as one of the structures. doMerge
     is only relevant if structList is of length Nsurf, in which case
     Nsurf children will be randomly generated before mutations.
     Inputs:
             fullOHTop - topology object for fully hydroxylated surface  
             Nsurf - number of surfaces to generate - must be even
             Nmod - number of modifications to make to each surface 
                    (i.e. number of hydroxyl pairs to attempt to condense)
             structList - list of SurfStructure objects
             generation - the current generation (for naming correctly)
             numCond - desired number of surface hydroxyl pairs to condense 
                     to reach desired density - only used if no parents, 
                     i.e. just Q2 surface structure provided
             doMerge - (default True) merges surfaces to create children
             face - (default '101') crystallographic face
     Outputs: 
             newSurfs - new list of SurfStructure objects
  """
  #Check length of structList
  if (len(structList) != Nsurf):
    if (len(structList) != 1):
      print "The length of structList must match Nsurf or 1. See help for more info."
      sys.exit(2)

  if Nsurf%2 != 0:
    print "Number of surfaces to generate must be even so that can pair if desired."
    sys.exit(2)

  savePrefix = "structures/gen%i" % generation

  newSurfs = []

  if len(structList) == 1:

    struct = structList[0]
    structInds = np.arange(len(struct.OHPairs))

    for i in range(Nsurf):
      newPrefix = "%s_%i" % (savePrefix, i)
      #Generate random mutations if has any condensed pairs (bridging oxygens)
      if np.any(struct.condPairs):
        newSites = genMutationsQuartz(Nmod, struct.condPairs, structInds, struct.exclPairs)
      else:
        #If don't have any 'True's in condPairs, then have fully hydroxylated surface
        newSites = genMutationsQuartz(numCond, struct.condPairs, structInds, struct.exclPairs)
      #Create a new surface
      newSurfs.append(createQuartzSurf(fullOHTop, struct.OHPairs, struct.exclPairs, 
                                                              newSites, newPrefix, face=face))

  elif len(structList) == Nsurf:

    if doMerge:

      #Uniquely pair up parents randomly by randomly ordering structList
      parentInds = np.random.choice(np.arange(len(structList)), size=len(structList), replace=False)

      for i in range(Nsurf/2):

        struct1 = structList[parentInds[i*2]]
        struct2 = structList[parentInds[i*2+1]]
        parentStructs = [[struct1, struct2], [struct2, struct1]]
        structInds = np.arange(len(struct1.OHPairs))

        #Know that both parents must have same number of condensed pairs and same number of hydroxylated pairs
        #So pick random number between zero and one
        #Take that fraction of currently True (condensed) pair indices from parent 1
        #Take a non-overlapping set of pair indices that are True from parent 2
        #Since sortSpatialOH roughly orders list of pairs by spatial location, sets from both parents
        #should be pairs that are close to each other... though this mainly only holds for parent 1 as
        #the requirement of a unique set (to maintain constant density) may mess up such correlations
        #At end, mutate a little
        #After done, flip the parents but keep the same random number 
        arand = np.random.random()
        for j, structs in enumerate(parentStructs):

          newSites = np.zeros(len(structs[0].OHPairs), dtype=bool)

          if np.sum(structs[0].condPairs) != np.sum(structs[1].condPairs):
            print "Somehow have different silanol densities on provided structures. Quitting."
            sys.exit(2)

          samprand1 = int(arand*np.sum(structs[0].condPairs))
          samprand2 = np.sum(structs[1].condPairs) - samprand1
          newCond1 = structInds[structs[0].condPairs][:samprand1]
          newSites[newCond1] = True
          exclCondSites = structs[0].exclPairs[newCond1].flatten()

          newCond2 = np.array([], dtype=int)
          for ind in structInds[structs[1].condPairs]:
            if ind not in newCond1:
              #Now also check that this ind is not in exlcusion list of newCond1...
              if ind not in exclCondSites:
                newCond2 = np.hstack((newCond2, ind))
                exclCondSites = np.hstack((exclCondSites, structs[1].exclPairs[ind].flatten()))

          #Now check if enough pair indices in newCond2 - if not, loop through remaining pairs and randomly pick
          mutForDens = samprand2 - len(newCond2)
          if len(newCond2) < samprand2:
            print "Due to exclusions for pair condensations, cannot maintain density with parent 2 alone."
            print "Pair indices that DID work were:"
            print newCond2
            print "Picking %i random pairs to condense." % (mutForDens)
            newSites[newCond2] = True
            cond2Count = len(newCond2)
            availCondSites = np.random.choice(structInds[np.invert(newSites)], 
                                                 size=np.sum(np.invert(newSites)), replace=False)
            for k, ind in enumerate(availCondSites):
              if ind not in exclCondSites:
                newCond2 = np.hstack((newCond2, ind))
                exclCondSites = np.hstack((exclCondSites, structs[0].exclPairs[ind].flatten()))
                cond2Count += 1
              if cond2Count >= samprand2:
                break
          else:
            newCond2 = newCond2[-samprand2:]
          newSites[newCond2] = True

          #And now mutate this surface, but only if mutated fewer than Nmod pairs to maintain density
          #If did, just mutate up until get to Nmod. If didn't mutate fewer, just mutate 1.
          if mutForDens >= Nmod:
            thisMod = 1
          else:
            thisMod = Nmod - mutForDens

          if j == 0:
            newPrefix = "%s_%i" % (savePrefix, i*2)
          elif j == 1:
            newPrefix = "%s_%i" % (savePrefix, i*2+1)

          #And generate some mutations!
          newSites = genMutationsQuartz(thisMod, newSites, structInds, structs[0].exclPairs)
          newSurfs.append(createQuartzSurf(fullOHTop, structs[0].OHPairs, structs[0].exclPairs, 
                                                                           newSites, newPrefix, face=face))
        
    else:

      for i in range(Nsurf):
        struct = structList[i]
        structInds = np.arange(len(struct.OHPairs))
        #Go straight into mutating each surface, don't pair parents
        newSites = genMutationsQuartz(Nmod, struct.condPairs, structInds, struct.exclPairs)
        newPrefix = "%s_%i" % (savePrefix, i)
        newSurfs.append(createQuartzSurf(fullOHTop, struct.OHPairs, struct.exclPairs, 
                                                                           newSites, newPrefix, face=face))

  #And return new list of SurfStructure objects
  return newSurfs


def genSurfsSAM(chainList, Nsurf, Nmod, parentStructs, generation, numNon, chainsX, chainsY, doMerge=True, constantDens=True):
  """Generates SAM surfaces given set of input files.
     If parentStructs is zero in length, randomly
     generates surfaces with specified number of other chain types.
     If parentStructs has members, will only do cross-over
     step if doMerge is true. Otherwise, just generates
     mutations of each and returns.
     Inputs:
             chainList - list of various headgroup SAM parmed topology objects
             Nsurf - number of surfaces to generate and return
             Nmod - number of modifications (mutations) for each surface
                    If more than two chain types, should be an array
             parentStructs - a list of SAMStructure objects to use as parents
                             for the next generation; this MUST be same length
                             as Nsurf, or be zero in length.
             generation - the generation for these surfaces
             numNon - array of numbers of non-zero type SAM chains in the surface
                      As with Nmod, this could also just be an integer or float
                      if there are only two chain types, OR you just want the same
                      number of mutations and chain types for all non-zero types
             chainsX - (default 8) number of chains in x-dimension of surface
             chainsY - (default 9) number of chains in y-dimension of surface
             doMerge - (default True) if True, performs cross-over with parents
                       in parentStructs; otherwise goes straight to mutations
             constantDens - (default True) if True, enforces constant surface density
    Outputs:
             newSurfs - a list of length Nsurf containing new SAMStructure objects
  """
  #Check length of parentStructs
  if (len(parentStructs) != Nsurf):
    if (len(parentStructs) != 0):
      print("The length of parentStructs must match Nsurf or 0. See help for more info.")
      sys.exit(2)

  if Nsurf%2 != 0:
   print("Number of surfaces to generate must be even so that can pair if desired.")
   sys.exit(2)

  #Make sure Nmod is an array - if only two surface types and given int or float, caste it
  #Also, check if length is appropriate for the number of chain types
  #If it's length 1, just use same number modifications, etc. for all non-zero chain types
  #Otherwise, throw an error
  Nmod = np.array([Nmod]).flatten()
  if len(Nmod) != len(chainList)-1:
    if (len(chainList) > 2) and (len(Nmod) == 1):
      Nmod = Nmod[0]*np.ones(len(chainList)-1, dtype=int)
    else:
      print("Lenght of array specifying number of mutations for non-zero chain types is not equal to number of chain types minus 1. Quitting.")
      sys.exit(2)

  #Same thing with numNon
  numNon = np.array([numNon]).flatten()
  if len(numNon) != len(chainList)-1:
    if (len(chainList) > 2) and (len(numNon) == 1):
      numNon = numNon[0]*np.ones(len(chainList)-1, dtype=int)
    else:
      print("Length of array specifying number of chains for each non-zero chain type is not equal to number of chain types minus 1. Quitting.")
      sys.exit(2)

  savePrefix = "structures/gen%i" % generation

  newSurfs = []

  #Get total number of SAM chains 
  numChains = chainsX*chainsY

  if len(parentStructs) == 0:
    #Just generate some random surfaces
    for i in range(Nsurf):
      newPrefix = "%s_%i" % (savePrefix, i)
      newChains = genMutationsSAM(numNon, [], numChains)
      newSurfs.append(createSAMSurf(chainList, newChains, newPrefix, chainsX=chainsX, chainsY=chainsY))

  elif len(parentStructs) == Nsurf:

    if doMerge:
      #Uniquely pair up parents randomly by randomly ordering parentStructs
      parentInds = np.random.choice(np.arange(len(parentStructs)), size=len(parentStructs), replace=False)

      for i in range(Nsurf/2):
        oldchains1 = parentStructs[parentInds[i*2]].chainTypes
        oldchains2 = parentStructs[parentInds[i*2+1]].chainTypes
        parentChains = [[oldchains1, oldchains2], [oldchains2, oldchains1]]

        #Loop over pairs of parent chains
        for j, oldChains in enumerate(parentChains):

          if j == 0:
            newPrefix = "%s_%i" % (savePrefix, i*2)
          elif j == 1:
            newPrefix = "%s_%i" % (savePrefix, i*2+1)

          #Pick random number to pull from currently k+1 type chains from parent 1, then rest from parent 2
          #Will flip after done once
          arand = np.random.random()

          #Set up array to hold new chain types for child surface
          newChains = np.zeros(len(oldChains[0]), dtype=int)
          
          #Keep track of number of mutations needed for each surface chain type
          #This will generally be the number of mutations necessary after preserving density
          thisMod = copy.deepcopy(Nmod)

          #If not preserving density, change how we recombine the surfaces
          if not constantDens:
            samprand = int(arand*len(oldChains[0]))
            newChains[:samprand] = oldChains[0][:samprand]
            newChains[samprand:] = oldChains[1][samprand:]

            thisMod[:] = Nmod[:]

          #Otherwise do fancy things to preserve density of ALL chain types
          else: 
            #Loop over chain types to pull appropriate number from each parent
            #Will help preserve density/makes it easier to correct later
            for k in range(len(chainList)-1):

              #Get the indices in the old chain type lists of this chain type 
              oldNonInds1 = np.where(oldChains[0] == (k+1))[0]
              oldNonInds2 = np.where(oldChains[1] == (k+1))[0]

              if len(oldNonInds1) != len(oldNonInds2):
                print "Somehow have different densities of chain type %i on provided structures. Quitting."%(k+1)
                sys.exit(2)

              samprand = int(arand*len(oldNonInds1))
              newNonInds1 = oldNonInds1[:samprand]
              newChains[newNonInds1] = k+1
              newNonInds2 = oldNonInds2[samprand:]
              newChains[newNonInds2] = k+1

            #Now mutate from zero to k+1 type for as many as have left to reach desired density
            #MUST do after all combinations of non-zero chain type indices 
            #This is because higher integer types may overwrite lower
            for k in range(len(chainList)-1):

              #Get the indices in the old chain type lists of this chain type 
              oldNonInds1 = np.where(oldChains[0] == (k+1))[0]
              oldNonInds2 = np.where(oldChains[1] == (k+1))[0]

              newNonInds = np.where(newChains == (k+1))[0]
              mutForDens = len(oldNonInds1) - len(newNonInds)
              if mutForDens > 0:
                print "Due to overlap of chain type arrays, need to mutate to reach desired density."
                currZeroInds = np.where(newChains == 0)[0]
                changeToNonInds = np.random.choice(currZeroInds, size=mutForDens, replace=False)
                newChains[changeToNonInds] = k+1
              elif mutForDens == 0:
                print " "
              else:
                print "Have somehow reached higher density through cross-over. Quitting."
                sys.exit(2)

              #And now mutate surface, but only if mutated fewer than Nmod[k] pairs to maintain density
              if mutForDens >= Nmod[k]:
                thisMod[k] = 1
              else:
                thisMod[k] = Nmod[k] - mutForDens

          #And generate some mutations!
          newChains = genMutationsSAM(thisMod, newChains, numChains)
          newSurfs.append(createSAMSurf(chainList, newChains, newPrefix, chainsX=chainsX, chainsY=chainsY))

    else:
      #Just mutate each structure in list
      for i in range(Nsurf):
        currChains = parentStructs[i].chainTypes
        newPrefix = "%s_%i" % (savePrefix, i)
        newChains = genMutationsSAM(Nmod, currChains, numChains)
        newSurfs.append(createSAMSurf(chainList, newChains, newPrefix, chainsX=chainsX, chainsY=chainsY))

  #Finally return new list of SAMStructure objects
  return newSurfs


def genSurfsToy(strucTemplateObj, topTemplateStr, Nsurf, Nmod, parentStructs, generation, numAttr, numLJ,
                oldtype='Mr', oldres='MR', newtype='Ma', newres='MA', doMerge=True):
  """Generates toy (LJ) surfaces given set of input files.
     If parentStructs is zero in length, randomly
     generates surfaces with specified number of OH.
     If parentStructs has members, will only do cross-over
     step if doMerge is true. Otherwise, just generates
     mutations of each and returns.
     Inputs:
             strucTemplateObj - parmed structure object template to modify in creating new surfaces
             topTemplateStr - the topology file template (as a string) to modify in creating new surfaces
             Nsurf - number of surfaces to generate and return
             Nmod - number of modifications (mutations) for each surface
             parentStructs - a list of SAMStructure objects to use as parents
                             for the next generation; this MUST be same length
                             as Nsurf, or be zero in length.
             generation - the generation for these surfaces
             numAttr - number of attractive LJ particles in the surface
             numLJ - total number of LJ particles on the surface slab interface
             oldtype - (default 'Mr') the old atom type to replace in the template files
             oldres - (default 'MR') the old residue type to replace in the template files
             newtype - (default 'Ma') the new atom type to add to template files
             newres - (default 'MA') the new residue type to add to template files
             doMerge - (default True) if True, performs cross-over with parents
                       in parentStructs; otherwise goes straight to mutations
    Outputs:
             newSurfs - a list of length Nsurf containing new SAMStructure objects
  """
  #Check length of parentStructs
  if (len(parentStructs) != Nsurf):
    if (len(parentStructs) != 0):
      print "The length of parentStructs must match Nsurf or 0. See help for more info."
      sys.exit(2)

  if Nsurf%2 != 0:
   print "Number of surfaces to generate must be even so that can pair if desired."
   sys.exit(2)

  savePrefix = "structures/gen%i" % generation

  newSurfs = []

  if len(parentStructs) == 0:
    #Just generate some random surfaces
    for i in range(Nsurf):
      newPrefix = "%s_%i" % (savePrefix, i)
      newTypes = genMutationsToy(numAttr, [], numLJ)
      newSurfs.append(createToySurf(newTypes, newPrefix, strucTemplateObj, topTemplateStr, 
                                    oldType=oldtype, oldRes=oldres, newType=newtype, newRes=newres))

  elif len(parentStructs) == Nsurf:

    if doMerge:
      #Uniquely pair up parents randomly by randomly ordering parentStructs
      parentInds = np.random.choice(np.arange(len(parentStructs)), size=len(parentStructs), replace=False)

      for i in range(Nsurf/2):
        oldtypes1 = parentStructs[parentInds[i*2]].LJTypes
        oldtypes2 = parentStructs[parentInds[i*2+1]].LJTypes
        parentTypes = [[oldtypes1, oldtypes2], [oldtypes2, oldtypes1]]
        typeInds = np.arange(len(oldtypes1))

        #Pick random number to pull from currently attractive LJ particles from parent 1, then rest from parent 2
        #Will flip after done once
        arand = np.random.random()
        for j, oldTypes in enumerate(parentTypes):

          newTypes = np.zeros(len(oldTypes[0]), dtype=bool)

          if np.sum(oldTypes[0]) != np.sum(oldTypes[1]):
            print "Somehow have different attractive LJ particle densities on provided structures. Quitting."
            sys.exit(2)

          samprand1 = int(arand*np.sum(oldTypes[0]))
          samprand2 = np.sum(oldTypes[1]) - samprand1
          newAttrInds1 = typeInds[oldTypes[0]][:samprand1]
          newTypes[newAttrInds1] = True
          newAttrInds2 = typeInds[oldTypes[1]][samprand1:]
          newTypes[newAttrInds2] = True

          #Now mutate from repulsive to attractive for as many as have left to reach desired density
          mutForDens = np.sum(oldTypes[0]) - np.sum(newTypes)
          if mutForDens > 0:
            print "Due to overlap of boolean arrays, need to mutate to reach desired density."
            currRepInds = typeInds[np.invert(newTypes)]
            changeToAttrInds = np.random.choice(currRepInds, size=mutForDens, replace=False)
            newTypes[changeToAttrInds] = True
          elif mutForDens == 0:
            print " "
          else:
            print "Have somehow reached higher density through cross-over. Quitting."
            sys.exit(2)

          #And now mutate surface, but only if mutated fewer than Nmod pairs to maintain density
          if mutForDens >= Nmod:
            thisMod = 1
          else:
            thisMod = Nmod - mutForDens

          if j == 0:
            newPrefix = "%s_%i" % (savePrefix, i*2)
          elif j == 1:
            newPrefix = "%s_%i" % (savePrefix, i*2+1)

          #And generate some mutations!
          newTypes = genMutationsToy(thisMod, newTypes, numLJ)
          newSurfs.append(createToySurf(newTypes, newPrefix, strucTemplateObj, topTemplateStr, 
                                        oldType=oldtype, oldRes=oldres, newType=newtype, newRes=newres))

    else:
      #Just mutate each structure in list
      for i in range(Nsurf):
        currTypes = parentStructs[i].LJTypes
        newPrefix = "%s_%i" % (savePrefix, i)
        newTypes = genMutationsToy(Nmod, currTypes, numLJ)
        newSurfs.append(createToySurf(newTypes, newPrefix, strucTemplateObj, topTemplateStr, 
                                      oldType=oldtype, oldRes=oldres, newType=newtype, newRes=newres))

  #Finally return new list of ToyStructure objects
  return newSurfs


def strucRMSD(abool1, abool2):
  """Given two boolean arrays corresponding to two surface objects, computes the RMSD 
     between them, which, ignoring degeneracies, gives their distance from each other in 
     the space that the genetic algorithm searches.
     Inputs:
            abool1 - boolean array of first SurfStructure object
            abool2 - boolean array of second SurfStructure object
     Outputs:
            rmsdval - the RMSD between the boolean arrays
  """
  #Convert the condPairs arrays to float values
  abool1 = np.array(abool1, dtype=float)   
  abool2 = np.array(abool2, dtype=float)   

  #And compute RMSD
  rmsdval = np.sqrt(np.average((abool1 - abool2)**2))

  return rmsdval


def clusterSurfs(surfList, surfType, Cutoff=0.42, MaxIter=200, MaxCluster=-32, MaxClusterWork=None, Verbose=False):
  """Clustering algorithm based on ClusterMSS in rmsd.py. Just customized for surface structure class.
     Note that Scott Shell gets all the credit for this code.
     Inputs:
              surfList - list of surface structure objects to cluster 
                         (can be quartz -> SurfStructure or SAM -> SAMStructure)
              surfType - type of surface structure object ('quartz' or 'SAM')
              Cutoff - (default 0.42 - seemed good from limited tests) RMSD cutoff for a config to belong to a cluster
              MaxIter - (default 3) maximum number of iterations to perform
              MaxCluster - (default -32) maximum number of clusters
                           a negative number forces all configs into a cluster
              MaxClusterWork - (default None) max number of working clusters
              Verbose - (default False) to print information or not
     Outputs:
              clustList - a list of lists of SurfStructure objects, each within a cluster
  """
  if surfType not in ['quartz', 'SAM']:
    print "To run clustering, surfType must be either \'quartz\' or \'SAM\'"
    sys.exit(2)

  print "For clustering, using RMSD cluster cutoff of %f" % Cutoff

  Iteration = 0
  WeightSum = [] #Total weights (number of members) in each cluster
  PosSum = [] #List of cluster configuration array (i.e. centroid structure of each cluster)
  FinalIters = 0 #number of iterations without additions/deletions of clusters
  NSurfs = len(surfList)
  StartInd, NewStartInd = 0, -1

  #Now set-up, start loop
  while FinalIters < 2:

    Iteration += 1
    FinalIters += 1

    if Iteration > MaxIter:
      print "Did not converge within maximum number of iterations"
      break
    if Verbose: print "Cluster iteration %i" % Iteration
    if Verbose: print "Starting with %i clusters" % len(PosSum)

    ClustNum = np.zeros(NSurfs, dtype=int) #cluster number of each configuration, starting at 1
    NAddThis = [0]*len(PosSum) #number of configs added to each cluster this iteration
    ThisInd = 0
    PosSumThis = copy.deepcopy(PosSum)
    WeightSumThis = copy.deepcopy(WeightSum)

    #Get where to start
    if NewStartInd >= 0: StartInd = NewStartInd
    NewStartInd = -1

    for CurInd in range(StartInd, NSurfs)+range(0, StartInd):
      if surfType == 'quartz':
        CurSurf = np.array(surfList[CurInd].condPairs, dtype=float) #Set type to float to make RMSD metric work
      elif surfType == 'SAM':
        CurSurf = np.array(surfList[CurInd].chainTypes, dtype=float)
      ThisInd += 1
      ind = -1 #cluster number assigned to this config; -1 means no cluster
      #calculate RMSD between this configuration and each cluster config,
      #but stop when an RMSD is found which is below the cutoff
      minRMSD = 1.e300
      for (i, PosSumi) in enumerate(PosSum):
        r = strucRMSD(PosSumi/WeightSum[i], CurSurf)
        minRMSD = min(minRMSD, r)
        if r < Cutoff:
          #Then put it in this cluster
          ind = i
          break
      if ind >= 0:
        #Add surface structure to the cluster
        PosSum[ind] = PosSum[ind] + CurSurf
        WeightSum[ind] = WeightSum[ind] + 1.
        NAddThis[ind] = NAddThis[ind] + 1
        ClustNum[CurInd] = ind+1
      elif len(PosSum) < MaxClusterWork or MaxClusterWork is None:
        #create a new cluster with this config if haven't reached max number working clusters
        if minRMSD == 1.e300: minRMSD = 0.
        if Verbose: print "Adding cluster: surf structure %d | min RMSD %.1f | %d clusters tot" % (ThisInd, minRMSD, len(PosSum)+1)
        PosSum.append(CurSurf)
        WeightSum.append(1.)
        NAddThis.append(1)
        ClustNum[CurInd] = len(PosSum)
        FinalIters = 0
      else:
        #do nothing with this surface
        ClustNum[CurInd] = 0
        FinalIters = 0
        if NewStartInd < 0:
          NewStartInd = CurInd
          if Verbose: print "Ran out of clusters. Next iteration starting from surface index %d" % ThisInd
 
    #Remove contribution to centroids from all but this round
    for i in range(len(PosSumThis)):
      #Make sure this cluster actually added configs, though
      if not NAddThis[i] == 0:
        PosSum[i] = PosSum[i] - PosSumThis[i]
        WeightSum[i] = WeightSum[i] - WeightSumThis[i]
    del PosSumThis
    del WeightSumThis

    #Loop through clusters
    i=0
    while i < len(PosSum):
      #remove clusters with no additions this iteration
      if NAddThis[i] == 0:
        if Verbose: print "Removing cluster %d" % (i+1)
        del PosSum[i]
        del NAddThis[i]
        for (k, cn) in enumerate(ClustNum):
          if cn > i + 1:
            ClustNum[k] -= 1
          elif cn == i + 1:
            ClustNum[k] = -1
        FinalIters = 0
      else:
        i += 1

    #Sort clusters and then remove any beyond MaxCluster
    ClustNum = np.array(ClustNum, dtype=int)
    if Verbose: print "Reordering clusters by population"
    ClustsSorted = [(WeightSum[i], i) for i in range(len(PosSum))]
    ClustsSorted.sort()
    ClustsSorted.reverse()
    PosSum = [PosSum[j] for (i,j) in ClustsSorted]
    WeightSum = [WeightSum[j] for (i,j) in ClustsSorted]
    NAddThis = [NAddThis[j] for (i,j) in ClustsSorted]
    #create dictionary mapping new cluster number to old cluster number
    Trans = {0:0}
    for i in range(len(PosSum)):
      anind = ClustsSorted[i][1] + 1
      Trans[anind] = i+1
      Trans[-anind] = -i -1
    #Update ClustNum
    ClustNum = np.array([Trans[i] for i in ClustNum], dtype=int)
    
    #Crop off any extraneous clusters; clusterless configs get assigned a cluster index of 0
    if not MaxCluster is None and len(PosSum) > abs(MaxCluster):
      del PosSum[abs(MaxCluster):]
      WeightSum = WeightSum[:abs(MaxCluster)]
      NAddThis = NAddThis[:abs(MaxCluster)]
      ClustNum[abs(ClustNum) > abs(MaxCluster)] = 0

  #Crop off any extraneous clusters; clusterless configs get assigned a cluster index of 0
  if not MaxCluster is None and len(PosSum) > abs(MaxCluster):
    del PosSum[abs(MaxCluster):]
    WeightSum = WeightSum[:abs(MaxCluster)]
    NAddThis = NAddThis[:abs(MaxCluster)]
    ClustNum[abs(ClustNum) > abs(MaxCluster)] = 0

  ClustPops = copy.deepcopy(WeightSum)

  #If MaxCluster is negative force everything to nearest cluster
  if not MaxCluster == None and MaxCluster < 0:
    c = np.sum(ClustNum == 0)
    print "Forcing %d extraneous configurations to existing clusters" % c
    for (j, CurSurf) in enumerate(surfList):
      if ClustNum[j] == 0:
        CurPos = np.array(CurSurf.condPairs, dtype=int)
        ind = -1
        minr = 0.
        for (i, PosSumi) in enumerate(PosSum):
          r = strucRMSD(PosSumi/WeightSum[i], CurPos)
          if r < minr or ind < 0:
            ind = i
            minr = r
        ClustNum[j] = ind + 1
        ClustPops[ind] += 1

  #Need to re-sort clusters by population again before placing in returned list
  ClustsSorted = [(ClustPops[i], i) for i in range(len(PosSum))]
  ClustsSorted.sort()
  ClustsSorted.reverse()
  PosSum = [PosSum[j] for (i,j) in ClustsSorted]
  ClustPops = [ClustPops[j] for (i,j) in ClustsSorted]
  #create dictionary mapping new cluster number to old cluster number
  Trans = {0:0}
  for i in range(len(PosSum)):
    anind = ClustsSorted[i][1] + 1
    Trans[anind] = i+1
    Trans[-anind] = -i -1
  #Update ClustNum
  ClustNum = np.array([Trans[i] for i in ClustNum], dtype=int)

  #Print some useful information - could change to only print with Verbose flag, but nice to see
  print "Cluster populations are as follows:"
  for i, pop in enumerate(ClustPops):
    print "Cluster %i: %d" % (i+1, pop)

  #Now take each structure in surfList and place in a list 
  clustList = [[] for _ in range(len(PosSum))]
  for i, surf in enumerate(surfList):
    clustList[ClustNum[i]-1].append(surf)

  return clustList


def tournamentSelect(objValues, Nparents=8, bracketSize=2, optMax=False):
  """Performs a tournament selection that first truncates to only top performers
     Inputs:
             objValues - value of objective function for all possible parents
             Nparents - number of parents to select in end (number of tournaments to perform)
             bracketSize - number of indivduals in each tournament
             optMax - whether maximizing (True) or minimizing (False) the objective function
     Outputs:
             parentInds - indices in objValues of winning individuals to become parents
  """

  #Make sure have array
  objValues = np.array(objValues)

  #Define bracketSize if set to None
  if bracketSize is None:
    #Set bracket size to grow with number of individuals compared to parents
    bracketSize = len(objValues) / Nparents

  #Create set of indices to randomly pick from
  allInds = np.arange(len(objValues))

  #To hone in on top performers, truncate so that fill out brackets nicely
  if optMax:
    availInds = allInds[-Nparents*2*bracketSize:]
  else:
    availInds = allInds[:Nparents*2*bracketSize]

  if len(availInds) == Nparents:
    parentInds = availInds
  else:
    parentInds = []

  #Conduct tournaments to decide parents, but don't allow same parent to be included twice
  while len(parentInds) < Nparents:
    #Randomly pick bracketSize individuals from the available indices, WITHOUT replacement
    randInds = np.random.choice(availInds, size=bracketSize, replace=False)
    randVals = objValues[randInds]
    if optMax:
      #Make sure not already in set
      thisInd = randInds[np.argsort(randVals)[-1]]
      if thisInd in parentInds:
        continue
      else:
        parentInds.append(thisInd)
    else:
      thisInd = randInds[np.argsort(randVals)[0]]
      if thisInd in parentInds:
        continue
      else:
        parentInds.append(thisInd)

  return np.array(parentInds)


def calcDiff(topFile, trjFile, surfType, method='MSD'):
  """Calculates the diffusivity near to upper and lower surfaces.
     Inputs:  
             topFile - topology file for trajectory
             trjFile - trajectory file
             surfType - type of surface (can be 'quartz' or 'SAM' or 'toy')
             method - can be 'MSD' (diffusivity parallel to surfaces using MSD curves)
                      or 'Surv' (diffusivity from survival probabilities)
     Outputs:
             DsurfLo - diffusivity on lower surface
             DsurfHi - diffusivity near upper surface
             DsurfNet - diffusivity for both surfaces together
             Dbulk - diffusivity in (should be small) "bulk" region
  """

  #First define survival fitting if desired - best to define this within calcDiff
  def fitSurvivalD(nWat, Xc, Yc, Zc, norder, tdat, SurfBool=True):
    """Uses non-linear least-squares to find the best fit for D given
       data on the number of waters, each of the x, y, and z cutoffs,
       the order to compute sums of exponentials and time data. Set 
       SurfBool to True for near the surface (reflective boundary on one
       side of z) or False for in bulk (absorbing boundaries on both
       sides.
    """
  
    def survFuncSurf(Lx, Ly, Lz, n, t, D):
      """Computes survival probability over region Lx, Ly, Lz to nth 
         order in sums and at time t with diffusion constant D. Should
         be used for region near surface since assumes reflective boundary
         on one side of z and absorbing on the other."""
    
      prefac = 128.0 / (np.pi**6)
      
      nvecxy = np.arange(1, n*2, 2)
      nvecz = np.arange(1, n, 1)
    
      xsum = np.sum( np.exp(-((nvecxy*np.pi/Lx)**2)*D*t) / (nvecxy**2) )
      ysum = np.sum( np.exp(-((nvecxy*np.pi/Ly)**2)*D*t) / (nvecxy**2) )
      zsum = np.sum( np.exp(-(((nvecz-0.5)*np.pi/Lz)**2)*D*t) / ((nvecz - 0.5)**2) )
    
      funcval = prefac*xsum*ysum*zsum
    
      return funcval
    
    def survFuncSurfDeriv(Lx, Ly, Lz, n, t, D):
      """Computes derivative of survival probability over region Lx, Ly, Lz to nth 
         order in sums and at time t with diffusion constant D. Should
         be used for region near surface since assumes reflective boundary
         on one side of z and absorbing on the other."""
    
      prefac = -128.0 * (t**3) / ((Lx*Ly*Lz)**2)
      
      nvecxy = np.arange(1, n*2, 2)
      nvecz = np.arange(1, n, 1)
    
      xsum = np.sum( np.exp(-((nvecxy*np.pi/Lx)**2)*D*t) )
      ysum = np.sum( np.exp(-((nvecxy*np.pi/Ly)**2)*D*t) )
      zsum = np.sum( np.exp(-(((nvecz-0.5)*np.pi/Lz)**2)*D*t) )
    
      funcval = prefac*xsum*ysum*zsum
    
      return funcval
    
    def survFuncBulk(Lx, Ly, Lz, n, t, D):
      """Computes survival probability over region Lx, Ly, Lz to nth 
         order in sums and at time t with diffusion constant D. Should
         be used for region bulk region since assumes absorbing boundaries
         on both sides of z direction."""
    
      prefac = 512.0 / (np.pi**6)
      
      nvec = np.arange(1, n*2, 2)
    
      xsum = np.sum( np.exp(-((nvec*np.pi/Lx)**2)*D*t) / (nvec**2) )
      ysum = np.sum( np.exp(-((nvec*np.pi/Ly)**2)*D*t) / (nvec**2) )
      zsum = np.sum( np.exp(-((nvec*np.pi/Lz)**2)*D*t) / (nvec**2) )
    
      funcval = prefac*xsum*ysum*zsum
    
      return funcval
    
    def survFuncBulkDeriv(Lx, Ly, Lz, n, t, D):
      """Computes derivative of survival probability over region Lx, Ly, Lz to nth 
         order in sums and at time t with diffusion constant D. Should
         be used for region bulk region since assumes absorbing boundaries
         on both sides of z direction."""
    
      prefac = -512.0 * (t**3) / ((Lx*Ly*Lz)**2)
      
      nvec = np.arange(1, n*2, 2)
    
      xsum = np.sum( np.exp(-((nvec*np.pi/Lx)**2)*D*t) )
      ysum = np.sum( np.exp(-((nvec*np.pi/Ly)**2)*D*t) )
      zsum = np.sum( np.exp(-((nvec*np.pi/Lz)**2)*D*t) )
    
      funcval = prefac*xsum*ysum*zsum
  
      return funcval
  
    #Now necessary functions are defined, define residuals
    if not SurfBool:
  
      def lsfunc(D, tvals, yvals):
        retvec = [survFuncSurf(Xc, Yc, Zc, 100, tval, D[0]) for tval in tvals]
        return np.array(retvec) - yvals
  
      def dlsfuncdD(D, tvals, yvals):
        retvec = [survFuncSurfDeriv(Xc, Yc, Zc, 100, tval, D[0]) for tval in tvals]
        return np.array([retvec]).T
  
    else:
  
      def lsfunc(D, tvals, yvals):
        retvec = [survFuncBulk(Xc, Yc, Zc, 100, tval, D[0]) for tval in tvals]
        return np.array(retvec) - yvals
  
      def dlsfuncdD(D, tvals, yvals):
        retvec = [survFuncBulkDeriv(Xc, Yc, Zc, 100, tval, D[0]) for tval in tvals]
        return np.array([retvec]).T
  
    guess = np.array([0.5])
  
    #Do least squares
    sol_lsq = optimize.least_squares(lsfunc, guess, jac=dlsfuncdD, args=(tdat, nWat), bounds=(0.0, 100.0), verbose=1)
  
    #Want to return D, then also a 2D vector of times and probability of the result
    vecmodel = np.arange(0.0, np.max(tdat), 0.01)
    vecmodel = np.vstack((vecmodel, lsfunc(sol_lsq.x, vecmodel, np.zeros(len(vecmodel)))))
  
    return sol_lsq.x[0], vecmodel

  #And now get on with running the methods
  methodList = ['MSD', 'Surv']

  if method not in methodList:
    print "%s diffusivity calculation method unknown. Must choose from:" % method
    print methodList
    sys.exit(2)

  if surfType not in ['quartz', 'SAM', 'toy']:
    print "%s surface type not recognized. Must be \'quartz\' or \'SAM\' or \'toy\'." % surfType
    sys.exit(2)

  #Read in topology with parmed
  top = pmd.load_file(topFile)

  #Just in case have Ryckaert-Bellemans torsions (do for SAM system), remove so can convert
  #Don't need this info for the analysis I'm interested in
  top.rb_torsions = pmd.TrackedList([])
  top = pt.load_parmed(top, traj=False)

  #Read trajectory in and link to topology
  traj = pt.iterload(trjFile, top)

  #Find number of frames 
  nFrames = traj.n_frames 

  #Get box dimensions
  boxDims = traj.unitcells[0, 0:3]

  #Get indices of water oxygens, water hydrogens, and surface atoms (Si or terminal cap groups)
  if surfType == 'quartz':
    surfInds = top.select(':XXX@Si') #Use Si atoms to find surface location
  elif surfType == 'SAM':
    surfInds = top.select('@%CT,OH') #Use atom types CT and OH to find surface location
  elif surfType == 'toy':
    surfInds = top.select('@Ma,Mr') #Use attractive (Ma) and repulsive (Mr) colloids
  owInds = top.select('@OW')

  #Define z cutoff for waters to be near surface
  #Will define surface from minimum and maximum z values of silicon atoms
  if method == 'MSD':
    zCut = 8.0
  elif method == 'Surv':
    zCut = 10.0
  else:
    zCut = 8.0

  #Define separate z-regions for tracking number of surviving waters over time in the z-direction only
  zCloseSurv1 = 1.50
  zCloseSurv2 = 3.50 #Somewhat arbitrary 2 Angstrom thick z-slice near surface
  zBulkSurv1 = 10.5
  zBulkSurv2 = 12.5 #And another arbitrary 2 Angstrom thick z-slice in bulk water region

  #Also define x and y cutoffs for region to track waters in, as list
  xCut = [0.0, 10.0]
  yCut = [0.0, 10.0]

  #Set time between MSD starting origins
  #Set to 50 steps (25 ps) with 0.5 ps per frame
  msdSteps = 50
  timeVals = np.arange(msdSteps)*0.5

  #Also set time over which to track waters within a defined volume for survival probabilities
  #if method == 'Surv':
  survSteps = 200
  #else:
  #  survSteps = msdSteps #Saves a little time
  #Except that need more time to accurately compute average residence time in z-slices

  #Partition trajectory into chunks to match time origins
  #Start each time origin spaced at msdSteps, but each time, go past to look at survival probabilities
  #Since go from each time origin forward survSteps, make sure can do this for all start points 
  startPoints = np.arange(0, nFrames, msdSteps)
  while startPoints[-1] + survSteps >= nFrames:
    startPoints = startPoints[:-1]

  #Define fraction of beginning and end of MSD curve to to be excluded in linear fit
  initFrac = 0.08
  tailFrac = 0.42

  #Will save sum of MSD versus time for waters near lower surface, near upper surface and bulk
  #Lower surface will be column 0, upper 1, and bulk region 2
  sumMSDs = np.zeros((msdSteps, 3))

  #To correctly average MSD at each step in msdSteps, need to save number of samples
  #(won't be consistent, because waters can change from being surface or bulk)
  numMSDs = np.zeros((msdSteps, 3))

  #Track number of water oxygens in the bulk and surface volumes at each time step
  numWat = np.zeros((survSteps, 3))

  #Also keep track of number of waters over time in each 2 Angstrom z-slice
  numWatZ = np.zeros((survSteps, 2))

  avgZs = np.zeros(3)

  #Loop through all time origins
  for i, t in enumerate(startPoints):

    #Get coordinates
    allCurrCoords = np.array(traj[t].xyz)

    #Do wrapping of all atoms in the system in two steps...
    #First, use the first atom specifying the surface as the reference which should 
    #put the entire surface into one piece because the simulation box in the z 
    #direction is large enough that any surface atoms currently on the opposite side 
    #of the box from this first atom will be at least half a box-length away.
    #Goal is to get the surface itself in the middle of the z-axis with water equally
    #on both sides so that the algorithm for defining water regions below works.
    #Second, once all surface indices are on the same side of the box, use their
    #center of geometry as the reference to get all the waters to fill in about 
    #equally on each side.
    wrapCOM1 = allCurrCoords[surfInds[0]]
    allCurrCoords = wl.reimage(allCurrCoords, wrapCOM1, boxDims)
    wrapCOM2 = np.average(allCurrCoords[surfInds], axis=0)
    allCurrCoords = wl.reimage(allCurrCoords, wrapCOM2, boxDims)

    #Set up array of reference water coordinates for this time origin
    refOcoords = allCurrCoords[owInds]

    #Will also need set of oxygen coordinates to use for (un)imaging
    #Will change at every step to do imaging right
    prevOcoords = allCurrCoords[owInds]

    #Need to track SD for each water oxygen - make -1 so clear if nothing recorded
    watSDs = (-1) * np.ones((len(owInds),survSteps))

    #Find maximum and minimum z coordinates of the silica surface for this time origin
    #Then add zCut to define near and far waters
    maxSurfZ = np.max(allCurrCoords[surfInds,2])
    minSurfZ = np.min(allCurrCoords[surfInds,2])
    maxZ = maxSurfZ + zCut
    minZ = minSurfZ - zCut

    #To differentiate upper and lower surfaces, find midpoint z
    midZ = 0.5*(maxZ + minZ)

    avgZs[0] += minZ / len(startPoints)
    avgZs[1] += maxZ / len(startPoints)
    avgZs[2] += (boxDims[2] - (maxZ - minZ)) / len(startPoints)

    #Need starting water region (lower surface 0, upper 1, bulk 2)
    watRegion = np.zeros(len(owInds), dtype=int)
    for k, zPos in enumerate(refOcoords[:,2]):
      if zPos >= minZ:
        if zPos < midZ:
          watRegion[k] = 0
        elif zPos <= maxZ:
          watRegion[k] = 1
        else:
          watRegion[k] = 2
      else:
        watRegion[k] = 2

    #Also need list of waters starting in each z-region measuring water survival
    closeZregionList = []
    bulkZregionList = []
    for k, zPos in enumerate(refOcoords[:,2]):
      if (   ((zPos >= maxSurfZ + zCloseSurv1) and (zPos <= maxSurfZ + zCloseSurv2))
          or ((zPos <= minSurfZ - zCloseSurv1) and (zPos >= minSurfZ - zCloseSurv2)) ):
        closeZregionList.append(k)
      elif ((zPos >= maxSurfZ + zBulkSurv1) and (zPos <= maxSurfZ + zBulkSurv2)):
        bulkZregionList.append(k)

    #And keep track of number of these waters that have not left yet at each time step
    tempNumWatZ = np.zeros((survSteps, 2))

    #Need flag to say if water is in x and y cutoffs as well - same for each z-region
    #Do this with boolean array after re-imaging the x and y cutoffs
    thisXcut = wl.reimage(xCut, wrapCOM1[0], boxDims[0])
    thisXcut = wl.reimage(thisXcut, wrapCOM2[0], boxDims[0])
    thisXcut.sort()
    thisYcut = wl.reimage(yCut, wrapCOM1[1], boxDims[1])
    thisYcut = wl.reimage(thisYcut, wrapCOM2[1], boxDims[1])
    thisYcut.sort()
    xyBool = ( (refOcoords[:,0] > thisXcut[0]) * (refOcoords[:,0] < thisXcut[1])
              * (refOcoords[:,1] > thisYcut[0]) * (refOcoords[:,1] < thisYcut[1]) )

    #Also need to keep track of whether or not waters have changed z-region
    #Can't just check if same as in watRegion, because could have left and come back
    sameRegion = np.ones(len(owInds), dtype=bool)

    #Create temporary array to hold numbers of water as function of time
    #Start with total number in region for all times in array and subtract as leave
    tempNumWat = np.zeros((survSteps, 3))
    tempNumWat[:,0] = np.sum(xyBool[np.where(watRegion==0)[0]])
    tempNumWat[:,1] = np.sum(xyBool[np.where(watRegion==1)[0]])
    tempNumWat[:,2] = np.sum(xyBool[np.where(watRegion==2)[0]])

    for j, frame in enumerate(traj(t,t+survSteps,1)):

      #Get all current coordinates for this frame
      tempCurrCoords = np.array(frame.xyz)

      #Do wrapping procedure for this frame and get z cut-offs
      wrapCOM1 = tempCurrCoords[surfInds[0]]
      tempCurrCoords = wl.reimage(tempCurrCoords, wrapCOM1, boxDims)
      wrapCOM2 = np.average(tempCurrCoords[surfInds], axis=0)
      tempCurrCoords = wl.reimage(tempCurrCoords, wrapCOM2, boxDims)

      #Find maximum and minimum z coordinates of the silica surface for this time origin
      #Then add zCut to define near and far waters
      maxSurfZ = np.max(allCurrCoords[surfInds,2])
      minSurfZ = np.min(allCurrCoords[surfInds,2])
      maxZ = maxSurfZ + zCut
      minZ = minSurfZ - zCut

      #To differentiate upper and lower surfaces, find midpoint z
      midZ = 0.5*(maxZ + minZ)

      #Get current oxygen and hydrogen positions
      tempOcoords = tempCurrCoords[owInds]

      #Figure out oxygen occupancy for survival z regions quickly
      tempCloseZregionList = []
      for k, zPos in enumerate(tempOcoords[closeZregionList,2]):
        if (   ((zPos >= maxSurfZ + zCloseSurv1) and (zPos <= maxSurfZ + zCloseSurv2))
              or ((zPos <= minSurfZ - zCloseSurv1) and (zPos >= minSurfZ - zCloseSurv2)) ):
          tempCloseZregionList.append(closeZregionList[k])
      closeZregionList = copy.deepcopy(tempCloseZregionList)
      tempNumWatZ[j,0] = len(closeZregionList)

      tempBulkZregionList = []
      for k, zPos in enumerate(tempOcoords[bulkZregionList,2]):
        if ((zPos >= maxSurfZ + zBulkSurv1) and (zPos <= maxSurfZ + zBulkSurv2)):
          tempBulkZregionList.append(bulkZregionList[k])
      bulkZregionList = copy.deepcopy(tempBulkZregionList)
      tempNumWatZ[j,1] = len(bulkZregionList)

      #Loop over all water oxygens
      for k, pos in enumerate(tempOcoords):

        #Check if water has left its starting region (in z direction)
        if not sameRegion[k]:
          continue

        thisRegion = 2 #Default to put in bulk; 0 is lower surface, 1 is upper surface

        #Determine which region this water is in currently
        if pos[2] >= minZ:
          if pos[2] < midZ:
            thisRegion = 0
          elif pos[2] <= maxZ:
            thisRegion = 1
          else:
            thisRegion = 2
        else:
          thisRegion = 2

        #Check if region matches previous classification
        if thisRegion != watRegion[k]:

          #If doesn't match, set sameRegion flag to False for this water, skip SD computation
          sameRegion[k] = False

          #Also subtract one from number of waters in the tracked volume for this region for this time onwards
          #But only if haven't already done so because left x and y regions!
          if xyBool[k] == True:
            tempNumWat[j:, watRegion[k]] += -1

          continue

        #Now check if this water has left the box in the x or y directions
        #Note that only check if first hasn't left z-region and has never left x and y region
        if xyBool[k] == True:

          thisXcut = wl.reimage(xCut, wrapCOM1[0], boxDims[0])
          thisXcut = wl.reimage(thisXcut, wrapCOM2[0], boxDims[0])
          thisXcut.sort()
          thisYcut = wl.reimage(yCut, wrapCOM1[1], boxDims[1])
          thisYcut = wl.reimage(thisYcut, wrapCOM2[1], boxDims[1])
          thisYcut.sort()

          if ( (pos[0] < thisXcut[0])
              or (pos[0] > thisXcut[1])
              or (pos[1] < thisYcut[0]) 
              or (pos[1] > thisYcut[1]) ):

            xyBool[k] = False
            tempNumWat[j:, watRegion[k]] += -1

        #If have reached msdSteps, don't bother with SD calculations
        if j >= msdSteps:
          continue

        #If still in same region (as checked by if statements above), update SD
        #Compute image from last position
        newpos = wl.reimage([pos], prevOcoords[k], boxDims)[0]

        #Update previous coordinates
        prevOcoords[k] = newpos

        #Compute SD from reference
        thisSD = (newpos - refOcoords[k])**2

        #Determine position in watSDs to update and update with SD
        sdInd = np.where(watSDs[k] == -1)[0][0]
        watSDs[k,sdInd] = np.sum(thisSD[:2]) #Only doing lateral diffusion

    #Now go through collected SDs and add info to overall SD sums and residence times
    for k in range(len(owInds)):

      try:
        sdInd = np.where(watSDs[k] == -1)[0][0]

        #If this works, water left before end of time chunk (i.e. msdSteps)
        sumMSDs[:sdInd,watRegion[k]] += watSDs[k,:sdInd]
        numMSDs[:sdInd,watRegion[k]] += 1

      except IndexError:
        sumMSDs[:,watRegion[k]] += watSDs[k,:msdSteps]
        numMSDs[:,watRegion[k]] += 1

    #Also collect information for surviving water oxygens
    numWat += tempNumWat / len(startPoints)

    #And for just 2 Angstrom z slices
    numWatZ += tempNumWatZ / len(startPoints)

  #Finish computing net MSD over all data accumulated
  netMSDs = sumMSDs / numMSDs

  #Set surviving waters to fraction of initial starting amount (goes to 1 at time zero)
  numWat[:,0] = numWat[:,0] / numWat[0,0]
  numWat[:,1] = numWat[:,1] / numWat[0,1]
  numWat[:,2] = numWat[:,2] / numWat[0,2]

  #Finally compute diffusion coefficients
  #Note that this will fail if no waters stayed in each region for at least msdSteps
  if method == 'MSD':

    slopeLo, interceptLo, rvalLo, pvalLo, stderrLo = stats.linregress(timeVals[int(initFrac*msdSteps):int(tailFrac*msdSteps)], netMSDs[int(initFrac*msdSteps):int(tailFrac*msdSteps),0])
  
    DsurfLo = slopeLo / 4.0
    DsurfErrLo = stderrLo / 4.0
  
    slopeHi, interceptHi, rvalHi, pvalHi, stderrHi = stats.linregress(timeVals[int(initFrac*msdSteps):int(tailFrac*msdSteps)], netMSDs[int(initFrac*msdSteps):int(tailFrac*msdSteps),1])
  
    DsurfHi = slopeHi / 4.0
    DsurfErrHi = stderrHi / 4.0
  
    netSurfMSDs = (sumMSDs[:,0]+sumMSDs[:,1]) / (numMSDs[:,0]+numMSDs[:,1])
  
    slopeNet, interceptNet, rvalNet, pvalNet, stderrNet = stats.linregress(timeVals[int(initFrac*msdSteps):int(tailFrac*msdSteps)], netSurfMSDs[int(initFrac*msdSteps):int(tailFrac*msdSteps)])
  
    DsurfNet = slopeNet / 4.0
    DsurfNetErr = stderrNet / 4.0
  
    slopeBulk, interceptBulk, rvalBulk, pvalBulk, stderrBulk = stats.linregress(timeVals[int(initFrac*msdSteps):int(tailFrac*msdSteps)], netMSDs[int(initFrac*msdSteps):int(tailFrac*msdSteps),2])
  
    Dbulk = slopeBulk / 4.0
    DbulkErr = stderrBulk / 4.0

  elif method == 'Surv':

    DsurfLo, DfitLo = fitSurvivalD(numWat[:,0], xCut[1]-xCut[0], yCut[1]-yCut[0], zCut, 100, 
                                                     np.arange(survSteps)*0.5, SurfBool=True)
    DsurfHi, DfitHi = fitSurvivalD(numWat[:,1], xCut[1]-xCut[0], yCut[1]-yCut[0], zCut, 100, 
                                                     np.arange(survSteps)*0.5, SurfBool=True)
    DsurfNet, DfitNet = fitSurvivalD(0.5*(numWat[:,0]+numWat[:,1]), xCut[1]-xCut[0], 
                                             yCut[1]-yCut[0], zCut, 100, np.arange(survSteps)*0.5, SurfBool=True)
    Dbulk, DfitBulk = fitSurvivalD(numWat[:,2], xCut[1]-xCut[0], yCut[1]-yCut[0], avgZs[2], 100, 
                                                          np.arange(survSteps)*0.5, SurfBool=False)

  #Compute average survival times in z-slices
  #(Assumes probability is zero past survSteps time, so probably an underestimate by a little)
  avgZTimeClose = np.sum(0.5*np.arange(survSteps)*numWatZ[:,0]/numWatZ[0,0])
  avgZTimeBulk = np.sum(0.5*np.arange(survSteps)*numWatZ[:,1]/numWatZ[0,1])

  #np.savetxt('Num_Wat_Z_regions.txt', numWatZ)

  #Return calculated diffusion coefficients
  #ALSO returning computed average z-slice survival times as last two elements in tuple
  #Will be backwards compatible because these elements of the tuple just won't be referenced ever
  return DsurfLo, DsurfHi, DsurfNet, Dbulk, avgZTimeClose, avgZTimeBulk


def doJobCalcDiff(procnum, return_dict, TopFile, TrjFile, SurfType, Method):
  """A wrapper function for calcDiff so that calcDiff may be run in parellel
     if desired. Takes additional parameters procnum and return_dict, so puts
     output of call to calcDiff in a global dictionary passed with the procnum
     as the key. This way, can run as many calcDiff in parallel as like, then
     reference each by the procnum it was given.
     Intended to be used with multiprocessing
  """
  return_dict[procnum] = calcDiff(TopFile, TrjFile, SurfType, method=Method)


def getAvgTP(logFile):
  """Given a log file for a GROMACS trajectory, finds the average temperature, pressure, and z-pressure.
     Inputs:
             logFile - name of log file (absolute path or from current directory)
     Outputs:
             avgT - average temperature
             avgP - average pressure
             avgPzz - average z-pressure (zz component of pressure tensor)
             xyzDims - list of x, y, and z box dimensions during the course of the NVT production run
                       For this part, assumes log file and output gro file have same file prefix.
  """

  TPOrderpipe = subprocess.Popen("grep 'Temperature' %s | tail -n 1"%logFile, 
                                                                     stdout=subprocess.PIPE, shell=True)

  TPOrder = TPOrderpipe.communicate()[0]
  TPOrderList = [TPOrder[k:k+15] for k in range(0, len(TPOrder), 15)]
  Tind = [k for k, elem in enumerate(TPOrderList) if 'Temperature' in elem][0]
  Pind = [k for k, elem in enumerate(TPOrderList) if 'Pressure' in elem][0]

  avgTPpipe = subprocess.Popen("grep -A1 'Temperature' %s | tail -n 1"%logFile, 
                                                                     stdout=subprocess.PIPE, shell=True)
  avgTPList = avgTPpipe.communicate()[0].strip().split()
  avgT = float(avgTPList[Tind])

  avgP = float(avgTPList[Pind])

  avgPzzpipe = subprocess.Popen("grep -A3 '  Pressure' %s | tail -n 1 | awk '{ print $3 }'"%logFile, 
                                                                     stdout=subprocess.PIPE, shell=True)
  avgPzz = float(avgPzzpipe.communicate()[0].strip())

  xyzDimspipe = subprocess.Popen("tail -n 1 %s"%(logFile[:-3]+'gro'), stdout=subprocess.PIPE, shell=True)

  xyzDims = xyzDimspipe.communicate()[0].strip().split()

  xyzDims = [float(x) for x in xyzDims]

  return avgT, avgP, avgPzz, xyzDims


def calcdGsolv(simDirs, kXY, refX, refY, distRefX, distRefY, numStates, endTime=-1, kB=0.008314459848, T=298.15, verbose=False):
  """This script runs a free energy calculation for inserting molecule(s) on a surface using simulations with different restraints.
     May seem like a lot of inputs, but provides flexibility, and can hard code if really want to.
     It is assumed below that the restraints are just in the x and y direction and are flat-bottom in form.
     Note that the spring constants and reference distances, etc. should be given in kJ/mol*A^2 and Angstroms, respectively.
     This is because by default the NetCDF trajectory writer uses Angstroms and so does pytraj.
     Inputs:
             simDirs - list of the directories containing each simulation with a specific restraining potential to assist sampling
             kXY - array or list of the spring constants for restraints in the x and y directions
             refX - array or list of the reference positions in the x-coordinate to calculate distances from (minimum image)
             refY - array or list of the reference positions in the y-coordinate to calculate distances from (minimum image)
             distRefX - array or list of the reference distances in the x-coordinate for the harmonic, 
                        flat-bottom potential (below this distance, no bias, above biases with harmonic potential
             distRefY - same as above, but for y-coordinate
             numStates - the number of alchemical states
             endTime - (default -1) the number of frames to read, with -1 being all of them
             kB - (default in kJ/mol*K) Boltzmann Constant in desired units
             T - (default 298.15 K) temperature
             verbose - (default False) if true, also prints full dG information across all states
     Outputs:
             dGout - the free energy of solvating the solute restrained at the interface, with xy restraints removed
             dGoutErr - estimated error in the free energy from MBAR (note that have NOT taken uncorrelated set of samples)
             samplesFirst - total number of samples at the first alchemical state
             samplesLast - total samples at the last alchemical state
             minFirst - minimum number of samples in the histogram for just the first state
             maxFirst - max number samples in first state histogram
             minLast - same as above but for last alchemical state
             maxLast - same as above but for last alchemical state
"""
  #First define some constants
  kBT = kB*T
  beta = 1.0 / kBT

  #First make sure all the input arrays have the same dimensions
  numSims = len(simDirs)
  allLens = np.array([len(a) for a in [kXY, refX, refY, distRefX, distRefY]])
  if not np.all(allLens == numSims):
    print("Need to provide same number of inputs for spring constants, etc. as number of simulation directories. Returning nones.")
    dGout = None
    dGoutErr = None
    return dGout 

  #Want to loop over all trajectories provided, storing solute position information to calculate restraints
  xyPos = None #X and Y coordinates of first heavy atom for all solutes - get shape later
  wrapxyPos = None #Wrapped X and Y coordinates of solutes relative to reference on surface
  nSamps = np.zeros((len(simDirs), numStates)) #Have as many x-y restraints as sims and then the same number of lambda states for each 
  allPots = np.array([[]]*numStates).T #Potential energies, EXCLUDING RESTRAINT, for each simulation frame and lambda state
  alchStates = np.array([]) #Will store alchemical state for each configuration
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
    cuIndices = []
    for res in top.residues:
      if res.name not in ['OTM', 'CTM', 'STM', 'NTM', 'SOL']: #Assumes working with SAM surface...
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
      alchStates = np.hstack((alchStates, alcDat[:, 2]))
    else:
      thisPot = alcDat[:endTime, 3:-1]
      alchStates = np.hstack((alchStates, alcDat[:endTime, 2]))

    #Next load in the trajectory and get all solute coordinates that matter
    top.rb_torsions = pmd.TrackedList([])
    top = pt.load_parmed(top, traj=False)
    if endTime == -1:
      traj = pt.iterload(trajFile, top, frame_slice=(startFrame, -1))
    else:
      traj = pt.iterload(trajFile, top, frame_slice=(startFrame, startFrame+endTime))
    nFrames = len(traj)
    xyBox = np.array(traj[0].box.values)[:2] #A little lazy, but all boxes should be same and fixed in X and Y dimensions

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
      thisxyPos[j,:] = thisXY 
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
  #Note we ignore the pV term, as it won't matter here because all states have same V for given configuration
  #ACTUALLY, pV term could matter for removing restraints in each surface quadrant, but probably negligible...
  #In other words, would have to have significant change in volume for solute at one surface location versus another
  #Also note that we haven't actually sampled the states we're interested in (i.e. unrestrained)
  #So we have to reweight, or use perturbation to get the free energies we're interested in
  mbarObj = mbar.MBAR(Ukn, nSamps.flatten())
  dG, dGerr = mbarObj.computePerturbedFreeEnergies(allPots.T)
  dGout = (-1.0) * dG[0][-1] #CAREFUL!  I promised the solvation free energy, so negating because last state is decoupled (ideal gas)
                          #The first state is the coupled state, so really want first state minus last
                          #To really be correct, also need to divide by the total number of solutes
  dGoutErr = dGerr[0][-1]

  if verbose:
    print('All free energy differences from state zero:')
    print(dG[0].tolist())
    print(dGerr[0].tolist())
    #To make it easy to recompute free energies, etc., will pickle the MBAR object
    with open('mbar_object.pkl', 'w') as outfile:
      pickle.dump(mbarObj, outfile)
    #For correct reweighting, will also need to save the potential energies without xy restraints
    np.savetxt('alchemical_U_noXYres.txt', allPots, 
               header='Potential energies at all interaction (lambda) states, xy restraints removed')
    #Also save the XY and wrapped XY positions of first heavy atom in solute, just in case need it later
    for k in range(len(heavyIndices)):
      np.savetxt('solute_%i_XY.txt'%(k), np.hstack((xyPos[:,k,:], wrapxyPos[:,k,:])), 
                 header='Raw solute 1st heavy atom X and Y, then XY wrapped around surface first reside U10 or U20 atom')

  #Should really also return some metric to demonstrate that solutes are sampling xy config space well...
  #Ideally demonstrate for both the decoupled state AND the coupled state
  #Histogram for all samples from fully coupled and decoupled states
  xWidth = xyBox[0] / 10.0
  yWidth = xyBox[1] / 10.0
  xBins = np.arange(-0.5*xyBox[1], 0.5*xyBox[0]+0.01, xWidth)
  yBins = np.arange(-0.5*xyBox[1], 0.5*xyBox[1]+0.01, yWidth)
  indsFirst = np.where(alchStates == 0)[0]
  indsLast = np.where(alchStates == (numStates-1))[0]
  samplesFirst = len(indsFirst)
  samplesLast = len(indsLast)
  histFirst = np.zeros((len(xBins)-1, len(yBins)-1))
  histLast = np.zeros((len(xBins)-1, len(yBins)-1))
  for k in range(len(heavyIndices)):
    thishistf, xedge, yedge = np.histogram2d(wrapxyPos[indsFirst,k,0], wrapxyPos[indsFirst,k,1], 
                                             bins=[xBins, yBins])
    thishistl, xedge, yedge = np.histogram2d(wrapxyPos[indsLast,k,0], wrapxyPos[indsLast,k,1], 
                                             bins=[xBins, yBins])
    histFirst += thishistf
    histLast += thishistl
  minFirst = np.min(histFirst)
  maxFirst = np.max(histFirst)
  minLast = np.min(histLast)
  maxLast = np.max(histLast)

  return dGout, dGoutErr, nSamps, samplesFirst, samplesLast, minFirst, maxFirst, minLast, maxLast


def doJobCalcdGsolv(procnum, return_dict, SimDirs, KXY, RefX, RefY, DistRefX, DistRefY, NumStates):
  """A wrapper function for calcdGsolv so that calcdGsolv may be run in parellel
     if desired. Takes additional parameters procnum and return_dict, so puts
     output of call to calcdGsolv in a global dictionary passed with the procnum
     as the key. This way, can run as many calcdGsolv in parallel as like, then
     reference each by the procnum it was given.
     Intended to be used with multiprocessing
  """
  return_dict[procnum] = calcdGsolv(SimDirs, KXY, RefX, RefY, DistRefX, DistRefY, NumStates)


def getAvgTV(logFile):
  """Given a log file from an OpenMM simulation, calculates the average temperature and box volume.
  """
  allDat = np.loadtxt(logFile)
  avgT = np.average(allDat[:, 5])
  avgV = np.average(allDat[:, 6])
  return avgT, avgV


def analyzeOptQuartz(libraryFile, gensize=8, topCartoons=1, optMax=False):
  """Analyzes the result of an optimization, plotting average and optimum diffusivity versus time.
     Input:
            libraryFile - pickled file containing the structure library
            gensize - (default 8) size of a generation
            topCartoons - (default 1) number of top-performers to create cartoon representations for
            optMax - (default False) whether optimization is max (True) or min (False)
     Output:
            None, just generates plots
  """

  import matplotlib
  showplots = True
  try:
    os.environ["DISPLAY"]
  except KeyError:
    showplots = False
    matplotlib.use('Agg')
  import matplotlib.pyplot as plt

  #Read in pickle file
  with open(libraryFile, 'r') as infile:
    allstructs = pickle.load(infile)

  #Get all diffusivities
  alldiffs = [struct.metric for struct in allstructs]

  #Make sure structures are sorted by diffusivity
  allstructs = [x for (y,x) in sorted(zip(alldiffs, allstructs))]

  #And generations
  allgens = [struct.gen for struct in allstructs]

  #Get rid of any seeds... should have generation of -1
  seedstructs = np.where(np.array(allgens) == -1)[0]
  print seedstructs
  print len(allgens)
  seedmask = np.ones(len(allstructs), dtype=bool)
  seedmask[seedstructs] = 0
  alldiffs = np.array(alldiffs)[seedmask].tolist()
  allstructs = np.array(allstructs)[seedmask].tolist()
  allgens = np.array(allgens)[seedmask].tolist()
  print len(allgens)

  print "Minimum diffusivity: %f (A^2/ps), %s" % (allstructs[0].metric, allstructs[0].topFile)
  print "Maximum diffusivity: %f (A^2/ps), %s" % (allstructs[-1].metric, allstructs[-1].topFile)

  #for struct in allstructs:
  #  print "%f,  %s" % (struct.metric, struct.topFile)

  sortbygen = [x for (y,x) in sorted(zip(allgens, alldiffs))]
  sortbygensurfs = [x for (y,x) in sorted(zip(allgens, allstructs))]

  convexhull = np.zeros(len(sortbygen)/gensize)

  if optMax:
    optdiffgen = [np.max(sortbygen[k:k+gensize]) for k in range(0,len(sortbygen)-7,gensize)]
    for k, diff in enumerate(optdiffgen):
      if k == 0:
        convexhull[k] = diff
      else:
        if diff > convexhull[k-1]:
          convexhull[k] = diff
        else:
          convexhull[k] = convexhull[k-1]
  else:
    optdiffgen = [np.min(sortbygen[k:k+gensize]) for k in range(0,len(sortbygen)-7,gensize)]
    for k, diff in enumerate(optdiffgen):
      if k == 0:
        convexhull[k] = diff
      else:
        if diff < convexhull[k-1]:
          convexhull[k] = diff
        else:
          convexhull[k] = convexhull[k-1]

  #Compute stats for single generation
  avgdiffgen = np.zeros(len(sortbygen)/gensize)
  stddiffgen = np.zeros(len(sortbygen)/gensize)
  rmsdgen = np.zeros(len(sortbygen)/gensize)
  for l, k in enumerate(range(0, len(sortbygen)-7, gensize)):
    avgdiffgen[l] = np.average(sortbygen[k:k+gensize])
    stddiffgen[l] = np.std(sortbygen[k:k+gensize])
    #To get average rmsd between structures in this generation, have to loop over pairs
    temprmsd = []
    currgensurfs = sortbygensurfs[k:k+gensize]
    for i in range(len(currgensurfs)):
      for j in range(i+1, len(currgensurfs)):
        currrmsd = strucRMSD(currgensurfs[i].condPairs, currgensurfs[j].condPairs)
        temprmsd.append(currrmsd)
    rmsdgen[l] = np.average(temprmsd)
    print "Max RMSD at generation %i is %f" % (k/gensize, np.max(temprmsd))
    print "Min RMSD at generation %i is %f" % (k/gensize, np.min(temprmsd))
    #Also perform clustering for all structures up through current generation
    thisClustList = clusterSurfs(sortbygensurfs[:k+gensize], 'quartz', 0.44, MaxIter=200, MaxCluster=-32, MaxClusterWork=None, Verbose=False)
    print len(thisClustList)
    maxmindiffs = np.zeros((len(thisClustList), 2))
    for i, alist in enumerate(thisClustList):
      print len(alist)
      adifflist = [struct.metric for struct in alist]
      maxmindiffs[i,0] = np.min(adifflist)
      maxmindiffs[i,1] = np.max(adifflist)
    print "At generation %i, have %i clusters each with min/max diffusivities:" % (k/gensize, len(thisClustList))
    print maxmindiffs
 
  for agen, aval in enumerate(rmsdgen):
    print "%i:  %f" % (agen, aval)  

  dFig, dAx = plt.subplots(3, sharex=True)
  dAx[0].plot(range(len(sortbygen)/gensize), optdiffgen, 'o')
  if optMax:
    dAx[0].set_ylabel(r'Maximum Diffusivity ($\AA^2/ps$)')
  else:
    dAx[0].set_ylabel(r'Minimum Diffusivity ($\AA^2/ps$)')
  dAx[0].plot(range(len(sortbygen)/gensize), convexhull, 'k-', linewidth=2.5)
  dAx[1].errorbar(range(len(sortbygen)/gensize), avgdiffgen, yerr=stddiffgen, marker='o', linestyle='-')
  dAx[2].plot(range(len(sortbygen)/gensize), rmsdgen, 'o')
  dAx[1].set_ylabel(r'Average Diffusivity ($\AA^2/ps$)')
  dAx[2].set_ylabel(r'Average RMSD between individuals')
  dAx[2].set_xlabel(r'Generation')
  dAx[2].set_xlim((-1,len(sortbygen)/gensize))
  dAx[0].axes.get_yaxis().set_major_locator(matplotlib.ticker.MaxNLocator(prune='both'))
  dAx[1].axes.get_yaxis().set_major_locator(matplotlib.ticker.MaxNLocator(prune='both'))
  dAx[2].axes.get_yaxis().set_major_locator(matplotlib.ticker.MaxNLocator(prune='both'))
  dFig.subplots_adjust(hspace=0.0)

  #Also make cartoon of as many top-performing surfaces as desired
  if topCartoons >= 4:
    cartoonFig, cartoonAx = plt.subplots(int(np.ceil(topCartoons/4.0)), 4)
  else:
    cartoonFig, cartoonAx = plt.subplots(1, topCartoons, figsize=(4*topCartoons,4))
    if topCartoons == 1:
      cartoonAx = np.array([[cartoonAx]])
    else:
      cartoonAx = np.array([cartoonAx])
  if optMax:
    optstructs = allstructs[-topCartoons:]
    optstructs.reverse() #Flip so most 'fit' (highest diffusivity here) first
  else:
    optstructs = allstructs[:topCartoons]
  cmap = plt.get_cmap('Reds')

  for k, optstruct in enumerate(optstructs):
    row = k/4
    col = k%4
    optcartoon, optbonds = optstruct.makeCartoon()
    #plot oxygen positions
    cartoonAx[row, col].imshow(optcartoon, aspect=(optcartoon.shape[1]/2.0)/optcartoon.shape[0], 
                                           extent=(0,optcartoon.shape[1]/2,optcartoon.shape[0],0), 
                                                   interpolation='none', origin='upper', cmap=cmap)
    #plot bonds
    for bondpoints in optbonds:
      cartoonAx[row, col].plot(bondpoints[:,1]*0.5+0.25, bondpoints[:,0]+0.5, '-k')
    #and make look pretty
    for xgridtick in np.arange(0.0, optcartoon.shape[1]/2 + 0.5, 0.5):
      for ygridtick in np.arange(optcartoon.shape[0]+1):
        cartoonAx[row, col].plot([xgridtick, xgridtick], [0.0, optcartoon.shape[0]], 'k-', linewidth=2.0)
        cartoonAx[row, col].plot([0.0, optcartoon.shape[1]/2], [ygridtick, ygridtick], 'k-', linewidth=2.0)
    cartoonAx[row, col].axes.get_xaxis().set_ticks([])
    cartoonAx[row, col].axes.get_yaxis().set_ticks([])
    for axis in ['top', 'bottom', 'left', 'right']:
      cartoonAx[row, col].spines[axis].set_linewidth(0.0)
    cartoonAx[row, col].axes.get_xaxis().set_ticklabels([])
    cartoonAx[row, col].axes.get_yaxis().set_ticklabels([])
    cartoonAx[row, col].set_xlim((-0.5, optcartoon.shape[1]/2 + 0.5))
    cartoonAx[row, col].set_ylim((-1.0, optcartoon.shape[0] + 0.5))
    cartoonAx[row, col].set_title(r'%i, %s, $D=%f$' % (k, optstruct.topFile.split('/')[1], optstruct.metric), fontsize=11)
  
  cartoonFig.tight_layout()

  if showplots:
    plt.show()
  else:
    dFig.savefig('optdiff_analysis.png')
    cartoonFig.savefig('optimum_cartoon.png')


def analyzeOptCristab(libraryFile, gensize=8, topCartoons=1, optMax=False):
  """Analyzes the result of an optimization, plotting average and optimum diffusivity versus time.
     Input:
            libraryFile - pickled file containing the structure library
            gensize - (default 8) size of a generation
            topCartoons - (default 1) number of top-performers to create cartoon representations for
            optMax - (default False) whether optimization is max (True) or min (False)
     Output:
            None, just generates plots
  """

  import matplotlib
  showplots = True
  try:
    os.environ["DISPLAY"]
  except KeyError:
    showplots = False
    matplotlib.use('Agg')
  import matplotlib.pyplot as plt

  #Read in pickle file
  with open(libraryFile, 'r') as infile:
    allstructs = pickle.load(infile)

  #Get all diffusivities
  alldiffs = [struct.metric for struct in allstructs]

  #Make sure structures are sorted by diffusivity
  allstructs = [x for (y,x) in sorted(zip(alldiffs, allstructs))]

  #And generations
  allgens = [struct.gen for struct in allstructs]

  #Get rid of any seeds... should have generation of -1
  seedstructs = np.where(np.array(allgens) == -1)[0]
  print seedstructs
  print len(allgens)
  seedmask = np.ones(len(allstructs), dtype=bool)
  seedmask[seedstructs] = 0
  alldiffs = np.array(alldiffs)[seedmask].tolist()
  allstructs = np.array(allstructs)[seedmask].tolist()
  allgens = np.array(allgens)[seedmask].tolist()
  print len(allgens)

  print "Minimum diffusivity: %f (A^2/ps), %s" % (allstructs[0].metric, allstructs[0].topFile)
  print "Maximum diffusivity: %f (A^2/ps), %s" % (allstructs[-1].metric, allstructs[-1].topFile)

  #for struct in allstructs:
  #  print "%f,  %s" % (struct.metric, struct.topFile)

  sortbygen = [x for (y,x) in sorted(zip(allgens, alldiffs))]
  sortbygensurfs = [x for (y,x) in sorted(zip(allgens, allstructs))]

  convexhull = np.zeros(len(sortbygen)/gensize)

  if optMax:
    optdiffgen = [np.max(sortbygen[k:k+gensize]) for k in range(0,len(sortbygen)-7,gensize)]
    for k, diff in enumerate(optdiffgen):
      if k == 0:
        convexhull[k] = diff
      else:
        if diff > convexhull[k-1]:
          convexhull[k] = diff
        else:
          convexhull[k] = convexhull[k-1]
  else:
    optdiffgen = [np.min(sortbygen[k:k+gensize]) for k in range(0,len(sortbygen)-7,gensize)]
    for k, diff in enumerate(optdiffgen):
      if k == 0:
        convexhull[k] = diff
      else:
        if diff < convexhull[k-1]:
          convexhull[k] = diff
        else:
          convexhull[k] = convexhull[k-1]

  #Compute stats for single generation
  avgdiffgen = np.zeros(len(sortbygen)/gensize)
  stddiffgen = np.zeros(len(sortbygen)/gensize)
  rmsdgen = np.zeros(len(sortbygen)/gensize)
  minrmsdgen = np.zeros(len(sortbygen)/gensize)
  maxrmsdgen = np.zeros(len(sortbygen)/gensize)
  minrmsdclust = np.zeros(len(sortbygen)/gensize)
  maxrmsdclust = np.zeros(len(sortbygen)/gensize)
  maxminindrmsd = np.zeros(len(sortbygen)/gensize)
  rmsdDistFig, rmsdDistAx = plt.subplots(1)
  for l, k in enumerate(range(0, len(sortbygen)-7, gensize)):
    avgdiffgen[l] = np.average(sortbygen[k:k+gensize])
    stddiffgen[l] = np.std(sortbygen[k:k+gensize])
    #To get average rmsd between structures in this generation, have to loop over pairs
    temprmsd = []
    currgensurfs = sortbygensurfs[k:k+gensize]
    for i in range(len(currgensurfs)):
      for j in range(i+1, len(currgensurfs), 1):
        currrmsd = strucRMSD(currgensurfs[i].condPairs, currgensurfs[j].condPairs)
        temprmsd.append(currrmsd)
    rmsdgen[l] = np.average(temprmsd)
    tempfullrmsd = []
    fullgensurfs = sortbygensurfs[0:k+gensize]
    individualmins = 1000.0*np.ones(len(fullgensurfs))
    for i in range(len(fullgensurfs)):
      for j in range(i+1, len(fullgensurfs), 1):
        currrmsd = strucRMSD(fullgensurfs[i].condPairs, fullgensurfs[j].condPairs)
        if currrmsd == 0.0:
          print "At generation %i, RMSD is zero between structures %s and %s." % (l, fullgensurfs[i].topFile, fullgensurfs[j].topFile)
          print "These structures have condPairs sum of differences of:"
          print np.sum(fullgensurfs[i].condPairs - fullgensurfs[j].condPairs)
        tempfullrmsd.append(currrmsd)
        if currrmsd < individualmins[i]:
          individualmins[i] = currrmsd
        if currrmsd < individualmins[j]:
          individualmins[j] = currrmsd
    minrmsdgen[l] = np.min(tempfullrmsd)
    maxrmsdgen[l] = np.max(tempfullrmsd)
    maxminindrmsd[l] = np.max(individualmins)
    print "Maximum min invidual RMSD is: %f" % np.max(individualmins)
    if k == range(0, len(sortbygen)-7, gensize)[-1]:
      rmsdhist, rmsdbins = np.histogram(tempfullrmsd, bins=30)
      rmsdbinmids = 0.5*(rmsdbins[1:] + rmsdbins[:-1])
      barwidth = rmsdbinmids[1] - rmsdbinmids[0]
      rmsdDistAx.bar(rmsdbinmids, rmsdhist, barwidth)
      rmsdDistAx.set_xlabel('RMSD')
      rmsdDistAx.set_ylabel('Histogram Count')
    #print "Max RMSD at generation %i is %f" % (k/gensize, np.max(temprmsd))
    #print "Min RMSD at generation %i is %f" % (k/gensize, np.min(temprmsd))
    #Also perform clustering for all structures up through current generation
    thisClustList = clusterSurfs(sortbygensurfs[:k+gensize], 'quartz', 0.44, MaxIter=10, MaxCluster=-32, MaxClusterWork=None, Verbose=False)
    maxmindiffs = np.zeros((len(thisClustList), 2))
    avgstructs = []
    for i, alist in enumerate(thisClustList):
      adifflist = [struct.metric for struct in alist]
      maxmindiffs[i,0] = np.min(adifflist)
      maxmindiffs[i,1] = np.max(adifflist)
      avgstructs.append(np.average([asurf.condPairs for asurf in alist], axis=0))
    clustclustRMSDs = []
    for i in range(len(avgstructs)):
      for j in range(i+1, len(avgstructs), 1):
        clustclustRMSDs.append(strucRMSD(avgstructs[i], avgstructs[j]))
    minrmsdclust[l] = np.min(clustclustRMSDs)
    maxrmsdclust[l] = np.max(clustclustRMSDs)
    print "At generation %i, have %i clusters each with min/max diffusivities:" % (k/gensize, len(thisClustList))
    for thesediffs in maxmindiffs:
      print thesediffs
    print "And have minimum and maximum RMSD between clusters of:   %f,  %f" % (np.min(clustclustRMSDs), np.max(clustclustRMSDs))
    
  for agen, aval in enumerate(rmsdgen):
    print "%i:  %f" % (agen, aval)  

  dFig, dAx = plt.subplots(4, sharex=True)
  dAx[0].plot(range(len(sortbygen)/gensize), optdiffgen, 'o')
  if optMax:
    dAx[0].set_ylabel(r'Maximum Diffusivity ($\AA^2/ps$)')
  else:
    dAx[0].set_ylabel(r'Minimum Diffusivity ($\AA^2/ps$)')
  dAx[0].plot(range(len(sortbygen)/gensize), convexhull, 'k-', linewidth=2.5)
  dAx[1].errorbar(range(len(sortbygen)/gensize), avgdiffgen, yerr=stddiffgen, marker='o', linestyle='-')
  dAx[2].plot(range(len(sortbygen)/gensize), rmsdgen, 'o')
  dAx[3].plot(range(len(sortbygen)/gensize), minrmsdgen, label='Min overall RMSD')
  dAx[3].plot(range(len(sortbygen)/gensize), maxrmsdgen, label='Max overall RMSD')
  dAx[3].plot(range(len(sortbygen)/gensize), minrmsdclust, label='Min cluster RMSD')
  dAx[3].plot(range(len(sortbygen)/gensize), maxrmsdclust, label='Max cluster RMSD')
  dAx[3].plot(range(len(sortbygen)/gensize), maxminindrmsd, label='Max min RMSD')
  dAx[1].set_ylabel(r'Average Diffusivity ($\AA^2/ps$)')
  dAx[2].set_ylabel(r'Average RMSD between individuals')
  dAx[2].set_xlim((-1,len(sortbygen)/gensize))
  dAx[3].set_ylabel('RMSD')
  dAx[3].set_xlabel(r'Generation')
  dAx[3].legend()
  dAx[0].axes.get_yaxis().set_major_locator(matplotlib.ticker.MaxNLocator(prune='both'))
  dAx[1].axes.get_yaxis().set_major_locator(matplotlib.ticker.MaxNLocator(prune='both'))
  dAx[2].axes.get_yaxis().set_major_locator(matplotlib.ticker.MaxNLocator(prune='both'))
  dAx[3].axes.get_yaxis().set_major_locator(matplotlib.ticker.MaxNLocator(prune='both'))
  dFig.tight_layout()
  dFig.subplots_adjust(hspace=0.0)

  #Also make cartoon of as many top-performing surfaces as desired
  if topCartoons >= 4:
    cartoonFig, cartoonAx = plt.subplots(int(np.ceil(topCartoons/4.0)), 4)
  else:
    cartoonFig, cartoonAx = plt.subplots(1, topCartoons, figsize=(4*topCartoons,4))
    if topCartoons == 1:
      cartoonAx = np.array([[cartoonAx]])
    else:
      cartoonAx = np.array([cartoonAx])
  if optMax:
    optstructs = allstructs[-topCartoons:]
    optstructs.reverse() #Flip so most 'fit' (highest diffusivity here) first
  else:
    optstructs = allstructs[:topCartoons]
  cmap = plt.get_cmap('Reds')

  for k, optstruct in enumerate(optstructs):
    row = k/4
    col = k%4
    optcartoon, optbonds = optstruct.makeCartoon()
    #plot oxygen positions
    cartoonAx[row, col].imshow(optcartoon, aspect=optcartoon.shape[1]/optcartoon.shape[0], 
                                           extent=(0,optcartoon.shape[1],optcartoon.shape[0],0), 
                                                   interpolation='none', origin='upper', cmap=cmap)
    #plot bonds
    for bondpoints in optbonds:
      cartoonAx[row, col].plot(bondpoints[:,1]+0.5, bondpoints[:,0]+0.5, '-k')
    #and make look pretty
    for xgridtick in np.arange(optcartoon.shape[1]+1):
      for ygridtick in np.arange(optcartoon.shape[0]+1):
        cartoonAx[row, col].plot([xgridtick, xgridtick], [0.0, optcartoon.shape[0]], 'k-', linewidth=2.0)
        cartoonAx[row, col].plot([0.0, optcartoon.shape[1]], [ygridtick, ygridtick], 'k-', linewidth=2.0)
    cartoonAx[row, col].axes.get_xaxis().set_ticks([])
    cartoonAx[row, col].axes.get_yaxis().set_ticks([])
    for axis in ['top', 'bottom', 'left', 'right']:
      cartoonAx[row, col].spines[axis].set_linewidth(0.0)
    cartoonAx[row, col].axes.get_xaxis().set_ticklabels([])
    cartoonAx[row, col].axes.get_yaxis().set_ticklabels([])
    cartoonAx[row, col].set_xlim((-0.5, optcartoon.shape[1] + 0.5))
    cartoonAx[row, col].set_ylim((-0.5, optcartoon.shape[0] + 0.5))
    cartoonAx[row, col].set_title(r'%i, %s, $D=%f$' % (k, optstruct.topFile.split('/')[1], optstruct.metric), fontsize=11)
  
  cartoonFig.tight_layout()

  if showplots:
    plt.show()
  else:
    dFig.savefig('optdiff_analysis.png')
    cartoonFig.savefig('optimum_cartoon.png')
    rmsdDistFig.savefig('RMSD_histogram.png')


def analyzeOptSAM(libraryFile, gensize=8, topCartoons=1, optMax=False):
  """Analyzes the result of an optimization, plotting average and optimum diffusivity versus time.
     Input:
            libraryFile - pickled file containing the structure library
            gensize - (default 8) size of a generation
            topCartoons - (default 1) number of top-performers to create cartoon representations for
            optMax - (default False) whether optimization is max (True) or min (False)
     Output:
            None, just generates plots
  """

  import matplotlib
  showplots = True
  try:
    os.environ["DISPLAY"]
  except KeyError:
    showplots = False
    matplotlib.use('Agg')
  import matplotlib.pyplot as plt

  #Read in pickle file
  with open(libraryFile, 'r') as infile:
    allstructs = pickle.load(infile)

  #Get all diffusivities
  alldiffs = [struct.metric for struct in allstructs]

  #Make sure structures are sorted by diffusivity
  allstructs = [x for (y,x) in sorted(zip(alldiffs, allstructs))]

  #And generations
  allgens = [struct.gen for struct in allstructs]

  #Get rid of any seeds... should have generation of -1
  seedstructs = np.where(np.array(allgens) == -1)[0]
  print seedstructs
  print len(allgens)
  seedmask = np.ones(len(allstructs), dtype=bool)
  seedmask[seedstructs] = 0
  alldiffs = np.array(alldiffs)[seedmask].tolist()
  allstructs = np.array(allstructs)[seedmask].tolist()
  allgens = np.array(allgens)[seedmask].tolist()
  print len(allgens)

  print "Minimum diffusivity: %f (A^2/ps), %s" % (allstructs[0].metric, allstructs[0].topFile)
  print "Maximum diffusivity: %f (A^2/ps), %s" % (allstructs[-1].metric, allstructs[-1].topFile)

  #for struct in allstructs:
  #  print "%f,  %s" % (struct.metric, struct.topFile)

  sortbygen = [x for (y,x) in sorted(zip(allgens, alldiffs))]
  sortbygensurfs = [x for (y,x) in sorted(zip(allgens, allstructs))]

  convexhull = np.zeros(len(sortbygen)/gensize)

  if optMax:
    optdiffgen = [np.max(sortbygen[k:k+gensize]) for k in range(0,len(sortbygen)-7,gensize)]
    for k, diff in enumerate(optdiffgen):
      if k == 0:
        convexhull[k] = diff
      else:
        if diff > convexhull[k-1]:
          convexhull[k] = diff
        else:
          convexhull[k] = convexhull[k-1]
  else:
    optdiffgen = [np.min(sortbygen[k:k+gensize]) for k in range(0,len(sortbygen)-7,gensize)]
    for k, diff in enumerate(optdiffgen):
      if k == 0:
        convexhull[k] = diff
      else:
        if diff < convexhull[k-1]:
          convexhull[k] = diff
        else:
          convexhull[k] = convexhull[k-1]

  #Compute stats for single generation
  avgdiffgen = np.zeros(len(sortbygen)/gensize)
  stddiffgen = np.zeros(len(sortbygen)/gensize)
  rmsdgen = np.zeros(len(sortbygen)/gensize)
  for l, k in enumerate(range(0, len(sortbygen)-7, gensize)):
    avgdiffgen[l] = np.average(sortbygen[k:k+gensize])
    stddiffgen[l] = np.std(sortbygen[k:k+gensize])
    #To get average rmsd between structures in this generation, have to loop over pairs
    temprmsd = []
    currgensurfs = sortbygensurfs[k:k+gensize]
    for i in range(len(currgensurfs)):
      for j in range(i+1, len(currgensurfs)):
        currrmsd = strucRMSD(currgensurfs[i].chainTypes, currgensurfs[j].chainTypes)
        temprmsd.append(currrmsd)
    rmsdgen[l] = np.average(temprmsd)
    print "Max RMSD at generation %i is %f" % (k/gensize, np.max(temprmsd))
    print "Min RMSD at generation %i is %f" % (k/gensize, np.min(temprmsd))
    #Also perform clustering for all structures up through current generation
    thisClustList = clusterSurfs(sortbygensurfs[:k+gensize], 'SAM', 0.44, MaxIter=200, MaxCluster=-32, MaxClusterWork=None, Verbose=False)
    print len(thisClustList)
    maxmindiffs = np.zeros((len(thisClustList), 2))
    for i, alist in enumerate(thisClustList):
      print len(alist)
      adifflist = [struct.metric for struct in alist]
      maxmindiffs[i,0] = np.min(adifflist)
      maxmindiffs[i,1] = np.max(adifflist)
    print "At generation %i, have %i clusters each with min/max diffusivities:" % (k/gensize, len(thisClustList))
    print maxmindiffs
    
  for agen, aval in enumerate(rmsdgen):
    print "%i:  %f" % (agen, aval)  

  dFig, dAx = plt.subplots(3, sharex=True)
  dAx[0].plot(range(len(sortbygen)/gensize), optdiffgen, 'o')
  if optMax:
    dAx[0].set_ylabel(r'Maximum Diffusivity ($\AA^2/ps$)')
  else:
    dAx[0].set_ylabel(r'Minimum Diffusivity ($\AA^2/ps$)')
  dAx[0].plot(range(len(sortbygen)/gensize), convexhull, 'k-', linewidth=2.5)
  dAx[1].errorbar(range(len(sortbygen)/gensize), avgdiffgen, yerr=stddiffgen, marker='o', linestyle='-')
  dAx[2].plot(range(len(sortbygen)/gensize), rmsdgen, 'o')
  dAx[1].set_ylabel(r'Average Diffusivity ($\AA^2/ps$)')
  dAx[2].set_ylabel(r'Average RMSD between individuals')
  dAx[2].set_xlabel(r'Generation')
  dAx[2].set_xlim((-1,len(sortbygen)/gensize))
  dAx[0].axes.get_yaxis().set_major_locator(matplotlib.ticker.MaxNLocator(prune='both'))
  dAx[1].axes.get_yaxis().set_major_locator(matplotlib.ticker.MaxNLocator(prune='both'))
  dAx[2].axes.get_yaxis().set_major_locator(matplotlib.ticker.MaxNLocator(prune='both'))
  dFig.subplots_adjust(hspace=0.0)

  #Also make cartoon of as many top-performing surfaces as desired
  if topCartoons >= 4:
    cartoonFig, cartoonAx = plt.subplots(int(np.ceil(topCartoons/4.0)), 4)
  else:
    cartoonFig, cartoonAx = plt.subplots(1, topCartoons, figsize=(4*topCartoons,4))
    if topCartoons == 1:
      cartoonAx = np.array([[cartoonAx]])
    else:
      cartoonAx = np.array([cartoonAx])
  if optMax:
    optstructs = allstructs[-topCartoons:]
    optstructs.reverse() #Flip so most 'fit' (highest diffusivity here) first
  else:
    optstructs = allstructs[:topCartoons]
  cmap = plt.get_cmap('Reds')

  for k, optstruct in enumerate(optstructs):
    row = k/4
    col = k%4
    optcartoon = optstruct.makeCartoon()
    #plot oxygen positions
    cartoonAx[row, col].imshow(optcartoon, aspect=optcartoon.shape[1]/optcartoon.shape[0], 
                                           extent=(0,optcartoon.shape[1],optcartoon.shape[0],0), 
                                                   interpolation='none', origin='upper', cmap=cmap)
    #and make look pretty
    for xgridtick in np.arange(optcartoon.shape[1]+1):
      for ygridtick in np.arange(optcartoon.shape[0]+1):
        cartoonAx[row, col].plot([xgridtick, xgridtick], [0.0, optcartoon.shape[0]], 'k-', linewidth=2.0)
        cartoonAx[row, col].plot([0.0, optcartoon.shape[1]], [ygridtick, ygridtick], 'k-', linewidth=2.0)
    cartoonAx[row, col].axes.get_xaxis().set_ticks([])
    cartoonAx[row, col].axes.get_yaxis().set_ticks([])
    for axis in ['top', 'bottom', 'left', 'right']:
      cartoonAx[row, col].spines[axis].set_linewidth(0.0)
    cartoonAx[row, col].axes.get_xaxis().set_ticklabels([])
    cartoonAx[row, col].axes.get_yaxis().set_ticklabels([])
    cartoonAx[row, col].set_xlim((-0.5, optcartoon.shape[1] + 0.5))
    cartoonAx[row, col].set_ylim((-0.5, optcartoon.shape[0] + 0.5))
    cartoonAx[row, col].set_title(r'%i, %s, $D=%f$' % (k, optstruct.topFile.split('/')[1], optstruct.metric), fontsize=11)
  
  cartoonFig.tight_layout()

  if showplots:
    plt.show()
  else:
    dFig.savefig('optdiff_analysis.png')
    cartoonFig.savefig('optimum_cartoon.png')


