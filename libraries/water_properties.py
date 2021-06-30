#A library of code to examine properties of bulk water and near solutes
#
#Should eventually be able to handle local densities and fluctuations,
#solute-water and water-water energies, 3-body angles, hydrogen bonds,
#energy densities, and all of this as a function of space. Additionally,
#should also be able to compute interfaces, such as Willard-Chandler
#instantaneous interface, or vdW surface, SASA and volume of solute.
#
#Will work with pytraj interface for trajectory analysis, since this
#should later allow easier energy decomposition?
#If doesn't work out, will go back to sim package with netcdf plugin.
#
#Also, should have test script and some test system where know answers
#

import sys, os
import numpy as np
import scipy.optimize as optimize
from scipy.special import sph_harm
import waterlib as wl

#Define constants and unit conversions

#conversion for surface tension
kBJ = 1.38064852*(10**(-23))
temp = 300.0
tomJm2 = kBJ*temp*1000.0*(10**20) #converts kBT/Angstrom^2 to mJ/m^2

#Convert potential energy to kBT
kBTkcal = 0.0019858775*300.0

#Water density
watdens = 0.033456 # molecules or oxygens per Angstrom ^ 3 near 300 K

#Define library of useful functions

def SASAperAtom(pos, radii, radius=1.4, nPoints = 1000, nExpose = 10):
  """Inputs:
     pos - Nx3 array of atomic positions
     radii - N array of atomic radii
     radius - solvent radius to "roll" over surface
     nPoints - number points on each sphere
     nExpose - number exposed points on atom (sphere) to be considered on surface
     Outputs:
     SASAper - SASA for each atom
     surfAtoms - array of 1 for solvent exposed, 0 for not on surface
  """

  points = wl.spherepoints(nPoints)
  SASAper, surfAtoms = wl.spheresurfaceareas(pos, radii+radius, points, nExpose)

  return SASAper, surfAtoms


def PepWatHBonds(allPos, pepAccInds, pepDonInds, watInds, distCut = 2.1, angCut = 30.0):
  """Currently kind of wack (does acceptor to hydrogen distance). Also, calculating 
H-bonds geometrically seems less useful.
     Inputs:
     allPos - full position array for trajectory frame (all atoms included)
     pepAccInds - global indices of peptide acceptors
     pepDonInds - global indices of peptide donors
     watInds - global indices of water atoms in selected hydration shell(s)
     distCut(=2.1) - distance cutoff for H-bond detection
     angCut(=30.0) - angle cutoff for H-bond detection
     Outputs:
     NBonds - number of detected H-bonds
     bondsPer - number H-bonds for each water molecule with peptide
     donors - indices of donors (H atoms only) as string
     acceptors - indices of acceptors as string
  """

  #Get H-bond info
  NBonds, watAcc, watDon, pepAcc, pepDon = wl.findhbonds(
                             allPos[pepAccInds], allPos[pepDonInds], allPos[watInds], distCut, angCut)

  #And sort nicely into just acceptors and donors
  acceptorsList = []
  donorsList = []
  bondsWat = np.zeros(int(len(watInds)/3))

  for (j, val) in enumerate(pepAcc):
    acceptorsList = acceptorsList + (val*[pepAccInds[j]])

  for (j, val) in enumerate(pepDon):
    donorsList = donorsList + (val*[pepDonInds[j]])

  for (j, val) in enumerate(watAcc):
    acceptorsList = acceptorsList + (val*[watInds[j]])
    bondsWat[int(j/3)] = bondsWat[int(j/3)] + val

  for (j, val) in enumerate(watDon):
    donorsList = donorsList + (val*[watInds[j]])
    bondsWat[int(j/3)] = bondsWat[int(j/3)] + val

  #Above uses properties of python lists to add each index the number of H-bonds it participates in

  bondsPer = bondsWat
  
  #For easy file writing, make donors and acceptors into strings of indices
  #Remember that the sim package indexes at zero!
  donors = ''.join(str(e)+"|" for e in donorsList)
  acceptors = ''.join(str(e)+"|" for e in acceptorsList)
    
  return NBonds, bondsPer, acceptors, donors


def BBHBonds(allPos, pepAccInds, pepDonInds, distCut = 2.1, angCut = 30.0):
  """Finds H bonds between two list of acceptors and donors. Intended for just peptide backbone.
     Inputs:
     allPos - full position array for trajectory frame
     pepAccInds - global indics of peptide acceptors
     pepDonInds - global indices of peptide doneors
     distCut(=2.1) - distance cutoff for H-bond detection
     angCut(=30.0) - angle cutoff for H-bond detection
     Outputs:
     NBonds - number of detected H-bonds
     donors - indices of donors as string
     acceptors - indices of acceptors as string
  """

  #Get H-bonds
  NBonds, pepAcc, pepDon = wl.bbhbonds(allPos[pepAccInds], allPos[pepDonInds], distCut, angCut)
  
  #Sort nicely
  acceptorsList = []
  donorsList = []

  for (j, val) in enumerate(pepAcc):
    acceptorsList = acceptorsList + (val*[pepAccInds[j]])

  for (j, val) in enumerate(pepDon):
    donorsList = donorsList + (val*[pepDonInds[j]])

  #set lists to strings and return
  donors = ''.join(str(e)+"|" for e in donorsList)
  acceptors = ''.join(str(e)+"|" for e in acceptorsList)
    
  return NBonds, acceptors, donors


def WatHBonds(allPos, watInds, allWatInds, BoxDims, distCut = 2.1, angCut = 30.0):
  """Also kind of wack, but keeping since used in peptide-surface pulling analysis.
     For a better, more general algorithm, use HBondsGeneral.
     Inputs:
     allPos - full position array for trajectory frame (all atoms included)
     watInds - global indices of water atoms in selected hydration shell(s)
     allWatInds - global indices of ALL water atoms
     BoxDims - dimensions of box to account for periodic BCs (to turn off, set to zero)
     distCut(=2.1) - distance cutoff for H-bond detection
     angCut(=30.0) - angle cutoff for H-bond detection
     Outputs:
     NBonds - number of detected H-bonds
     bondsPer - number of detected H-bonds for each water molecule in selection
     acceptors - indices of acceptors as string
     donors - indices of donors (H atoms only) as string
  """

  #Get H-bond info
  NBonds, watAcc, watDon = wl.wathbonds(allPos[watInds], allPos[allWatInds], BoxDims, distCut, angCut)

  #And sort nicely into just acceptors and donors
  #Also count number of H-bonds for each water to get estimate of average per water
  acceptorsList = []
  donorsList = []
  bondsWat = np.zeros(int(len(watInds)/3))

  for (j, val) in enumerate(watAcc):
    acceptorsList = acceptorsList + (val*[watInds[j]])
    bondsWat[int(j/3)] = bondsWat[int(j/3)] + val

  for (j, val) in enumerate(watDon):
    donorsList = donorsList + (val*[watInds[j]])
    bondsWat[int(j/3)] = bondsWat[int(j/3)] + val

  #Above uses properties of python lists to add each index the number of H-bonds it participates in

  #print bondsWat
  #bondsPer = np.average(bondsWat)
  bondsPer = bondsWat
  
  #For easy file writing, make donors and acceptors into strings of indices
  #Remember that the sim package indexes at zero!
  donors = ''.join(str(e)+"|" for e in donorsList)
  acceptors = ''.join(str(e)+"|" for e in acceptorsList)
    
  return NBonds, bondsPer, acceptors, donors


def getCosAngs(subPos, Pos, BoxDims, lowCut=0.0, highCut=3.413):
  """This is called getCosAngs, but actually just returns the angles themselves (faster to convert
     from cos(theta) to theta in Fortran)
     Inputs:
     subPos - positions of set of atoms to measure tetrahedrality of (may be different, subset, or same as Pos)
     Pos - positions of ALL atoms that can make tetrahedral configurations (needed if subPos not same as Pos)
     BoxDims - current box dimensions to account for periodicity
     lowCut - lower cutoff for nearest-neighbor shell (default 0.0)
     highCut - higher cutoff for nearest-neighbor shell (default 3.413 - see Chaimovich, 2014, but should really
               change to reflect first peak in g(r) for the chosen water model)
     Outputs:
     angVals - all angle values for current configuration of positions supplied
     numAngs - number of angles for each central oxygen atom (i.e. number neighbors factorial)
               This is useful for finding which angles belong to which central oxygens
               This return value was added on 07/09/2017, so any code using this function
               before then will break, unfortunately, but the fix is easy.
  """

  #Set-up array to hold angle results and stack as go... list increases in size!
  angVals = np.array([])
  numAngs = np.zeros(len(subPos))

  #Find nearest neighbors for ALL atoms in subPos
  #But make sure using efficient algorithm...
  #If subPos is same as Pos, use allnearneighbors instead
  if np.array_equal(subPos, Pos):
    nearNeighbs = wl.allnearneighbors(Pos, BoxDims, lowCut, highCut).astype(bool)
  else:
    nearNeighbs = wl.nearneighbors(subPos, Pos, BoxDims, lowCut, highCut).astype(bool)

  #Loop over each position in subPos, finding angle made with all neighbor pairs
  for (i, apos) in enumerate(subPos):
    #Make sure have nearest neighbors...
    if len(Pos[nearNeighbs[i]]) > 0:
      #below returns symmetric, square array (zero diagonal)
      tempAng = wl.tetracosang(apos, Pos[nearNeighbs[i]], BoxDims) 
      #Only want half of array, flattened
      angVals = np.hstack((angVals, tempAng[np.triu_indices(len(tempAng),k=1)].tolist()))
      numAngs[i] = tempAng.shape[0]

  return angVals, numAngs
  
  
def tetrahedralMetrics(angVals, nBins=500, binRange=[0.0, 180.0]):
  """Inputs:
     angVals - all angle values sampled
     nBins - number histogram bins to use
     binRange - histogram bin range to apply
     Outputs:
     angDist - distribution of angle
     bins - bins used in histogramming
     fracTet - fraction of distribution that is tetrahedral (integrate cosDist from -0.75 to 0.25 - see Chaimovich, 2014)
     avgCos - average Cos(angle) within tetrahedral peak
     stdCos - second moment of Cos(angle) within tetrahedral peak
  """
  
  #Histogram the data - note that density set so just returns number of counts, not normalized
  angDist, bins = np.histogram(angVals, bins=nBins, range=binRange, density=False)

  #Take index before since want histogram bin containing this value
  startTet = np.argmax(bins>np.arccos(0.25)*180.0/np.pi) - 1
  endTet = np.argmax(bins>np.arccos(-0.75)*180.0/np.pi) - 1 

  fracTet = np.sum(angDist[startTet:endTet]) / np.sum(angDist)

  #Take average and second moment within peak
  avgCos = 0.0
  stdCos = 0.0
  angCount = 0

  for ang in angVals:
    if (ang >= np.arccos(0.25)*180.0/np.pi) and (ang <= np.arccos(-0.75)*180.0/np.pi):
      avgCos = avgCos + np.cos(ang*np.pi/180.0)
      stdCos = stdCos + np.cos(ang*np.pi/180.0)**2
      angCount += 1

  avgCos = avgCos / angCount
  stdCos = stdCos / angCount

  return angDist, bins, fracTet, avgCos, stdCos


def getOrderParamq(subPos, Pos, BoxDims, lowCut=0.0, highCut=8.0):
  """Finds angles for 4 nearest neighbors of each water and returns for all waters the 
tetrahedral order parameter, q, used by Errington and Debenedetti (2001).
     Inputs: 
     subPos - positions of set of atoms to measure tetrahedrality of (may be different, subset, or same as Pos)
     Pos - positions of ALL atoms that can make tetrahedral configurations (needed if subPos not same as Pos)
     BoxDims - current box dimensions to account for periodicity
     lowCut - lower cutoff for nearest-neighbor shell (default 0.0)
     highCut - higher cutoff for nearest-neighbor shell used to find 4 nearest neighbors
     Outputs:
     qVals - returns an order parameter value for each water
     distNeighbs - returns distances from central oxygen to 4 nearest neighbors
  """

  #Set-up array to hold results
  qVals = np.zeros(len(subPos))
  distNeighbs = np.zeros((len(subPos), 4))

  #Find nearest neighbors for ALL atoms in subPos
  #But make sure using efficient algorithm...
  #If subPos is same as Pos, use allnearneighbors instead
  if np.array_equal(subPos, Pos):
    nearNeighbs = wl.allnearneighbors(Pos, BoxDims, lowCut, highCut).astype(bool)
  else:
    nearNeighbs = wl.nearneighbors(subPos, Pos, BoxDims, lowCut, highCut).astype(bool)

  #Loop over each position in subPos, finding angle made with the closest 4 neighbors, then q
  for (i, apos) in enumerate(subPos):
    #Make sure have nearest neighbors...
    if np.sum(nearNeighbs[i]) > 0:
      thisPos = wl.reimage(Pos[nearNeighbs[i]], apos, BoxDims)
      thisDists = np.linalg.norm(thisPos - apos, axis=1)
      sortInds = np.argsort(thisDists)
      newPos = thisPos[sortInds][:4]
      distNeighbs[i,:] = thisDists[sortInds][:4]
      #below returns symmetric, square array (zero diagonal)
      tempAng = wl.tetracosang(apos, newPos, BoxDims)
      #Only want half of array, flattened
      angVals = tempAng[np.triu_indices(len(tempAng),k=1)]
      #Now compute q for this set of angles
      qVals[i] = 1.0 - (3.0/8.0)*np.sum((np.cos(angVals*np.pi/180.0) + (1.0/3.0))**2)

  #Return all of the order parameter values
  return qVals, distNeighbs


def findSineCoeffs(allangs, Norder=180, doNormalize=False):
  """Given an array of angles, computes the sine coefficients to the given order.
Note that to get right coefficients, will need to divide by total number of angles.
This is not done by default, assuming that angles provided are for each frame.
     Inputs:
             allangs - array or list of angles
             Norder - (default 180) number of terms in sine series to use (excludes k=0)
             doNormalize - (default False) if true, divides by number of samples to correctly normalize
     Outputs:
             coeffs - Norder x 2 array; 1st column is k, second column is coefficient
                      Comes from fact that period is zero to Pi, so only keep sin(k*angle) in series
  """
  #Check if angles in radians - if any values are greater than Pi, assume in degrees
  if np.max(allangs) > np.pi:
    allangs = allangs * np.pi / 180.0
  coeffs = np.zeros((Norder,2))
  for k in range(Norder):
    coeffs[k,0] = k+1
    coeffs[k,1] = np.sqrt(2.0/np.pi)*np.sum(np.sin((k+1)*allangs))
  if doNormalize:
    coeffs = coeffs / len(allangs)
  return coeffs


def distFromCoeffs(coeffs, angvals=None, Norder=60):
  """Given an array of coefficients for a sine series, compute the distribution.
     Inputs:
             coeffs - coefficients for each term in a sine series
                      assuming that for sin(k*angle) form, this array is sorted from small to large k
             angvals - (default 0.0 to 180.0 by 0.01) angle values in degrees at which distribution 
                       should be evaluated - normalization will be done for PDF along degrees
             Norder - (default 60) number of terms in the series to use (i.e. number of coeffs)
     Outputs:
             adist - returns a normalized distribution
  """
  if angvals is None:
    angvals = np.arange(0.0, 180.0, 0.01)
  #Also define in radians
  radvals = angvals * np.pi / 180.0
  adist = np.zeros(len(angvals))
  normfac = 0.0
  for k in range(Norder):
    adist += coeffs[k]*np.sin((k+1)*radvals)
    if (k+1)%2 != 0:
      normfac += coeffs[k]*2.0/(k+1)
  adist = adist / (normfac*(angvals[1]-angvals[0]))
  return adist


def fitDist(refDists, Dist, bruteNs=200):
  """Given a set of reference distributions, as a numpy array with each distribution as a row,
fits the current distribution using a linear combination of the reference distributions.
  Inputs:
  refDists - array with each reference distribution as a row
  Dist - (3-body angle) distribution to fit as linear combination of references with
         the fitting parameters (linear coefficients) summing to one
  bruteNs - number of discrete bins to use along each parameter when searching for brute minimum
  Outputs:
  fitParams - fit parameters resulting from fitting 
  resSq - sum of squared residuals for fit
  resSigned - signed residuals at each point of fit
  """

  #Define tolerance
  tolf = 1.0e-12
  tolx = 1.0e-12  

  #Initialize parameters to seek for - start in 4 ways and take minimum of these
  initParams = np.eye(refDists.shape[0])
  initParams = np.vstack((initParams, np.ones(refDists.shape[0]) * (1.0/refDists.shape[0])))

  #Define an objective function to be minimized
  def funcMin(vals, *withcon):
    #Give it parameter values, returns squared residuals
    func = np.sum((np.dot(vals, refDists) - Dist)**2)
    if withcon:
      func = func + (np.sum(vals) - 1.0)**2
    return func

  def jacFunc(vals):
    #Returns the Jacobian of the function to minimize
    func = np.dot(refDists, 2.0*(np.dot(vals, refDists) - Dist))
    return func

  def funcSquares(vals):
    #Gives vector of squared residuals to see where best/worst parts of fit are
    func = (np.dot(vals, refDists) - Dist)**2
    return func

  #Define constraints... for now say that all parms must sum to one
  cons = ({'type' : 'eq',
           'fun' : lambda x: np.sum(x) - 1.0,
           'jac' : lambda x: np.ones(len(x))})

  #And define bounds to keep all params between 0 and 1
  bnds = [(0.0,1.0)]*refDists.shape[0]

  #For each set of starting conditions, do minimization, then pick global min
  globMinInfo = None

  #And will store squared residuals at found mins as go
  #Checks if one part of curve fits better than another
  resSq = np.zeros((refDists.shape[1], initParams.shape[0]))

  for (i, params) in enumerate(initParams):

    #If only one distribution given, don't use constraint
    if refDists.shape[0] == 1:
      mininfo = optimize.minimize(funcMin, params, jac=jacFunc, method='SLSQP',
                                                bounds=bnds, options={'ftol':tolf})
    else:
      mininfo = optimize.minimize(funcMin, params, jac=jacFunc, method='SLSQP',
                                constraints=cons, bounds=bnds, options={'ftol':tolf})
   
    #print "Minimum sum of squares: %e  at values "%mininfo.fun+str(mininfo.x)

    if globMinInfo != None:
      if mininfo.fun < globMinInfo.fun:
        globMinInfo = mininfo
    else:
      globMinInfo = mininfo

    resSq[:,i] = funcSquares(mininfo.x)

  #Compare to global min with brute force
  if refDists.shape[0] == 1:
    (bruteMinInfo) = optimize.brute(funcMin, tuple(bnds), Ns=bruteNs, finish=None, full_output=True, disp=False)
  else:
    (bruteMinInfo) = optimize.brute(funcMin, tuple(bnds), args=(1,), Ns=bruteNs, finish=None, full_output=True, disp=False)

  fitParams = bruteMinInfo[0]

  #print "Brute force finds minima at "+str(fitParams)

  #Also compute regular residuals, not squared
  resSigned = np.dot(fitParams, refDists) - Dist

  #print "Best fit found at:"
  #print [float(q) for q in fitParams]
  #print "And with parameters summing to %f" % np.sum(fitParams)
  return fitParams, resSq, resSigned 


def waterOrientationBinZ(Opos, Hpos, boxDim, refVec=[0.0, 0.0, 1.0], refBins=None, angBins=None):
  """Determines the angle between a reference vector and the dipoles and plane-normal vector
of all water molecule positions provided.
     Inputs:
     Opos - all water oxygen positions
     Hpos - all water hydrogen positions
     boxDim - box dimensions for imaging
     refVec - the reference vector for water orientation, default
              is the z-direction [1, 0, 0]
     refBins - bins along the direction of refVec that the waters 
               should be placed into (default is min and max along refVec)
     angBins - bins for calculated angles, default 500 bins from 0 to 180
     Outputs:
     plane2Dhist - 2D histogram with angle bins varying over rows and 
                   refVec bins varying over rows (for the water plane vector angles)
     dip2Dhist - 2D histogram as above, but for dipole vector angles
  """

  #Get positions of oxygen atoms along refVec, then create 
  #this array with each entry repeated
  refVec = refVec / np.linalg.norm(refVec)
  zOpos = np.dot(Opos, refVec)
  zOposforH = np.array([[z,z] for z in zOpos]).flatten()

  #Compute all of the angles with respect to the reference
  #Note that the dipole vector of each water molecule will be taken as
  #the sum of the OH bond vectors
  angDip, angPlane = wl.watorient(Opos, Hpos, refVec, boxDim)

  #Set refBins if not set
  if refBins is None:
    refBins = np.arange(np.min(zOpos), np.max(zOpos), 0.2)

  #Same for angBins
  if angBins is None:
    angBins = np.arange(0.0, 180.001, 180.0/500.0)

  #And do 2D histogramming
  plane2Dhist, angEdges, refEdges = np.histogram2d(angPlane, zOposforH, bins=[angBins, refBins], normed=False)
  dip2Dhist, angEdges, refEdges = np.histogram2d(angDip, zOpos, bins=[angBins, refBins], normed=False)

  return plane2Dhist, dip2Dhist


def waterOrientation(Opos, Hpos, boxDim, refVec=[0.0, 0.0, 1.0]):
  """This is a wrapper for the waterlib function watorient.
     Inputs:
     Opos - all water oxygen positions
     Hpos - all water hydrogen positions
     boxDim - box dimensions for imaging
     refVec - the reference vector for water orientation, default
              is the z-direction [1, 0, 0]
     Outputs:
     dipAngs - all angles of dipole vectors with reference vector for all waters
     planeAngs - all angles of plane-normal vector to reference vector for all waters
  """

  #Call watorient to get all angles
  dipAngs, planeAngs = wl.watorient(Opos, Hpos, refVec, boxDim)

  return dipAngs, planeAngs


def binnedVolumePofN(Opos, volBins, numBins, binMask=None):
  """Inputs:
     Opos - array of oxygen 3D coordinates
     volBins - volume (x,y,z coordinate) bin edges tiling the space to place waters into
               Form should be tuple of x, y, and z bin edge arrays
               Bins should be uniform (makes no sense to structure analysis this way otherwise)
     numBins - bin edges for histogramming number of waters in each volume of volBins
     binMask - boolean array of same dimension as number of bins in x, y, z
               Use to exclude some bins by changing certain coordinates to False
     Outputs:
     numWatHist - counts for number of waters in sub-volume of size edgeLxedgeLxedgeL
  """

  #Create mask if necessary
  if binMask is None:
    binMask = np.ones((len(volBins[0])-1, len(volBins[1])-1, len(volBins[2])-1), dtype=bool)
  else:
    if binMask.shape == (len(volBins[0])-1, len(volBins[1])-1, len(volBins[2])-1):
      binMask = binMask
    else:
      print "Dimensions of mask for spatial bins does not match dimensions of spatial bins. Quitting."
      sys.exit(2)

  #Want to use sphere rather than cube for statistics
  #So first find which bin each oxygen belongs to, then find distance
  #to center of bin and see if should exclude or not
  #Written in Fortran for speed
  hist = wl.binongrid(Opos, volBins[0], volBins[1], volBins[2])

  #Use numpy histogramming to count how many oxygens in each cube volume (doesn't use interior spheres)
  #hist, edges = np.histogramdd(Opos, bins=volBins, normed=False)

  #Now histogram number of waters in each subvolume, which will be P(N)
  numWatHist, watedges = np.histogram(hist[binMask].flatten(), bins=numBins, normed=False)

  return numWatHist

#Should also define some function "pointsInVol" that creates binMask based on given set of points or some geometry that should not be included when finding waters (i.e. like a hard sphere or protein)


def HBondsGeneral(accPos, donPos, donHPos, boxL, accInds, donInds, donHInds, distCut=3.5, angCut=150.0):
  """Wraps generalHbonds in the waterlib library to define H-bonds, and also returns their locations.
     Inputs:
     accPos - 3-dimensional vectors of acceptor heavy-atom positions
     donPos - 3D vectors of donor heavy-atom positions (if have multiple hydrogens, must list multiple times)
     donHPos - 3D vector of donor hydrogen positions (should be same length as donPos, which may have duplicates)
     accInds - indices of acceptor atoms
     donInds - indices of donor heavy-atoms
     donHInds - indices of donor hydrogen atoms
     boxL - box dimensions
     distCut - (default 3.5) heavy-atom to heavy-atom distance below which an H-bond may be defined 
     angCut - (default 150.0) O-H---O angle cut-off, in degrees, above which an H-bond may be defined
     Outputs:
     NumHB - number of hydrogen bonds for acceptor/donor set provided
     HBlist - NumHB x 2 array with acceptor index in the 1st column and donor index in the 2nd
     HBloc - NumHB x 3 array of h-bond locations, which is halfway between the acceptor and donor H
  """

  #First get H-bond matrix and locations
  HBboolMat = wl.generalhbonds(accPos, donPos, donHPos, boxL, distCut, angCut)
  HBboolMat = np.array(HBboolMat, dtype=bool)

  #Now parse through matrix, counting H-bonds and creating list of index pairs
  NumHB = np.sum(HBboolMat)
  HBlist = (-1)*np.ones((NumHB, 2))
  HBloc = np.zeros((NumHB, 3))
  HBlistCount = 0
  for i, abool in enumerate(HBboolMat):
    theseDonors = donInds[abool]
    if len(theseDonors) > 0:
      theseDonHPos = donHPos[abool]
      #Image donor H location around acceptor
      theseDonHPos = wl.reimage(theseDonHPos, accPos[i], boxL)
      for j, aDon in enumerate(theseDonors):
        HBlist[HBlistCount,:] = [accInds[i], aDon]
        HBloc[HBlistCount] = 0.5*(theseDonHPos[j] + accPos[i])
        HBlistCount += 1

  return NumHB, HBlist, HBloc


def computeSphericalFourierCoeffs(subPos, Pos, BoxDims, lowCut=0.0, highCut=3.413, minDegree=0, maxDegree=12):
  """Computes the vectors of Fourier coefficients for each degree of a spherical harmonic expansion 
     as described by Keys, Iacovella, and Glotzer, 2011. subPos is treated as the central atoms, 
     while Pos should include the atoms that may potentially be neighbors.
     Inputs:
     subPos - positions of atoms to treat as the central atoms
     Pos - positions of all other atoms, which will be considered for neighbor-searching; can be same as subPos
     BoxDims - box dimensions so that minimum images may be used
     lowCut - (default 0.0) the lower cutoff for the radial shell
     highCut - (default 3.413) the upper cutoff for the radial shell
     minDegree - (default 0) the minimum spherical harmonic degree (l)
     maxDegree - (default 12) the maximum spherical harmonic degree (l)
     Outputs:
     coeffVecs - a len(subPos) x (1 + maxDegree - minDegree) x (2*maxDegree + 1) matrix
                 For each central atom in subPos, a matrix of the complex-valued vectors (as rows)
                 is provided. This still allows magnitudes to be easily evaluated, since real and
                 imaginary parts of zero will contribute nothing to the magnitude
     numNeighbs - number of neighbors for each water molecule (necessary to compute global order parameters
                  or coefficients by multiplying by this and the dividing by the total of the waters
                  to "average" over)
  """

  #Set up the output matrix now since know size
  coeffVecs = np.zeros((len(subPos), 1+maxDegree-minDegree, 2*maxDegree+1), dtype=complex)
  
  #And array to return number of neighbors for each water
  numNeighbs = np.zeros(len(subPos), dtype='float16')

  #Would be nice to combine neighbor searching with 3-body angle computation or H-bonding code
  #But then harder to efficiently implement different cutoffs...
  #So that might be too ambitious
  #Find neighbors within cutoff for ALL atoms in subPos
  #But make sure using efficient algorithm...
  #If subPos is same as Pos, use allnearneighbors instead
  if np.array_equal(subPos, Pos):
    nearNeighbs = wl.allnearneighbors(Pos, BoxDims, lowCut, highCut).astype(bool)
  else:
    nearNeighbs = wl.nearneighbors(subPos, Pos, BoxDims, lowCut, highCut).astype(bool)

  #Loop over each position in subPos and find neighbor positions in spherical coordinates
  for (i, apos) in enumerate(subPos):
    #Make sure have nearest neighbors...
    if len(Pos[nearNeighbs[i]]) > 0:
      tempPos = wl.reimage(Pos[nearNeighbs[i]], apos, BoxDims) - apos
      numNeighbs[i] = len(tempPos)
      #Compute radial distances... unfortunate that have to do this again, but maybe improve later
      rdists = np.linalg.norm(tempPos, axis=1)
      #And get polar and azimuthal angles
      polarang = np.arccos(tempPos[:,2]/rdists)
      azimang = np.arctan2(tempPos[:,1], tempPos[:,0]) #Using special arctan2 function to get quadrant right
      #Now compute Fourier coefficient vectors (i.e. have complex-valued component of coefficient vector
      #associated with each m value, where m = -l, -l+1, ... , l)
      #Loop over the desired number of coefficients to compute
      for l in range(minDegree, maxDegree + 1):
        thisvec = np.zeros(2*l + 1, dtype=complex)
        #Also note that have one vector for each neighbor, so must loop over neighbors
        for j in range(len(tempPos)):
          thisvec += sph_harm(np.arange(-l, l+1), l, azimang[j], polarang[j])
        thisvec /= len(tempPos)
        #And compute the magnitude of this vector of complex numbers
        coeffVecs[i,l-minDegree,:(2*l+1)] = thisvec

  return coeffVecs, numNeighbs


def get1BodyDOFs(coordO, coordH1, coordH2):
  """Given O, H, and H 3D coordinates, identifies the 6 degrees of freedom for a single water
Note that this is assuming an inhomogeneous system
Vector returned is oxygen x, y, z, followed by the spherical coordinate angles for the 
dipole vector relative to the oxygen, and the angle of rotation around the dipole vector
COORDINATES SHOULD ALREADY BE IMAGED.
  """
  dofVec = np.zeros(6)
  dofVec[:3] = coordO[:]

  rOD = 0.5*(coordH1 + coordH2) - coordO
  rOD /= np.linalg.norm(rOD) #Could hard-code the rOD length for speed... maybe later

  rH1H2 = coordH2 - coordH1
  rH1H2 /= np.linalg.norm(rH1H2) #And could also hard-code this, too...

  #rOH1 = coordH1 - coordO
  #rOH1 /= np.linalg.norm(rOH1)

  #rOH2 = coordH2 - coordO
  #rOH2 /= np.linalg.norm(rOH2)

  unitX = np.array([0.0, 0.0, 1.0]) #Arbitrarily pick x axis to define reference plane for rotation about dipole

  #cross1 = np.cross(rOH1, rOH2)
  #cross1 /= np.linalg.norm(cross1)

  crossX = np.cross(rOD, unitX)
  crossX /= np.linalg.norm(crossX)

  dofVec[3] = np.arctan2(rOD[1], rOD[0]) #Making sure to use arctan2 to cover range [-pi, pi]
  dofVec[4] = np.arccos(rOD[2]) #Taking last element is same as dotting with unit Z vector
  dofVec[5] = np.arccos(np.dot(rH1H2, crossX))
  #dofVec[5] = np.arccos(np.dot(cross1, crossX))

  return dofVec


def get2BodyDOFs(coordO1, coordH11, coordH12, coordO2, coordH21, coordH22):
  """Given 3D coordinates for all atoms in two water molecules, computes specifically 2-body degrees of freedom
Note that returns only 6 degrees of freedom, so excludes the DOFs for the first water
ONLY gives those relevant to relative distance and orientation of two waters
Order in returned vector is rO1O2, theta1, theta2, phi, chi1, chi2 (see Lazaridis and Karplus for definitions)
COORDINATES SHOULD ALREADY BE IMAGED
  """
  dofVec = np.zeros(6)

  rO1O2 = coordO2 - coordO1
  dofVec[0] = np.linalg.norm(rO1O2)
  rO1O2 /= dofVec[0]
  rO2O1 = -rO1O2

  rO1D1 = 0.5*(coordH11 + coordH12) - coordO1
  rO1D1 /= np.linalg.norm(rO1D1) #Could hard-code to speed up... may do later

  rO2D2 = 0.5*(coordH21 + coordH22) - coordO2
  rO2D2 /= np.linalg.norm(rO2D2)

  #Need to figure out which H is closer to other oxygen to define rH11H12 according to Lazaridis and Karplus, 1996
  if np.linalg.norm(coordH11 - coordO2) <= np.linalg.norm(coordH12 - coordO2):
    rH11H12 = coordH12 - coordH11
  else:
    rH11H12 = coordH11 - coordH12
  rH11H12 /= np.linalg.norm(rH11H12) #Again, could hard code if wanted

  if np.linalg.norm(coordH21 - coordO1) <= np.linalg.norm(coordH22 - coordO1):
    rH21H22 = coordH22 - coordH21
  else:
    rH21H22 = coordH21 - coordH22
  rH21H22 /= np.linalg.norm(rH21H22)

  cross1 = np.cross(rO1O2, rO1D1)
  cross1 /= np.linalg.norm(cross1)

  cross2 = np.cross(rO2D2, rO2O1)
  cross2 /= np.linalg.norm(cross2)

  dofVec[1] = np.arccos(np.dot(rO1D1, rO1O2))
  dofVec[2] = np.arccos(np.dot(rO2D2, rO2O1))
  dofVec[3] = np.arccos(np.dot(cross1, cross2))
  dofVec[4] = np.arccos(np.dot(rH11H12, cross1))
  dofVec[5] = np.arccos(np.dot(rH21H22, cross2))

  return dofVec


def get3BodyDOFs(coordO1, coordH11, coordH12, coordO2, coordH21, coordH22, coordO3, coordH31, coordH32):
  """Like above function, but gives 6 DOFs pertaining to just the 3-body degrees of freedom
Order in returned vector  is rO1O3 (distance), theta3b (three-body angle),
omega (rotation of 3rd water around O1-O3 vector), then theta3, phi3, and chi3
(last three defined as for the second water in the 2-body DOFs, but for just the third water)
COORDINATES SHOULD ALREADY BE IMAGED
  """
  dofVec = np.zeros(6)

  rO1O2 = coordO2 - coordO1
  rO1O2 /= np.linalg.norm(rO1O2)
  rO2O1 = -rO1O2

  rO1O3 = coordO3 - coordO1
  dofVec[0] = np.linalg.norm(rO1O3)
  rO1O3 /= dofVec[0]
  rO3O1 = -rO1O3

  rO1D1 = 0.5*(coordH11 + coordH12) - coordO1
  rO1D1 /= np.linalg.norm(rO1D1)

  rO3D3 = 0.5*(coordH31 + coordH32) - coordO3
  rO3D3 /= np.linalg.norm(rO3D3)

  if np.linalg.norm(coordH31 - coordO1) <= np.linalg.norm(coordH32 - coordO1):
    rH31H32 = coordH32 - coordH31
  else:
    rH31H32 = coordH31 - coordH32
  rH31H32 /= np.linalg.norm(rH31H32)

  cross12 = np.cross(rO1O2, rO1D1)
  cross12 /= np.linalg.norm(cross12)

  cross13 = np.cross(rO1O3, rO1D1)
  cross13 /= np.linalg.norm(cross13)

  cross31 = np.cross(rO3D3, rO3O1)
  cross31 /= np.linalg.norm(cross31)

  rperp = rO1O3 - np.dot(rO1O2, rO1O3)*rO1O2
  rperp /= np.linalg.norm(rperp)

  dofVec[1] = np.arccos(np.dot(rO1O2, rO1O3))
  dofVec[2] = np.arccos(np.dot(rperp, cross12))
  dofVec[3] = np.arccos(np.dot(rO3D3, rO3O1))
  dofVec[4] = np.arccos(np.dot(cross13, cross31))
  dofVec[5] = np.arccos(np.dot(rH31H32, cross31))

  return dofVec


def distanceMetric1B(vec1, vec2, Rsq=(0.09572**2), sintw=(np.sin(104.52*np.pi/180.0)**2)):
  """Computes distance metric appropriate to 1-body DOFs.
     A direct Euclidean distance is not appropriate since using curvilinear coordinates, 
     so this defines a distance utilizing local curvature that is exact for very small
     differences. It comes from Taylor-expanding the formula for Euclidean distance in 
     spherical coordinates with respect to both angles to second order.
  """
  diffs = (vec2 - vec1)**2
  dist = np.sqrt(diffs[0] + diffs[1] + diffs[2] + Rsq*diffs[3]
                 + Rsq*np.sin(vec2[3])*np.sin(vec1[3])*diffs[4]
                 + Rsq*sintw*diffs[5])
  return dist


def distanceMetric2B(vec1, vec2, Rsq=(0.09572**2), sintw=(np.sin(104.52*np.pi/180.0)**2)):
  """Computes distance metric appropriate to 2-body DOFs.
     A direct Euclidean distance is not appropriate since using curvilinear coordinates, 
     so this defines a distance utilizing local curvature that is exact for very small
     differences. It comes from Taylor-expanding the formula for Euclidean distance in 
     spherical coordinates with respect to both angles to second order.
     Note that this includes 1-body degrees of freedom, so expects 12-dimensional vectors.
  """
  diffs = (vec2 - vec1)**2
  dist = np.sqrt(diffs[0] + diffs[1] + diffs[2] + Rsq*diffs[3]
                 + Rsq*np.sin(vec2[3])*np.sin(vec1[3])*diffs[4]
                 + Rsq*sintw*diffs[5]
                 + diffs[6] + Rsq*diffs[7] + Rsq*diffs[8]
                 + Rsq*np.sin(vec2[8])*np.sin(vec1[8])*diffs[9]
                 + Rsq*sintw*diffs[10] + Rsq*sintw*diffs[11])
  return dist


def distanceMetric3B(vec1, vec2, Rsq=(0.09572**2), sintw=(np.sin(104.52*np.pi/180.0)**2)):
  """Computes distance metric appropriate to 3-body DOFs.
     A direct Euclidean distance is not appropriate since using curvilinear coordinates, 
     so this defines a distance utilizing local curvature that is exact for very small
     differences. It comes from Taylor-expanding the formula for Euclidean distance in 
     spherical coordinates with respect to both angles to second order.
     Note that this includes 1- and 2-body degrees of freedom, so expects 18-dimensional vectors.
  """
  diffs = (vec2 - vec1)**2
  dist = np.sqrt(diffs[0] + diffs[1] + diffs[2] + Rsq*diffs[3]
                 + Rsq*np.sin(vec2[3])*np.sin(vec1[3])*diffs[4]
                 + Rsq*sintw*diffs[5]
                 + diffs[6] + Rsq*diffs[7] + Rsq*diffs[8]
                 + Rsq*np.sin(vec2[8])*np.sin(vec1[8])*diffs[9]
                 + Rsq*sintw*diffs[10] + Rsq*sintw*diffs[11]
                 + diffs[12] + vec2[12]*vec1[12]*diffs[13]
                 + vec2[12]*vec1[12]*np.sin(vec2[13])*np.sin(vec1[13])*diffs[14]
                 + Rsq*diffs[15]
                 + Rsq*np.sin(vec2[15])*np.sin(vec1[15])*diffs[16]
                 + Rsq*sintw*diffs[17])
  return dist

