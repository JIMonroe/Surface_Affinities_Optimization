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

#If want, will do a single plot of instantaneous interface for the purposes of debugging, otherwise, comment out
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#fig = plt.figure(figsize=(10, 10))
#ax = fig.add_subplot(111, projection='3d')


Usage="""surf_water_structure topFile simDirs 
  Computes various properties of water at an interface. This is intended to be
  used with expanded-ensemble simulation inserting a molecule on the TOP side
  (larger z-coordinate) of an interface, with the bottom side being in contact
  with just water. This way, reweighting of configurations is not needed, as
  we will assume that the bottom interface is not affected by the solute state
  or the way the solute is restrained.
  Inputs:
    topFile - topology file associated with trajectories
    simDirs - any number of directories containing single trajectory files 
              related to the interface of interest
  Outputs (all files, no returns):
    interface_Z.nc - netCDF4 file describing the height of the interface at each
                     of a set of x-y grid points
    z-densities.txt - densities of surface heavy atoms and water oxygens (for
                      both interface definitions) as a function of distance
                      from the interface
    probe_hists_mean.txt - histograms of numbers of waters in probes at various
                           distances from the mean interface
    probe_hists_instant.txt - same as above, but for instantaneous interface
    ang3b_hists_mean.txt - histograms of 3-body angles for various distance
                           slices from the mean interface
    ang3b_hists_instant.txt - same as above, but for instantaneous interface
    pair_hists_mean.txt - histograms of water O-O pair distances for various
                          distance slices from the mean interface
    pair_hists_instant.txt - same as above, but for instantaneous interface 
"""

def main(args):

  print time.ctime(time.time())

  #To define some things, read in the first simulation
  #Will then assume all other simulations are set up in an identical way
  #(i.e. same box size, same number of frames)
  topFile = args[0]
  simDirs = args[1:]
 
  #First load in the topology and trajectory
  top = pmd.load_file(topFile)
  top.rb_torsions = pmd.TrackedList([]) #This is just for SAM systems so that it doesn't break pytraj
  top = pt.load_parmed(top, traj=False)
  traj = pt.iterload(simDirs[0]+'/prod.nc', top)
  
  boxDims = np.array(traj[0].box.values[:3]) 
  
  #Before starting, create bins in the x, y, and z-directions 
  #This will be used to define instantaneous interfaces and record some spatially varying interfacial properties
  gridSize = 1.0 #Angstroms
  xGrid = np.arange(-boxDims[0]/2.0, boxDims[0]/2.0 + gridSize, gridSize) #Bins may overlap, but that's ok
  yGrid = np.arange(-boxDims[1]/2.0, boxDims[1]/2.0 + gridSize, gridSize)
  zGrid = np.arange((-boxDims[2] + boxDims[2]%gridSize)/2.0, 
                    (boxDims[2] - boxDims[2]%gridSize)/2.0 + 0.001, gridSize) #Symmetrizing grid in z direction

  print "Using following coarse grids in x-, y-, and z-dimensions:"
  print xGrid
  print yGrid
  print zGrid

  #Define grid-point centers
  xGridLocs = 0.5*(xGrid[:-1] + xGrid[1:])
  yGridLocs = 0.5*(yGrid[:-1] + yGrid[1:])
  zGridLocs = 0.5*(zGrid[:-1] + zGrid[1:])

  #And define grid sizes
  xSize = len(xGridLocs)
  ySize = len(yGridLocs)
  zSize = len(zGridLocs)

  #Now set-up fine grid for actually identifying the interface more precisely
  zGridSize = 0.1 #Set separate grid size in z direction for a fine grid used for finding all density profiles
  zGridFine = np.arange((-boxDims[2] + boxDims[2]%zGridSize)/2.0, 
                       (boxDims[2] - boxDims[2]%zGridSize)/2.0 + 0.001, zGridSize) 

  #Define grid-point centers for fine grid
  zGridLocsFine = 0.5*(zGridFine[:-1] + zGridFine[1:])
  zSizeFine = len(zGridLocsFine)

  #For calculating some properties as a function as distance from the surface, will use specific z slices
  #While all the grid stuff is relative to surface SU atoms, want z slices relative to the interface itself
  sliceGridSize = 0.5
  zSlice = np.arange(-6.0, 12.000001, sliceGridSize)
  zSliceLocs = 0.5 * (zSlice[:-1] + zSlice[1:])
  zSliceSize = len(zSliceLocs)
 
  #Set the default bulk density
  bulkDens = 0.0332 #Roughly right for TIP4P-EW at 298.15 K and 1 bar in inverse Angstroms cubed

  #Define the density fraction for determining the interface location
  fracDens = 0.3 #this is lower than 0.5 which is usually used by Willard and others
                 #however, it puts the interface a little closer to the surface atoms and,
                 #as Willard says in his 2014 paper, any value between 0.3 and 0.7 works
  densCut = fracDens*bulkDens
  print "\nUsing bulk density value of %f (TIP4P-Ew at 1 bar and 298.15 K)." % bulkDens
  print "To define interface, using a bulk density fraction of %f." % fracDens

  #Define the size of the probes used for assessing hydrophobicity
  probeRadius = 3.3 # radius in Angstroms; the DIAMETER of a methane (so assumes all other atoms methane-size)

  #Now define how we compute three-body angles with bins and cut-off
  #Shell cut-off
  shellCut = 3.32 #1st minimum distance for TIP4P-Ew water at 298.15 K and 1 bar
  #Number of angle bins
  nAngBins = 100 #500
  #Define bin centers (should be nBins equally spaced between 0 and 180)
  angBinCents = 0.5 * (np.arange(0.0, 180.001, 180.0/nAngBins)[:-1] + np.arange(0.0, 180.001, 180.0/nAngBins)[1:])

  #And distance bins for local RDF calculation 
  #(really distance histograms from central oxygens - can normalize however we want, really)
  distBinWidth = 0.05
  nDistBins = int(shellCut / distBinWidth)
  distBins = np.arange(0.0, nDistBins*distBinWidth+0.00001, distBinWidth)
  distBinCents = 0.5 * (distBins[:-1] + distBins[1:])

  #And bins for numbers of waters in probes
  probeBins = np.arange(0.0, 21.00001, 1.0)
  nProbeBins = len(probeBins) - 1 #Will use np.histogram, which includes left edge in bin (so if want up to 20, go to 21)

  #Should we do 2D histogram for angles and distances?
  #Or also do 2D histogram for number of waters in probe and three-body angle?
  #Interesting if make probe radius same size as three-body angle cutoff?

  #Need to create a variety of arrays to hold the data we're interested in
  interfaceZmean = np.zeros((xSize, ySize)) #Average mean interface height at each x-y bin - same in all x-y bins
  interfaceZ = np.zeros((xSize, ySize)) #Average instantaneous interface height at each x-y bin
  interfaceZSq = np.zeros((xSize, ySize)) #Squared average interface height (to compute height fluctuations)
                                             #This can be more easily compared to mean-field studies
  watDensFine = np.zeros(zSizeFine) #Water density on finer grid for instantaneous interface definition
  watDensFineMean = np.zeros(zSizeFine) #Same but for mean interface definition
  surfDensFine = np.zeros(zSizeFine) #Density profile for surface heavy atoms (ignores interface definition) 
  probeHists = np.zeros((nProbeBins, zSliceSize)) #Histograms for numbers of waters in probes at each z-slice 
  angHists = np.zeros((nAngBins, zSliceSize)) #Histograms of three-body angles for water oxygens within in each z-slice
  distHists = np.zeros((nDistBins, zSliceSize)) #Histograms of distances to water oxygens from central oxygens
  probeHistsMean = np.zeros((nProbeBins, zSliceSize)) #And same definitions but for mean interfaces
  angHistsMean = np.zeros((nAngBins, zSliceSize)) 
  distHistsMean = np.zeros((nDistBins, zSliceSize)) 

  #Since will use multiple trajectories, also need to count total frames we're averaging over
  totFrames = 0.0

  #Now need to define variables to hold the surface points and normal vectors at each point
  #Note that by surface points, here we just mean the INDICES of the x, y, and z grids
  #To access the actual points, need to reference xGridLocs, etc.
  #In the 'instant' interface definition, the exact locations of surface points changes, but
  #only keep one surface point per x-y bin, so when project onto x-y plane to look at heat
  #plots can use the same grid point locations as for the fixed, 'mean' interface
  #With the 'mean' definition, this will be set below and unchanged when computing metrics
  surfacePoints = np.zeros((xSize*ySize, 3), dtype=int)
  surfaceNorms = np.zeros((xSize*ySize, 3)) #Should keep as normalized vectors throughout
  surfacePointsMean = np.zeros((xSize*ySize, 3), dtype=int)
  surfaceNormsMean = np.zeros((xSize*ySize, 3)) 

  #But also want an actual list of all possible grid-point locations...
  xmesh, ymesh, zmesh = np.meshgrid(np.arange(xSize), np.arange(ySize), np.arange(zSize))
  gridPoints = np.vstack((xmesh.flatten(), ymesh.flatten(), zmesh.flatten())).T

  #At this point, want to actually loop over the simulation directories, load simulations,
  #and do calculations for each frame

  for adir in simDirs:

    top = pmd.load_file(topFile)
    top.rb_torsions = pmd.TrackedList([]) #This is just for SAM systems so that it doesn't break pytraj
    top = pt.load_parmed(top, traj=False)
    traj = pt.iterload(adir+'/prod.nc', top)
  
    print("\nTopology and trajectory loaded from directory %s" % adir)

    owInds = top.select('@OW')
    surfInds = top.select('(:OTM,CTM,STM,NTM)&!(@H=)')
    suInds = top.select('@SU') 
    print("\n\tFound %i water oxygens" % len(owInds))
    print("\tFound %i non-hydrogen surface atoms" % len(surfInds))
    print("\tFound %i SU surface atoms" % len(suInds))

    nFrames = float(traj.n_frames) #Make it a float to make averaging easier later
    totFrames += nFrames

    tempWatDensFineMean = np.zeros(zSizeFine)

    #Find the mean interface definition to use for this trajectory 
    #Need to loop over the trajectory once and find mean interface location
    for i, frame in enumerate(traj):
      
      boxDims = np.array(frame.box.values[:3])
  
      currCoords = np.array(frame.xyz)
  
      #Need to do wrapping procedure by first putting surface together, then wrapping water around it
      wrapCOM1 = currCoords[suInds[0]]
      currCoords = wl.reimage(currCoords, wrapCOM1, boxDims) - wrapCOM1
      wrapCOM2 = np.average(currCoords[suInds], axis=0)
      currCoords = wl.reimage(currCoords, wrapCOM2, boxDims) - wrapCOM2
 
      #Bin the current z coordinates
      thisSurfDens, tempBins = np.histogram(currCoords[surfInds, 2], bins=zGridFine, normed=False)
      thisWatDens, tempBins = np.histogram(currCoords[owInds, 2], bins=zGridFine, normed=False)
      surfDensFine += thisSurfDens
      tempWatDensFineMean += thisWatDens

    #Record fine density, then normalize to number densities for next calculation
    watDensFineMean += tempWatDensFineMean
    tempWatDensFineMean = tempWatDensFineMean / (nFrames*boxDims[0]*boxDims[1]*zGridSize)

    #Find the average density in a region far from the surface to use as a reference
    #Recall that the wrapping procedure puts the surface center of geometry at the origin
    refOWdens = np.average(0.5*(tempWatDensFineMean[-int(1.0/zGridSize)-1:-1]+tempWatDensFineMean[1:int(1.0/zGridSize)+1])) 
    #Above uses 1.0 A slices at edges of wrapped box
    #It's a useful quantity to report, but better to just use fixed cut-off for all simulations...
    print "\n\tDensity value in center of water phase (far from interface): %f" % refOWdens

    #And find where the water density is half of its "bulk" value
    #Exclude grid points near box edge because fluctuations in box lead to weird densities there
    loInd = np.argmin(abs(tempWatDensFineMean[5:(zSizeFine/2)] - densCut)) 
    
    print "\tOn lower surface, mean interface is at following Z-coordinate:"
    print "\t  Lower: %f" % zGridLocsFine[loInd]
    
    #Now set up arrays of surface points and surface normal vectors - easy for mean interface
    thisxmesh, thisymesh = np.meshgrid(np.arange(xSize, dtype=int), np.arange(ySize, dtype=int))
    surfacePointsMean[:, 0:2] = np.vstack((thisxmesh.flatten(), thisymesh.flatten())).T
    surfacePointsMean[:, 2] = loInd
    surfaceNormsMean[:, 2] = -1.0 #Just working with LOWER surface (solute is on top)

    #Just go ahead and record the average interface height at each x and y bin, since it won't change
    interfaceZmean[:,:] += zGridLocsFine[loInd]

    #Now should be ready to loop over the trajectory using both interface definitions and computing things
    print "\nPre-processing finished, starting main loop over trajectory."
  
    for i, frame in enumerate(traj):
  
      #if i%1000 == 0:
      #  print "On frame %i" % i
    
      boxDims = np.array(frame.box.values[:3])
  
      currCoords = np.array(frame.xyz)
  
      #Need to do a wrapping procedure to more easily find waters within certain layers of surface
      wrapCOM1 = currCoords[suInds[0]]
      currCoords = wl.reimage(currCoords, wrapCOM1, boxDims) - wrapCOM1
      wrapCOM2 = np.average(currCoords[suInds], axis=0)
      currCoords = wl.reimage(currCoords, wrapCOM2, boxDims) - wrapCOM2
    
      OWCoords = currCoords[owInds]
      suCoords = currCoords[suInds]
      surfCoords = currCoords[surfInds]
      surfMidZ = np.average(suCoords[:,2])

      #Get actual locations of surface points (not just indices)
      #Will overwrite with off-lattice x, y, and z positions for instantaneous interface
      #Just use fine z-grid for mean definition
      thisSurf = np.vstack((xGridLocs[surfacePoints[:,0]], 
                            yGridLocs[surfacePoints[:,1]], 
                            zGridLocsFine[surfacePoints[:,2]])).T
      thisSurfMean = np.vstack((xGridLocs[surfacePointsMean[:,0]], 
                                yGridLocs[surfacePointsMean[:,1]], 
                                zGridLocsFine[surfacePointsMean[:,2]])).T

      #Use water coordinates to find instantaneous interface
      #First need to find the density field
      #Note that we use 2.4 as the smoothing length, as in Willard and Chandler, 2010
      thisdensfield, thisdensnorms = wl.willarddensityfield(OWCoords, 
                                                            xGridLocs, yGridLocs, zGridLocs, boxDims, 2.4)

      #Next need to define the interface - using the marchinge cubes algorithm in scikit-image
      verts, faces, normals, values = skmeasure.marching_cubes_lewiner(thisdensfield, densCut, 
                                                                       spacing=(gridSize, gridSize, gridSize))

      #Shift the points returned...
      verts[:,0] += xGrid[0]
      verts[:,1] += yGrid[0]
      verts[:,2] += zGrid[0]

      #And make sure the points are in the box
      #Actually, for the purposes of making sure we have a single interface point for each x-y grid cell,
      #we DO NOT want to wrap. This is because the grid may extend past the box size. For accurately
      #calculating distances this is an issue because points outside the box will never overlap with 
      #wrapped atoms. But, we can't wrap now... have to wait until after we have our interface points.

      #Below plot is for debugging only
      #fig = plt.figure(figsize=(10, 10))
      #ax = fig.add_subplot(111, projection='3d')
      #ax.scatter(verts[:,0], verts[:,1], verts[:,2])
      #ax.set_xlabel('X')
      #ax.set_ylabel('Y')
      #ax.set_zlabel('Z')
      #fig.tight_layout()
      #plt.show()

      #Now need to trim the points so that we only have one point below the surface for
      #each x-y bin. To do this, I'm taking the min z of the upper surface and max z of the lower surface.
      #With this definition, hopefully odd blips in the bulk that satisfy the isosurface definition will be
      #excluded.
      newvertmat = np.ones((xSize, ySize, 3)) #Will flatten, but for now use data structure to help
      newvertmat[:,:,2] = -10000.0

      #Loop over old vertices
      for avert in verts:
        thisXind = np.digitize([avert[0]], xGrid)[0] - 1
        thisYind = np.digitize([avert[1]], yGrid)[0] - 1
        #Check the lower interface
        if ( avert[2] < surfMidZ 
             and avert[2] > newvertmat[thisXind, thisYind, 2]
             and avert[2] > zGrid[0] ):
          newvertmat[thisXind, thisYind, :] = avert

      #Need to make sure that all x-y bins had a vertex in them...
      #If not, use the z-value of one of the 4 adjacent points that isn't also too large
      unfilledbins = np.where(abs(newvertmat[:,:,:]) == 10000.0)
      for l in range(len(unfilledbins[0])):
        ind1 = unfilledbins[0][l] #The x bin
        ind2 = unfilledbins[1][l] #The y bin
        newvertmat[ind1, ind2, 0] = xGridLocs[ind1]
        newvertmat[ind1, ind2, 1] = yGridLocs[ind2]
        #Use modulo operator to do wrapping
        if abs(newvertmat[(ind1-1)%xSize, ind2, 2]) < 1000.0:
          newvertmat[ind1, ind2, 2] = newvertmat[(ind1-1)%xSize, ind2, 2]
        elif abs(newvertmat[(ind1+1)%xSize, ind2, 2]) < 1000.0:
          newvertmat[ind1, ind2, 2] = newvertmat[(ind1+1)%xSize, ind2, 2]
        elif abs(newvertmat[ind1, (ind2-1)%ySize, 2]) < 1000.0:
          newvertmat[ind1, ind2, 2] = newvertmat[ind1, (ind2-1)%ySize, 2]
        elif abs(newvertmat[ind1, (ind2+1)%ySize, 2]) < 1000.0:
          newvertmat[ind1, ind2, 2] = newvertmat[ind1, (ind2+1)%ySize, 2]

      #While the points are in a convenient format, record the interface height at each x-y bin
      interfaceZ += newvertmat[:,:,2]
      interfaceZSq += newvertmat[:,:,2]**2

      #If the above procedure didn't fix the issue, just quit and recommend a finer z-grid size
      unfilledbins = np.where(abs(newvertmat[:,:,2]) == 10000.0)
      if len(unfilledbins[0]) > 0:
        print "Error: after trimming surface points, unable to find surface point for each x-y bin."
        print "Could fix this by not using all bins and keeping track of bin counts, but maybe later."
        print "Try and use a finer bin size in the z-dimension."
        sys.exit(2)

      #Now put together list of surface points (indices on pre-set grid) and exact locations (and unit normals)
      newverts = np.reshape(newvertmat[:,:,:], (xSize*ySize,3))

      #fig = plt.figure(figsize=(10, 10))
      #ax = fig.add_subplot(111, projection='3d')
      #ax.scatter(newverts[:,0], newverts[:,1], newverts[:,2], c='gray')
      #ax.set_xlabel('X')
      #ax.set_ylabel('Y')
      #ax.set_zlabel('Z')
      #fig.tight_layout()
      #plt.show()

      surfacePoints[:,0] = np.digitize(newverts[:,0], xGrid) - 1
      surfacePoints[:,1] = np.digitize(newverts[:,1], yGrid) - 1
      surfacePoints[:,2] = np.digitize(newverts[:,2], zGrid) - 1

      thisSurf = copy.deepcopy(newverts)
      unusedDensVals, surfaceNorms = wl.willarddensitypoints(OWCoords, thisSurf, boxDims, 2.4)
      
      #print surfacePoints
      #print thisSurf
      #print surfaceNorms
      #print np.linalg.norm(surfaceNorms, axis=1)

      #At this point, MUST wrap our interface points into box so that distances from waters to them are accurate
      #We couldn't do this earlier because we need to assign a single interface point to each x-y grid cell
      thisSurf = wl.reimage(thisSurf, np.zeros(3), boxDims)
      thisSurfMean = wl.reimage(thisSurfMean, np.zeros(3), boxDims)

      #fig = plt.figure(figsize=(10, 10))
      #ax = fig.add_subplot(111, projection='3d')
      #ax.scatter(thisSurf[:,0], thisSurf[:,1], thisSurf[:,2], c='orange')
      #ax.set_xlabel('X')
      #ax.set_ylabel('Y')
      #ax.set_zlabel('Z')
      #fig.tight_layout()
      #plt.show()

      #Want to find the density profile 
      #To do this, first need to find which surface point closest to each water
      #Then project that water's distance from the point along the surface normal
      #This gives distances
      #Note that surfaceNorms should be normalized to length 1!
      #Also, don't worry about the random cutoff of 3.0 that I supplied... this is just part of the routine
      #Really only want thisWatDists from this function - other stuff is not as useful for this code
      thisWatClose, thisSurfClose, thisSliceNum, thisWatDists = wl.interfacewater(OWCoords, 
                                                                                  thisSurf, 
                                                                                  surfaceNorms, 
                                                                                  3.0, boxDims)

      #Now add to the instantaneous fine water density profile - mean is already done
      thisWatHist, tempBins = np.histogram(thisWatDists, bins=zGridFine, normed=False)
      watDensFine += thisWatHist

      #Now want to look at properties within z-slices moving normal to both interface definitions

      #For three-body angles and pair distances, digitize waters for each interface definition
      thisSliceInds = np.digitize(thisWatDists, zSlice) - 1
      thisSliceIndsMean = np.digitize(-1.0*(OWCoords[:,2] - zGridLocsFine[surfacePointsMean[0,2]]), zSlice) - 1

      #For probe insertions, placing probes at random x, y, and z locations within slices
      #Make sure to wrap points after randomization to make sure not outside the simulation box
      #Note different grid spacing for z slices, so need to do z separately
      randomGrid = np.zeros(thisSurf.shape)
      randomGrid[:,:2] = thisSurf[:,:2] + (np.random.random_sample((len(thisSurf),2))-0.5)*2.0*gridSize
      randomGrid[:,2] = thisSurf[:,2] + (np.random.random_sample(len(thisSurf))-0.5)*2.0*sliceGridSize
      randomGrid = wl.reimage(randomGrid, np.zeros(3), boxDims)
      randomGridMean = np.zeros(thisSurfMean.shape)
      randomGridMean[:,:2] = thisSurfMean[:,:2] + (np.random.random_sample((len(thisSurfMean),2))-0.5)*2.0*gridSize
      randomGridMean[:,2] = thisSurfMean[:,2] + (np.random.random_sample(len(thisSurfMean))-0.5)*2.0*sliceGridSize
      randomGridMean = wl.reimage(randomGridMean, np.zeros(3), boxDims)

      #And loop over z indices, selecting waters and calculating what we want in those slices
      for j in range(zSliceSize):

        thisSliceCoords = OWCoords[np.where(thisSliceInds == j)[0]]
        thisSliceCoordsMean = OWCoords[np.where(thisSliceIndsMean == j)[0]]

        #Make sure we have waters in slices before attempting anything
        if len(thisSliceCoords) > 0:
        
          #Three-body angles
          thisAngs, thisNumAngs = wp.getCosAngs(thisSliceCoords, OWCoords, boxDims, highCut=shellCut)
          thisAngHist, thisAngBins = np.histogram(thisAngs, bins=nAngBins, range=[0.0, 180.0], density=False)
          angHists[:,j] += thisAngHist

          #Distance histograms with these slice oxygens as central oxygens
          thisDistHist = wl.pairdistancehistogram(thisSliceCoords, OWCoords, distBinWidth, nDistBins, boxDims)
          distHists[:,j] += thisDistHist

        #Now probe occupancies
        #Need to get random z locations within this slice
        thisGrid = randomGrid + surfaceNorms*zSliceLocs[j]
        thisNum = wl.probegrid(np.vstack((OWCoords, surfCoords)), thisGrid, probeRadius, boxDims)
        thisProbeHist, thisProbeBins = np.histogram(thisNum, bins=probeBins, density=False)
        probeHists[:,j] += thisProbeHist

        if len(thisSliceCoordsMean) > 0:

          thisAngsMean, thisNumAngsMean = wp.getCosAngs(thisSliceCoordsMean, OWCoords, boxDims, highCut=shellCut)
          thisAngHistMean, thisAngBinsMean = np.histogram(thisAngsMean, bins=nAngBins, range=[0.0, 180.0], density=False)
          angHistsMean[:,j] += thisAngHistMean

          thisDistHistMean = wl.pairdistancehistogram(thisSliceCoordsMean, OWCoords, distBinWidth, nDistBins, boxDims)
          distHistsMean[:,j] += thisDistHistMean

        thisGridMean = randomGridMean + surfaceNormsMean*zSliceLocs[j]
        thisNumMean = wl.probegrid(np.vstack((OWCoords, surfCoords)), thisGridMean, probeRadius, boxDims)
        thisProbeHistMean, thisProbeBinsMean = np.histogram(thisNumMean, bins=probeBins, density=False)
        probeHistsMean[:,j] += thisProbeHistMean

  #Done looping over trajectories, etc. 
  #Now finish computing some quantities by averaging appropriately
  interfaceZmean /= float(len(simDirs))
  interfaceZ /= totFrames
  interfaceZSq /= totFrames
  watDensFine /= (totFrames*boxDims[0]*boxDims[1]*zGridSize)
  watDensFineMean /= (totFrames*boxDims[0]*boxDims[1]*zGridSize)
  surfDensFine /= (totFrames*boxDims[0]*boxDims[1]*zGridSize)
  #Just let the histograms be histograms... can normalize later if need to (don't think numbers will get too big)
  #probeHists /= totFrames
  #angHists /= totFrames
  #distHists /= totFrames
  #probeHistsMean /= totFrames
  #angHistsMean /= totFrames
  #distHistsMean /= totFrames

  #Now save everything to files
  #For 2D stuff (i.e. interfaces), use netCDF4 format
  outdata = Dataset("interface_Z.nc", "w", format="NETCDF4", zlib=True)
  #outdata.description("Interface heights (in the z-direction from surface atoms) and fluctuations for both mean and instantaneous interfaces.")
  #outdata.history = "Created " + time.ctime(time.time())
  xdim = outdata.createDimension("XGridPoints", xSize)
  ydim = outdata.createDimension("YGridPoints", ySize)
  outXLocs = outdata.createVariable("XGridPoints", "f8", ("XGridPoints",))
  outXLocs.units = "Angstroms"
  outYLocs = outdata.createVariable("YGridPoints", "f8", ("YGridPoints",))
  outYLocs.units = "Angstroms"
  outIntZ = outdata.createVariable("InterfaceHeightInstant", "f8", ("XGridPoints", "YGridPoints",))
  outIntZ.units = "Angstroms relative to SU surface atoms"
  outIntZ[:,:] = interfaceZ
  outIntZMean = outdata.createVariable("InterfaceHeightMean", "f8", ("XGridPoints", "YGridPoints",))
  outIntZMean.units = "Angstroms relative to SU surface atoms"
  outIntZMean[:,:] = interfaceZmean
  outIntZSq = outdata.createVariable("InterfaceHeightInstantSq", "f8", ("XGridPoints", "YGridPoints",))
  outIntZSq.units = "Squared Angstroms relative to SU surface atoms"
  outIntZSq[:,:] = interfaceZSq
  outdata.close()
  #For everything else, just use text files, even though some may be a little big (i.e. angles and pair distances)
  np.savetxt('z-densities.txt', np.vstack((zGridLocsFine, surfDensFine, watDensFineMean, watDensFine)).T,
             header='Distance normal to interface (A)     Surface heavy atom density (1/A^3)      Water oxygen density mean interface (at %f)       Water oxygen density instantaneous interface'%interfaceZmean[0,0])
  np.savetxt('probe_hists_mean.txt', np.hstack((np.array([probeBins[:-1]]).T, probeHistsMean)),
             header='Number waters in probe histograms at following distances from mean interface: %s'%str(zSliceLocs))
  np.savetxt('probe_hists_instant.txt', np.hstack((np.array([probeBins[:-1]]).T, probeHists)),
             header='Number waters in probe histograms at following distances from instantaneous interface: %s'%str(zSliceLocs))
  np.savetxt('ang3b_hists_mean.txt', np.hstack((np.array([angBinCents]).T, angHistsMean)),
             header='3-body angle histograms at following distances from mean interface: %s'%str(zSliceLocs))
  np.savetxt('ang3b_hists_instant.txt', np.hstack((np.array([angBinCents]).T, angHists)),
             header='3-body angle histograms at following distances from instantaneous interface: %s'%str(zSliceLocs))
  np.savetxt('pair_hists_mean.txt', np.hstack((np.array([distBinCents]).T, distHistsMean)),
             header='O-O pair-distance histograms at following distances from mean interface: %s'%str(zSliceLocs))
  np.savetxt('pair_hists_instant.txt', np.hstack((np.array([distBinCents]).T, distHists)),
             header='O-O pair-distance histograms at following distances from instantaneous interface: %s'%str(zSliceLocs))

  print time.ctime(time.time())
 

if __name__ == "__main__":
  main(sys.argv[1:])


