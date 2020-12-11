#A library of genetic algorithm optimization procedures

import sys, os
import shutil
import subprocess
import multiprocessing
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
from genetic_lib import *


#Defines a genetic algorithm optimization of self-assembled monolayer surfaces
#A lot of commenting out has been done - this is mainly to remove the clustering step
#Not removing commented out lines for now just to show how would do clustering if at any point wanted to
def doOptSAM(numSurfOH, genMax, randomGens, metricFunc, metricFuncArgs={}, findMax=False, CH3chainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/ctmSAM', OHchainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/otmSAM'):

  Usage = """
    Uses genetic algorithms to optimize solute solvation free energy near a SAM surface by
    adjusting the locations of hydroxyl-terminated chains at fixed density. This 
    is done by treating each chain lattice location as a 0 (CH3 chain) or 1 (OH
    chain). The resulting boolean array specifies where the OH chains are
    located on the standard lattice structure, with a mappping assumed such that
    point (0,0)->0 in the array, (0,1)->1, (0,2)->2, ... (N,M)->N*M+M. The 
    genetic algorithm uses binary tournament selection in order to pick parents. 
    Generation and mutation steps follow. 

    One must select the way in which fitness metrics are calculated by providing
    a function object to doOpt that performs this calculation. This allows for 
    simple switching between molecular dynamics and machine-learned functions
    to evaluate fitness.

    numSurfOH - total number of hydroxyls (per side) for desired density
    genMax - the number of generations that will be produced
    randomGens - the number of generations before optimization starts (random generations)
    metricFunc - A function object that will be used to obtain the fitness metric.
                 This MUST take a list of SAM structure objects as input.
                 It MUST output a list of fitness metrics of the same as the input.
    metricFuncArgs - (default empty dictionary) A dictionary of keyword arguments defining other 
                     parameters for the metric function object to use. Using keyword arguments 
                     to force the user to be very explicit and automatically check to make sure 
                     variables match up.
    findMax - (default False) whether or not to optimizes to maximum (True) versus minimum (False)
    CH3chainFilePrefix - (optional) file prefix specifying the location before .top
                         and .gro suffixes for the CH3-terminated single chain
    OHchainFilePrefix - (optional) file prefix for the OH-terminated chain
  """

  #Set up directory structure
  try:
    os.mkdir('structures') #Holds all surface structures tested
  except OSError:
    pass

  chainsX = 6
  chainsY = 8 #Number of chains along x and y dimensions of SAM lattice
  OHdens = np.sum(numSurfOH) / 10.06 #hydroxyl surface density (approximately)
  genSize = 8 #size of each generation (for performing simulations, calculating fitness metrics)
              #Also sets number of top performers to consider for evolution
  MutRate = 0.06 #6% mutation rate
  Nmods = int(MutRate*chainsX*chainsY/2.0)
  if Nmods < 1:
    Nmods = 1

  print(str(datetime.now()))
  print("Optimization of surface solute solvation free energy:")
  print("Run parameters:")
  if findMax:
    print("           Type of optimization: Max")
  else:
    print("           Type of optimization: Min")
  print("    Approximate surface density: %3.2f OH/nm^2" % OHdens)
  print("                Generation size: %i" % genSize)
  print("                  Mutation rate: %i per surface" % Nmods)
  print("  Maximum number of generations: %i" % genMax)

  #Read in CH3 and OH chain structure files and put in list
  CH3chain = pmd.load_file(CH3chainFilePrefix+'.top', xyz=CH3chainFilePrefix+'_tilted.gro')
  OHchain = pmd.load_file(OHchainFilePrefix+'.top', xyz=OHchainFilePrefix+'_tilted.gro')
  chainStructs = [CH3chain, OHchain]

  #Create lists to keep track of tested structures
  try:
    with open('structure_library.pkl', 'r') as infile:
      allStructs = pickle.load(infile)
  except IOError:
    allStructs = []

  #Also create list for all solvation fitness metrics o use every time we sort
  allMetrics = [struct.metric for struct in allStructs]

  #Also create temporary lists for current working structures and fitness metrics
  #Allow for restarts, though
  if len(allStructs) == 0:
    #Restarts should work, but currently cannot seed with specified starting structures
    #unless have already run them to get fitness metrics and set up structure classes
    #Until then, just have to get through random generation of surfaces at least once
    currStructs = genSurfsSAM(chainStructs, genSize, Nmods, [], 0, numSurfOH, chainsX, chainsY, doMerge=False)
    currMetrics = np.zeros(genSize).tolist()
    genCount = 0
  else:
    #Need to make sure structures sorted by fitness metric, smallest to largest
    combinedList = sorted(zip(allMetrics, allStructs))
    allStructs = [x for (y,x) in combinedList]
    allMetrics = [x for (x,y) in combinedList]
    genCount = np.max([struct.gen for struct in allStructs])
    #Pick current set of indices to use as parents for next generation
    #Do this by clustering, sorting, then tournament selection
    #Not doing clustering anymore!
    #Instead changing process... more initial random structures, then pooling multiple short optimizations
#    clustList = clusterSurfs(allStructs, 'SAM', 0.42, MaxIter=200, MaxCluster=-32, 
#                                                       MaxClusterWork=None, Verbose=False)
#    #Note that forcing configurations into MAX of 32 clusters
#    #setting the cutoff to 0.42 seems to provide good clustering, based on some experience
#    for k, aclust in enumerate(clustList):
#      thisMetrics = [asurf.metric for asurf in aclust]
#      thisCombinedList = sorted(zip(thisMetrics, aclust))
#      clustList[k] = [x for (y,x) in thisCombinedList]
#    #May not have 32 clusters, so fill out 32-member bracket by looping through clusters
#    #Note that 32 = genSize*bracketSize*2 -> seems to work well for creating some diversity but also drive
#    bracketSurfs = []
#    for k in range(len(clustList[0])): #zeroth should be biggest cluster
#      for aclust in clustList:
#        if len(aclust) > k:
#          if findMax:
#            bracketSurfs.append(aclust[-(k+1)])
#          else:
#            bracketSurfs.append(aclust[k])
#        if len(bracketSurfs) >= 32:
#          break
#      #Fancy break structure below breaks out of outer loop if break in inner loop triggered
#      else:
#        continue
#      break
#    print("After clustering and sorting, have %i surfaces going into bracket" % len(bracketSurfs))
    if findMax:
      bracketSurfs = allStructs[-32:]
    else:
      bracketSurfs = allStructs[:32]
    bracketMetrics = [asurf.metric for asurf in bracketSurfs]
    currInds = tournamentSelect(bracketMetrics, Nparents=genSize, bracketSize=2, optMax=findMax)
    #Below uses bracket size proportional to population size
    #currInds = tournamentSelect(allMetrics, Nparents=genSize, bracketSize=None, optMax=findMax)
    currStructs = [bracketSurfs[x] for x in currInds]
    currMetrics = [bracketMetrics[x] for x in currInds]
    print("From up to generation %i, have chosen following parents:" % genCount)
    for struct in currStructs:
      print("%s, %f, " % (struct.topFile, struct.metric))
    genCount += 1

  print("\nSet-up finished. Beginning genetic algorithm optimization.")

  #Loop until stopping criteria reached, for now number of iterations/generations
  while genCount <= genMax:

    if genCount != 0:
      #If at zeroth generation, already handled above, otherwise, check if this generation
      #should be random or driven by genetic algorithm
      if genCount >= randomGens:
        #Need to take current surfaces and produce new generation
        #Assumes currStructs contains overall most fit candidates only
        #Newly generated surfaces will be simulated and fitness metric determined
        #Then sorting will be re-performed and currStructs updated
        currStructs = genSurfsSAM(chainStructs, genSize, Nmods, currStructs, genCount, 
                                                   numSurfOH, chainsX, chainsY, doMerge=True)
      else:
        #If not past randomGens, need to generate random surfaces
        #Should always be done at least once, i.e. randomGens should be greater than zero
        currStructs = genSurfsSAM(chainStructs, genSize, Nmods, [], genCount, 
                                                   numSurfOH, chainsX, chainsY, doMerge=False)

    #Evaluate the metric function for each surface in currStructs to update currMetrics
    currMetrics = metricFunc(currStructs, **metricFuncArgs)
    if isinstance(currMetrics, np.ndarray):
      currMetrics = currMetrics.tolist()
    for k, astruct in enumerate(currStructs):
      astruct.metric = currMetrics[k]

    #Add current structures and fitness metric to total list
    allStructs = allStructs + currStructs
    allMetrics = allMetrics + currMetrics 

    #Sort by the fitness metric of interest
    combinedList = sorted(zip(allMetrics, allStructs))
    allStructs = [x for (y,x) in combinedList]
    allMetrics = [x for (x,y) in combinedList]

    #Save current structure library information (contains fitness metrics)
    with open('structure_library.pkl', 'w') as infile:
      pickle.dump(allStructs, infile)

    #Select indices to use as parents for next generation
    #Do this by clustering, sorting, then tournament selection
    #Not doing clustering anymore!
    #Instead changing process... more initial random structures, then pooling multiple short optimizations
#    clustList = clusterSurfs(allStructs, 'SAM', 0.42, MaxIter=200, MaxCluster=-32, 
#                                                        MaxClusterWork=None, Verbose=False)
#    #Note that forcing configurations into MAX of 32 clusters
#    #setting the cutoff to 0.42 seems to provide good clustering, based on some experience
#    for k, aclust in enumerate(clustList):
#      thisMetrics = [asurf.metric for asurf in aclust]
#      thisCombinedList = sorted(zip(thisMetrics, aclust))
#      clustList[k] = [x for (y,x) in thisCombinedList]
#    #May not have 32 clusters, so fill out 32-member bracket by looping through clusters
#    #Note that 32 = genSize*bracketSize*2 -> seems to work well for creating some diversity but also drive
#    bracketSurfs = []
#    for k in range(len(clustList[0])): #zeroth should be biggest cluster
#      for aclust in clustList:
#        if len(aclust) > k:
#          if findMax:
#            bracketSurfs.append(aclust[-(k+1)])
#          else:
#            bracketSurfs.append(aclust[k])
#        if len(bracketSurfs) >= 32:
#          break
#      #Fancy break structure below breaks out of outer loop if break in inner loop triggered
#      else:
#        continue
#      break
#    print("After clustering and sorting, have %i surfaces going into bracket" % len(bracketSurfs))
    if findMax:
      bracketSurfs = allStructs[-32:]
    else:
      bracketSurfs = allStructs[:32]
    bracketMetrics = [asurf.metric for asurf in bracketSurfs]
    currInds = tournamentSelect(bracketMetrics, Nparents=genSize, bracketSize=2, optMax=findMax)
    #Below uses bracket size proportional to population size instead
    #currInds = tournamentSelect(allMetrics, Nparents=genSize, bracketSize=None, optMax=findMax)
    currStructs = [bracketSurfs[x] for x in currInds]
    currMetrics = [bracketMetrics[x] for x in currInds]
    print("From up to generation %i, have chosen following parents:" % genCount)
    for struct in currStructs:
      print("%s, %f, " % (struct.topFile, struct.metric))

    if findMax:
      print("Current optimum (max) solvation free energy at generation %i: %f" % (genCount, allMetrics[-1]))
      print("From structure: %s  (%f)\n" % (allStructs[-1].topFile, allStructs[-1].metric))
    else:
      print("Current optimum (min) solvation free energy at generation %i: %f" % (genCount, allMetrics[0]))
      print("From structure: %s  (%f)\n" % (allStructs[0].topFile, allStructs[0].metric))

    genCount += 1


#Now a function that handles charged SAM head-groups and a neutral background head-group
def doOptSAMcharged(numSurfCharged, genMax, randomGens, metricFunc, metricFuncArgs={}, findMax=False, NeutralchainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/ctmSAM', NegchainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/stmSAM', PoschainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/ntmSAM'):

  Usage = """
    Uses genetic algorithms to optimize solute solvation free energy near a SAM surface by
    adjusting the locations of two types of charged chains at fixed density. Neutral chain 
    lattice locations are treated as 0 (neutral chain), while negative (sulfonate chains) 
    are 1 and positive (quaternary ammonium) 2. The resulting integer array specifies where 
    the charged chains are located on the standard lattice structure, with a mappping 
    assumed such that point (0,0)->0 in the array, (0,1)->1, (0,2)->2, ... (N,M)->N*M+M. 
    The genetic algorithm uses binary tournament selection in order to pick parents. 
    Generation and mutation steps follow. 

    One must select the way in which fitness metrics are calculated by providing
    a function object to doOpt that performs this calculation. This allows for 
    simple switching between molecular dynamics and machine-learned functions
    to evaluate fitness.

    numSurfCharged - total number of charged headgroups (per side) for desired density
                     SHOULD be length two array if you want to be really precise. 
                     Does NOT assume that the head-groups are of equivalent charge or 
                     that charge neutrality is maintained, so BE CAREFUL. 
                     This provides flexibility to use different net charges on 
                     positive/negative headgroups, but also makes it possible to 
                     SCREW UP REALLY BADLY.
                     If it's just a float or length 1 array, it WILL be assumed that 
                     you want the same number of both charged chain types, REGARDLESS
                     OF CHARGE.
    genMax - the number of generations that will be produced
    randomGens - the number of generations before optimization starts (random generations)
    metricFunc - A function object that will be used to obtain the fitness metric.
                 This MUST take a list of SAM structure objects as input.
                 It MUST output a list of fitness metrics of the same as the input.
    metricFuncArgs - (default empty dictionary) A dictionary of keyword arguments defining other 
                     parameters for the metric function object to use. Using keyword arguments 
                     to force the user to be very explicit and automatically check to make sure 
                     variables match up.
    findMax - (default False) whether or not to optimizes to maximum (True) versus minimum (False)
    NeutralchainFilePrefix - (optional) file prefix specifying the location before .top
                             and .gro suffixes for the neutral-terminated single chain
    NegchainFilePrefix - (optional) file prefix for the chain with negative termination
    PoschainFilePrefix - (optional) file prefix for the chain with positive termination
  """

  #Set up directory structure
  try:
    os.mkdir('structures') #Holds all surface structures tested
  except OSError:
    pass

  chainsX = 6
  chainsY = 8 #Number of chains along x and y dimensions of SAM lattice
  ChargeDens = np.array([numSurfCharged]).flatten() / 10.06 #hydroxyl surface density (approximately)
  genSize = 8 #size of each generation (for performing simulations, calculating fitness metrics)
              #Also sets number of top performers to consider for evolution
  MutRate = 0.06 #6% mutation rate
  Nmods = int(MutRate*chainsX*chainsY/2.0)
  if Nmods < 1:
    Nmods = 1

  print(str(datetime.now()))
  print("Optimization of surface solute solvation free energy:")
  print("Run parameters:")
  if findMax:
    print("           Type of optimization: Max")
  else:
    print("           Type of optimization: Min")
  if len(ChargeDens == 1):
    print("    Have specified single density for all charged chain types of approximately: %3.2f 1/nm^2" % ChargeDens[0])
  elif len(ChargeDens == 2): 
    print("    Approximate surface density for first charged chain type (default negative): %3.2f 1/nm^2" % ChargeDens[0])
    print("    Approximate surface density for second charged chain type (default positive): %3.2f 1/nm^2" % ChargeDens[1])
  else:
    print("    Have specified number of chains for more than two chain types - only supports two charged chain types.")
    sys.exit(2)
  print("                Generation size: %i" % genSize)
  print("                  Mutation rate: %i per surface" % Nmods)
  print("  Maximum number of generations: %i" % genMax)

  #Read in CH3 and OH chain structure files and put in list
  neutralChain = pmd.load_file(NeutralchainFilePrefix+'.top', xyz=NeutralchainFilePrefix+'_tilted.gro')
  negChain = pmd.load_file(NegchainFilePrefix+'.top', xyz=NegchainFilePrefix+'_tilted.gro')
  posChain = pmd.load_file(PoschainFilePrefix+'.top', xyz=PoschainFilePrefix+'_tilted.gro')
  chainStructs = [neutralChain, negChain, posChain]

  #Create lists to keep track of tested structures
  try:
    with open('structure_library.pkl', 'r') as infile:
      allStructs = pickle.load(infile)
  except IOError:
    allStructs = []

  #Also create list for all solvation fitness metrics o use every time we sort
  allMetrics = [struct.metric for struct in allStructs]

  #Also create temporary lists for current working structures and fitness metrics
  #Allow for restarts, though
  if len(allStructs) == 0:
    #Restarts should work, but currently cannot seed with specified starting structures
    #unless have already run them to get fitness metrics and set up structure classes
    #Until then, just have to get through random generation of surfaces at least once
    currStructs = genSurfsSAM(chainStructs, genSize, Nmods, [], 0, numSurfCharged, chainsX, chainsY, doMerge=False)
    currMetrics = np.zeros(genSize).tolist()
    genCount = 0
  else:
    #Need to make sure structures sorted by fitness metric, smallest to largest
    combinedList = sorted(zip(allMetrics, allStructs))
    allStructs = [x for (y,x) in combinedList]
    allMetrics = [x for (x,y) in combinedList]
    genCount = np.max([struct.gen for struct in allStructs])
    #Pick current set of indices to use as parents for next generation
    #Do this by clustering, sorting, then tournament selection
    #Not doing clustering anymore!
    #Instead changing process... more initial random structures, then pooling multiple short optimizations
    if findMax:
      bracketSurfs = allStructs[-32:]
    else:
      bracketSurfs = allStructs[:32]
    bracketMetrics = [asurf.metric for asurf in bracketSurfs]
    currInds = tournamentSelect(bracketMetrics, Nparents=genSize, bracketSize=2, optMax=findMax)
    #Below uses bracket size proportional to population size
    #currInds = tournamentSelect(allMetrics, Nparents=genSize, bracketSize=None, optMax=findMax)
    currStructs = [bracketSurfs[x] for x in currInds]
    currMetrics = [bracketMetrics[x] for x in currInds]
    print("From up to generation %i, have chosen following parents:" % genCount)
    for struct in currStructs:
      print("%s, %f, " % (struct.topFile, struct.metric))
    genCount += 1

  print("\nSet-up finished. Beginning genetic algorithm optimization.")

  #Loop until stopping criteria reached, for now number of iterations/generations
  while genCount <= genMax:

    if genCount != 0:
      #If at zeroth generation, already handled above, otherwise, check if this generation
      #should be random or driven by genetic algorithm
      if genCount >= randomGens:
        #Need to take current surfaces and produce new generation
        #Assumes currStructs contains overall most fit candidates only
        #Newly generated surfaces will be simulated and fitness metric determined
        #Then sorting will be re-performed and currStructs updated
        currStructs = genSurfsSAM(chainStructs, genSize, Nmods, currStructs, genCount, 
                                                   numSurfCharged, chainsX, chainsY, doMerge=True)
      else:
        #If not past randomGens, need to generate random surfaces
        #Should always be done at least once, i.e. randomGens should be greater than zero
        currStructs = genSurfsSAM(chainStructs, genSize, Nmods, [], genCount, 
                                                   numSurfCharged, chainsX, chainsY, doMerge=False)

    #Evaluate the metric function for each surface in currStructs to update currMetrics
    currMetrics = metricFunc(currStructs, **metricFuncArgs)
    if isinstance(currMetrics, np.ndarray):
      currMetrics = currMetrics.tolist()
    for k, astruct in enumerate(currStructs):
      astruct.metric = currMetrics[k]

    #Add current structures and fitness metric to total list
    allStructs = allStructs + currStructs
    allMetrics = allMetrics + currMetrics 

    #Sort by the fitness metric of interest
    combinedList = sorted(zip(allMetrics, allStructs))
    allStructs = [x for (y,x) in combinedList]
    allMetrics = [x for (x,y) in combinedList]

    #Save current structure library information (contains fitness metrics)
    with open('structure_library.pkl', 'w') as infile:
      pickle.dump(allStructs, infile)

    #Select indices to use as parents for next generation
    #Do this by clustering, sorting, then tournament selection
    #Not doing clustering anymore!
    #Instead changing process... more initial random structures, then pooling multiple short optimizations
    if findMax:
      bracketSurfs = allStructs[-32:]
    else:
      bracketSurfs = allStructs[:32]
    bracketMetrics = [asurf.metric for asurf in bracketSurfs]
    currInds = tournamentSelect(bracketMetrics, Nparents=genSize, bracketSize=2, optMax=findMax)
    #Below uses bracket size proportional to population size instead
    #currInds = tournamentSelect(allMetrics, Nparents=genSize, bracketSize=None, optMax=findMax)
    currStructs = [bracketSurfs[x] for x in currInds]
    currMetrics = [bracketMetrics[x] for x in currInds]
    print("From up to generation %i, have chosen following parents:" % genCount)
    for struct in currStructs:
      print("%s, %f, " % (struct.topFile, struct.metric))

    if findMax:
      print("Current optimum (max) solvation free energy at generation %i: %f" % (genCount, allMetrics[-1]))
      print("From structure: %s  (%f)\n" % (allStructs[-1].topFile, allStructs[-1].metric))
    else:
      print("Current optimum (min) solvation free energy at generation %i: %f" % (genCount, allMetrics[0]))
      print("From structure: %s  (%f)\n" % (allStructs[0].topFile, allStructs[0].metric))

    genCount += 1

  
