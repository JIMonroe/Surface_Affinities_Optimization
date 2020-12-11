#! /usr/bin/env python

import sys, os
import shutil
import glob
import pickle
from genetic_lib import SAMStructure
from genetic_lib import combineLibraries
from opt_lib import doOptSAM
from calc_fitness_lib import MDdGsolv

Usage="""python run_opt.py numHydroxyls soluteName
  Wraps the opt_SAM code for greater flexibility. Still runs same deterministic
  bracket genetic algorithm, but this code may be adjusted to change the process/tweak the algorithm.
  For instance, adjusting this code will allow one to adjust the number of random surfaces generated
  before optimization starts, and to adjust the number of optimizations that are performed and pooled
  before a final optimization run. This should thus be seen as a template. Additionally, the process
  of automating both a minimization and a maximization is possible.

  numHydroxyls - the number of OH-terminated chains desired (determines surface density)
  soluteName - the name of the solute to use as a prefix for grabbing .gro and .top files from FFfiles
"""

def main(args):

  #Set some default numbers
  randGens = 11 #Generations 0 to 10 will be randomly generated, resulting in 88 random structures
  optGens = 20 #Number of generations for each optimization run
  genSize = 8 #Number of individuals in each generation

  #Get the desired number of hydroxyl-terminated chains in the SAM
  try:
    numOH = int(args[0])
  except ValueError:
    print("Must pass single argument to specify number of desired hydroxyls, must be castable to int.")
    print(Usage)
    sys.exit(2)

  #Get the solute to simulate
  try:
    soluteName = args[1]
  except IndexError:
    print("Must provide the name (file prefix) for the solute you want to simulate.")
    print(Usage)
    sys.exit(2)

  workingDir = os.getcwd()

  ##################################### Minimization Run ##############################################

  #Set up a list to hold order of structure libraries generated, and a dictionary with it
  libListMin = []
  libDirDictMin = {}

  #Start by creating a library of random structures over randGens generations
  doOptSAM(numOH, randGens-1, randGens, 
           MDdGsolv, {'solPrefix':soluteName}, 
           findMax=False, 
           CH3chainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/ctmSAM',
           OHchainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/otmSAM')
  os.mkdir(workingDir+'/random_library')
  shutil.copytree(workingDir+'/structures', workingDir+'/random_library/structures')
  shutil.copy(workingDir+'/structure_library.pkl', workingDir+'/random_library/structure_library.pkl')
  libListMin.append(workingDir+'/random_library/structure_library.pkl')
  libDirDictMin[workingDir+'/random_library/structure_library.pkl'] = workingDir+'/random_library'

  #Now do optGens generations of minimization as the 1st minimization run
  doOptSAM(numOH, optGens+randGens-1, randGens, 
           MDdGsolv, {'solPrefix':soluteName}, 
           findMax=False, 
           CH3chainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/ctmSAM',
           OHchainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/otmSAM')
  os.mkdir(workingDir+'/minrun1')
  shutil.move(workingDir+'/structures', workingDir+'/minrun1/structures')
  shutil.move(workingDir+'/structure_library.pkl', workingDir+'/minrun1/structure_library.pkl')
  libListMin.append(workingDir+'/minrun1/structure_library.pkl')
  libDirDictMin[workingDir+'/minrun1/structure_library.pkl'] = workingDir+'/minrun1'
  #This time, need to modify the structure_library file to remove the randGens random generations
  with open(libListMin[-1], 'r') as inFile:
    libOpt1 = pickle.load(inFile)
  gensOpt1 = [struct.gen for struct in libOpt1]
  sortedOpt1 = [x for (y,x) in sorted(zip(gensOpt1, libOpt1))]
  with open(libListMin[-1], 'w') as outFile:
    pickle.dump(sortedOpt1[-(genSize*optGens):], outFile)
  #And remove first randGens random generation files from structures directory
  for agen in range(randGens):
    toremove = glob.glob(libDirDictMin[libListMin[-1]]+'/structures/gen'+str(agen)+'_*')
    for afile in toremove:
      os.remove(afile)

  #Do another optGens generation minimization also starting from the random generations as 2nd minimization
  shutil.copy(libListMin[0], workingDir+'/structure_library.pkl')
  shutil.copytree(libDirDictMin[libListMin[0]]+'/structures', workingDir+'/structures')
  doOptSAM(numOH, optGens+randGens-1, randGens, 
           MDdGsolv, {'solPrefix':soluteName}, 
           findMax=False, 
           CH3chainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/ctmSAM',
           OHchainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/otmSAM')
  os.mkdir(workingDir+'/minrun2')
  shutil.move(workingDir+'/structures', workingDir+'/minrun2/structures')
  shutil.move(workingDir+'/structure_library.pkl', workingDir+'/minrun2/structure_library.pkl')
  libListMin.append(workingDir+'/minrun2/structure_library.pkl')
  libDirDictMin[workingDir+'/minrun2/structure_library.pkl'] = workingDir+'/minrun2'
  #Again need to modify the structure_library file to remove the randGens random generations
  with open(libListMin[-1], 'r') as inFile:
    libOpt2 = pickle.load(inFile)
  gensOpt2 = [struct.gen for struct in libOpt2]
  sortedOpt2 = [x for (y,x) in sorted(zip(gensOpt2, libOpt2))]
  with open(libListMin[-1], 'w') as outFile:
    pickle.dump(sortedOpt2[-(genSize*optGens):], outFile)
  #And remove first randGens random generation files from structures directory
  for agen in range(randGens):
    toremove = glob.glob(libDirDictMin[libListMin[-1]]+'/structures/gen'+str(agen)+'_*')
    for afile in toremove:
      os.remove(afile)

  #Finally, combine the previous results into a single library, then do a final optGens generation minimization
  combineLibraries(libListMin, libDirDictMin)
  doOptSAM(numOH, 3*optGens+randGens-1, randGens, 
           MDdGsolv, {'solPrefix':soluteName}, 
           findMax=False, 
           CH3chainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/ctmSAM',
           OHchainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/otmSAM')
  #Can stop here if happy, but if also want to automate maximization, need to move files around again
  #Won't bother deleting any files from library or structures directory, though
  os.mkdir(workingDir+'/final_min')
  shutil.move(workingDir+'/structures', workingDir+'/final_min/structures')
  shutil.move(workingDir+'/structure_library.pkl', workingDir+'/final_min/structure_library.pkl')
 

  ##################################### Maximization Run ##############################################

  #Now do maximization with same procedure, but can skip creating the random library
  #Set up a list to hold order of structure libraries generated, and a dictionary with it
  libListMax = []
  libDirDictMax = {}

  libListMax.append(workingDir+'/random_library/structure_library.pkl')
  libDirDictMax[workingDir+'/random_library/structure_library.pkl'] = workingDir+'/random_library'

  #Now do optGens generations of maximization as the 1st maximization run
  shutil.copy(libListMax[0], workingDir+'/structure_library.pkl')
  shutil.copytree(libDirDictMax[libListMax[0]]+'/structures', workingDir+'/structures')
  doOptSAM(numOH, optGens+randGens-1, randGens, 
           MDdGsolv, {'solPrefix':soluteName}, 
           findMax=True, 
           CH3chainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/ctmSAM',
           OHchainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/otmSAM')
  os.mkdir(workingDir+'/maxrun1')
  shutil.move(workingDir+'/structures', workingDir+'/maxrun1/structures')
  shutil.move(workingDir+'/structure_library.pkl', workingDir+'/maxrun1/structure_library.pkl')
  libListMax.append(workingDir+'/maxrun1/structure_library.pkl')
  libDirDictMax[workingDir+'/maxrun1/structure_library.pkl'] = workingDir+'/maxrun1'
  #This time, need to modify the structure_library file to remove the randGens random generations
  with open(libListMax[-1], 'r') as inFile:
    libOpt1 = pickle.load(inFile)
  gensOpt1 = [struct.gen for struct in libOpt1]
  sortedOpt1 = [x for (y,x) in sorted(zip(gensOpt1, libOpt1))]
  with open(libListMax[-1], 'w') as outFile:
    pickle.dump(sortedOpt1[-(genSize*optGens):], outFile)
  #And remove first randGens random generation files from structures directory
  for agen in range(randGens):
    toremove = glob.glob(libDirDictMax[libListMax[-1]]+'/structures/gen'+str(agen)+'_*')
    for afile in toremove:
      os.remove(afile)

  #Do another optGens generation maximization also starting from the random generations as 2nd maximization
  shutil.copy(libListMax[0], workingDir+'/structure_library.pkl')
  shutil.copytree(libDirDictMax[libListMax[0]]+'/structures', workingDir+'/structures')
  doOptSAM(numOH, optGens+randGens-1, randGens, 
           MDdGsolv, {'solPrefix':soluteName}, 
           findMax=True, 
           CH3chainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/ctmSAM',
           OHchainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/otmSAM')
  os.mkdir(workingDir+'/maxrun2')
  shutil.move(workingDir+'/structures', workingDir+'/maxrun2/structures')
  shutil.move(workingDir+'/structure_library.pkl', workingDir+'/maxrun2/structure_library.pkl')
  libListMax.append(workingDir+'/maxrun2/structure_library.pkl')
  libDirDictMax[workingDir+'/maxrun2/structure_library.pkl'] = workingDir+'/maxrun2'
  #Again need to modify the structure_library file to remove the randGens random generations
  with open(libListMax[-1], 'r') as inFile:
    libOpt2 = pickle.load(inFile)
  gensOpt2 = [struct.gen for struct in libOpt2]
  sortedOpt2 = [x for (y,x) in sorted(zip(gensOpt2, libOpt2))]
  with open(libListMax[-1], 'w') as outFile:
    pickle.dump(sortedOpt2[-(genSize*optGens):], outFile)
  #And remove first randGens random generation files from structures directory
  for agen in range(randGens):
    toremove = glob.glob(libDirDictMax[libListMax[-1]]+'/structures/gen'+str(agen)+'_*')
    for afile in toremove:
      os.remove(afile)

  #Finally, combine the previous results into a single library, then do a final optGens generation maximization
  combineLibraries(libListMax, libDirDictMax)
  doOptSAM(numOH, 3*optGens+randGens-1, randGens, 
           MDdGsolv, {'solPrefix':soluteName}, 
           findMax=True, 
           CH3chainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/ctmSAM',
           OHchainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/otmSAM')
  os.mkdir(workingDir+'/final_max')
  shutil.move(workingDir+'/structures', workingDir+'/final_max/structures')
  shutil.move(workingDir+'/structure_library.pkl', workingDir+'/final_max/structure_library.pkl')
 

if __name__ == "__main__":
  main(sys.argv[1:])
 
