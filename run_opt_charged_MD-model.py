#! /usr/bin/env python

import sys, os
import shutil
import glob
import pickle
from genetic_lib import SAMStructure
from genetic_lib import combineLibraries
from opt_lib import doOptSAMcharged
from calc_fitness_lib import MDdGsolv
from calc_fitness_lib import linregdGsolv
from metric_models import *
from property_functions import gr_int
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

Usage="""python run_opt.py numCharged soluteName
  Wraps the opt_lib code for greater flexibility. Still runs same deterministic
  bracket genetic algorithm, but this code may be adjusted to change the process/tweak the algorithm.
  For instance, adjusting this code will allow one to adjust the number of random surfaces generated
  before optimization starts, and to adjust the number of optimizations that are performed and pooled
  before a final optimization run. This should thus be seen as a template. Additionally, the process
  of automating both a minimization and a maximization is possible.

  numCharged - the number of chains with charged head-groups (determines surface density)
               same number specifies both positive and negative, so change this if have different net charges
  soluteName - the name of the solute to use as a prefix for grabbing .gro and .top files from FFfiles
"""

def main(args):

  #Set some default numbers
  randGens = 6 #Generations 0 to 5 will be randomly generated, resulting in 48 random structures
  optGensMD = 5 #Number of generations for each optimization run of MD
  optGensModel = 30 #Number of generations for optimization runs with predictions from a model
  genSize = 8 #Number of individuals in each generation

  #Get the desired number of hydroxyl-terminated chains in the SAM
  try:
    numCharged = int(args[0])
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

  #Set up models for free energies of solvation based on surface RDFs
  #Assuming just three chain types, one neutral background and two charged
  #Fits with rest of code, but could be generalized
  initial_props = []
  for i in range(3):
    for i_bin in range(15):
      this_property = rdfLinRegProperty(label='gr_%i_%i'%(i, i_bin), property_function=gr_int)
      initial_props.append(this_property)
  #lin_reg_model = linear_model.LinearRegression()
  #dGsolvModel = rdfLinRegModel(initial_properties=initial_props, model=lin_reg_model)
  lasso_model = linear_model.Lasso(alpha=1e-4) # HARD CODING HERE
  scaler = StandardScaler()
  dGsolvModel = rdfLinRegModel(initial_properties=initial_props, model=lasso_model, scaler=scaler, rmse_norm_threshold=0.9)
 
 
  ##################################### Starting MD optimization ##############################################

  #Set up a list to hold order of structure libraries generated, and a dictionary with it
  libList = []
  libDirDict = {}

  #Start by creating a library of random structures over randGens generations
  doOptSAMcharged(numCharged, randGens-1, randGens, 
                  MDdGsolv, {'solPrefix':soluteName}, 
                  findMax=False, 
                  NeutralchainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/ctmSAM',
                  NegchainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/stmSAM',
                  PoschainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/ntmSAM')
  #Copy this somewhere safe
  os.mkdir(workingDir+'/random_library')
  shutil.copytree(workingDir+'/structures', workingDir+'/random_library/structures')
  shutil.copy(workingDir+'/structure_library.pkl', workingDir+'/random_library/structure_library.pkl')
  #And add to list of library files and directories
  libList.append(workingDir+'/random_library/structure_library.pkl')
  libDirDict[workingDir+'/random_library/structure_library.pkl'] = workingDir+'/random_library'

  #Now do optGensMD generations of minimization as the 1st MD minimization run
  doOptSAMcharged(numCharged, optGensMD+randGens-1, randGens, 
                  MDdGsolv, {'solPrefix':soluteName}, 
                  findMax=False, 
                  NeutralchainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/ctmSAM',
                  NegchainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/stmSAM',
                  PoschainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/ntmSAM')
  #Move this somewhere safe
  os.mkdir(workingDir+'/MDminrun1')
  shutil.move(workingDir+'/structures', workingDir+'/MDminrun1/structures')
  shutil.move(workingDir+'/structure_library.pkl', workingDir+'/MDminrun1/structure_library.pkl')
  #And add to list of library files and directories
  libList.append(workingDir+'/MDminrun1/structure_library.pkl')
  libDirDict[workingDir+'/MDminrun1/structure_library.pkl'] = workingDir+'/MDminrun1'
  #This time, need to modify the structure_library file to remove the randGens random generations
  with open(libList[-1], 'r') as inFile:
    libOpt = pickle.load(inFile)
  gensOpt = [struct.gen for struct in libOpt]
  sortedOpt = [x for (y,x) in sorted(zip(gensOpt, libOpt))]
  with open(libList[-1], 'w') as outFile:
    pickle.dump(sortedOpt[-(genSize*optGensMD):], outFile)
  #And remove first randGens random generation files from structures directory
  for agen in range(randGens):
    toremove = glob.glob(libDirDict[libList[-1]]+'/structures/gen'+str(agen)+'_*')
    for afile in toremove:
      os.remove(afile)

  #Now do optGensMD generations of maximization as the 1st MD maximization run
  shutil.copy(libList[0], workingDir+'/structure_library.pkl')
  shutil.copytree(libDirDict[libList[0]]+'/structures', workingDir+'/structures')
  doOptSAMcharged(numCharged, optGensMD+randGens-1, randGens,
                  MDdGsolv, {'solPrefix':soluteName}, 
                  findMax=True, 
                  NeutralchainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/ctmSAM',
                  NegchainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/stmSAM',
                  PoschainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/ntmSAM')
  #Move this somewhere safe
  os.mkdir(workingDir+'/MDmaxrun1')
  shutil.move(workingDir+'/structures', workingDir+'/MDmaxrun1/structures')
  shutil.move(workingDir+'/structure_library.pkl', workingDir+'/MDmaxrun1/structure_library.pkl')
  #And add to list of library files and directories
  libList.append(workingDir+'/MDmaxrun1/structure_library.pkl')
  libDirDict[workingDir+'/MDmaxrun1/structure_library.pkl'] = workingDir+'/MDmaxrun1'
  #This time, need to modify the structure_library file to remove the randGens random generations
  with open(libList[-1], 'r') as inFile:
    libOpt = pickle.load(inFile)
  gensOpt = [struct.gen for struct in libOpt]
  sortedOpt = [x for (y,x) in sorted(zip(gensOpt, libOpt))]
  with open(libList[-1], 'w') as outFile:
    pickle.dump(sortedOpt[-(genSize*optGensMD):], outFile)
  #And remove first randGens random generation files from structures directory
  for agen in range(randGens):
    toremove = glob.glob(libDirDict[libList[-1]]+'/structures/gen'+str(agen)+'_*')
    for afile in toremove:
      os.remove(afile)


  #######################################  Starting model prediction  ##################################################

  #Want to now train the model with data from the random generations, the minimization, and maximization runs
  #I think using both min and max will help fit... can switch up if not true.
  #For now ignoring automatic flag to tell us how good the model is
  combineLibraries(libList, libDirDict)
  model_at_par = dGsolvModel.build_model('structure_library.pkl')
  if not model_at_par:
    print("Model not passing criteria for doing well on test-set data, but ignoring warnings and using anyway.")
  with open('dGsolv_LinRegModel.pkl', 'w') as f_model:
    pickle.dump(dGsolvModel, f_model)

  #Now perform minimization and maximization using the model to predict fitness instead of MD
  #For both runs, will just leave the structure library in place, adding all structures to it
  #Won't bother keeping track of where each set of structures actually came from and organizing
  #Just remember how many generations did for each!

  #And do this as independent optimizations with recombination for final optimization
  #(this should help prevent getting stuck at one local optimum)

  #First make copies of the current structure library and structures directory
  shutil.copytree(workingDir+'/structures', workingDir+'/backup_structures')
  shutil.copy(workingDir+'/structure_library.pkl', workingDir+'/backup_structure_library.pkl')

  #Minimization first
  for i in range(1,5):
    #Get the set of structures all the independent optimizations work off of if already done this once
    if i > 1:
      shutil.copytree(workingDir+'/backup_structures', workingDir+'/structures')
      shutil.copy(workingDir+'/backup_structure_library.pkl', workingDir+'/structure_library.pkl')
    #And do minimization
    doOptSAMcharged(numCharged, optGensModel+2*optGensMD+randGens-1, randGens,
                    linregdGsolv, {'model':dGsolvModel},
                    findMax=False,
                    NeutralchainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/ctmSAM',
                    NegchainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/stmSAM',
                    PoschainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/ntmSAM')
    #Move this somewhere safe
    os.mkdir(workingDir+'/Modelminrun%i'%(i))
    shutil.move(workingDir+'/structures', workingDir+'/Modelminrun%i/structures'%(i))
    shutil.move(workingDir+'/structure_library.pkl', workingDir+'/Modelminrun%i/structure_library.pkl'%(i))
    #And add to list of library files and directories
    libList.append(workingDir+'/Modelminrun%i/structure_library.pkl'%(i))
    libDirDict[workingDir+'/Modelminrun%i/structure_library.pkl'%(i)] = workingDir+'/Modelminrun%i'%(i)
    #This time, need to modify the structure_library file to remove the randGens random generations
    with open(libList[-1], 'r') as inFile:
      libOpt = pickle.load(inFile)
    gensOpt = [struct.gen for struct in libOpt]
    sortedOpt = [x for (y,x) in sorted(zip(gensOpt, libOpt))]
    with open(libList[-1], 'w') as outFile:
      pickle.dump(sortedOpt[-(genSize*optGensModel):], outFile)
    #And remove files from structures directory not involved in this optimization
    for agen in range(2*optGensMD+randGens):
      toremove = glob.glob(libDirDict[libList[-1]]+'/structures/gen'+str(agen)+'_*')
      for afile in toremove:
        os.remove(afile)

  #And do final minimization combining all of the independent work up to this point
  #Making this last one twice as long as the others
  combineLibraries(libList, libDirDict)
  doOptSAMcharged(numCharged, 6*optGensModel+2*optGensMD+randGens-1, randGens,
                  linregdGsolv, {'model':dGsolvModel},
                  findMax=False,
                  NeutralchainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/ctmSAM',
                  NegchainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/stmSAM',
                  PoschainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/ntmSAM')
  #Move this somewhere safe
  os.mkdir(workingDir+'/Modelminrun5')
  shutil.move(workingDir+'/structures', workingDir+'/Modelminrun5/structures')
  shutil.move(workingDir+'/structure_library.pkl', workingDir+'/Modelminrun5/structure_library.pkl')
  #And add to list of library files and directories
  libList.append(workingDir+'/Modelminrun5/structure_library.pkl')
  libDirDict[workingDir+'/Modelminrun5/structure_library.pkl'] = workingDir+'/Modelminrun5'
  #This time, need to modify the structure_library file to remove the randGens random generations
  with open(libList[-1], 'r') as inFile:
    libOpt = pickle.load(inFile)
  gensOpt = [struct.gen for struct in libOpt]
  sortedOpt = [x for (y,x) in sorted(zip(gensOpt, libOpt))]
  with open(libList[-1], 'w') as outFile:
    pickle.dump(sortedOpt[-(genSize*2*optGensModel):], outFile)
  #And remove files from structures directory not involved in this optimization
  for agen in range(4*optGensModel+2*optGensMD+randGens):
    toremove = glob.glob(libDirDict[libList[-1]]+'/structures/gen'+str(agen)+'_*')
    for afile in toremove:
      os.remove(afile)
  
  #And maximization
  for i in range(1,5):
    #Get the set of structures all the independent optimizations work off of - not including model minimizations here
    shutil.copytree(workingDir+'/backup_structures', workingDir+'/structures')
    shutil.copy(workingDir+'/backup_structure_library.pkl', workingDir+'/structure_library.pkl')
    #And do maximization
    doOptSAMcharged(numCharged, optGensModel+2*optGensMD+randGens-1, randGens,
                    linregdGsolv, {'model':dGsolvModel},
                    findMax=True,
                    NeutralchainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/ctmSAM',
                    NegchainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/stmSAM',
                    PoschainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/ntmSAM')
    #Move this somewhere safe
    os.mkdir(workingDir+'/Modelmaxrun%i'%(i))
    shutil.move(workingDir+'/structures', workingDir+'/Modelmaxrun%i/structures'%(i))
    shutil.move(workingDir+'/structure_library.pkl', workingDir+'/Modelmaxrun%i/structure_library.pkl'%(i))
    #And add to list of library files and directories
    libList.append(workingDir+'/Modelmaxrun%i/structure_library.pkl'%(i))
    libDirDict[workingDir+'/Modelmaxrun%i/structure_library.pkl'%(i)] = workingDir+'/Modelmaxrun%i'%(i)
    #This time, need to modify the structure_library file to remove the randGens random generations
    with open(libList[-1], 'r') as inFile:
      libOpt = pickle.load(inFile)
    gensOpt = [struct.gen for struct in libOpt]
    sortedOpt = [x for (y,x) in sorted(zip(gensOpt, libOpt))]
    with open(libList[-1], 'w') as outFile:
      pickle.dump(sortedOpt[-(genSize*optGensModel):], outFile)
    #And remove files from structures directory not involved in this optimization
    for agen in range(2*optGensMD+randGens):
      toremove = glob.glob(libDirDict[libList[-1]]+'/structures/gen'+str(agen)+'_*')
      for afile in toremove:
        os.remove(afile)

  #And do final maximization combining all of the independent work up to this point
  #Making this last one twice as long as the others
  #Here just using minimization data too... 
  #Shouldn't make a difference, just need to note to combine libraries correctly later
  #And to figure out how to set the number of generations to go to (and to remove up to when saving)
  combineLibraries(libList, libDirDict)
  doOptSAMcharged(numCharged, 12*optGensModel+2*optGensMD+randGens-1, randGens,
                  linregdGsolv, {'model':dGsolvModel},
                  findMax=True,
                  NeutralchainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/ctmSAM',
                  NegchainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/stmSAM',
                  PoschainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/ntmSAM')
  #Move this somewhere safe
  os.mkdir(workingDir+'/Modelmaxrun5')
  shutil.move(workingDir+'/structures', workingDir+'/Modelmaxrun5/structures')
  shutil.move(workingDir+'/structure_library.pkl', workingDir+'/Modelmaxrun5/structure_library.pkl')
  #And add to list of library files and directories
  libList.append(workingDir+'/Modelmaxrun5/structure_library.pkl')
  libDirDict[workingDir+'/Modelmaxrun5/structure_library.pkl'] = workingDir+'/Modelmaxrun5'
  #This time, need to modify the structure_library file to remove the randGens random generations
  with open(libList[-1], 'r') as inFile:
    libOpt = pickle.load(inFile)
  gensOpt = [struct.gen for struct in libOpt]
  sortedOpt = [x for (y,x) in sorted(zip(gensOpt, libOpt))]
  with open(libList[-1], 'w') as outFile:
    pickle.dump(sortedOpt[-(genSize*2*optGensModel):], outFile)
  #And remove files from structures directory not involved in this optimization
  for agen in range(10*optGensModel+2*optGensMD+randGens):
    toremove = glob.glob(libDirDict[libList[-1]]+'/structures/gen'+str(agen)+'_*')
    for afile in toremove:
      os.remove(afile)

  #Now want to go back and check the model predictions - first combine all libraries up to this point
  combineLibraries(libList, libDirDict)
  #Check model predictions by just running another minimization and maximization with MD evaluating fitness
  #In these last steps, we will be lazy
  #Will stop carefully storing away the structure library file and structures directory
  #Should be just checking and polishing in these last steps anyway
  doOptSAMcharged(numCharged, 12*optGensModel+3*optGensMD+randGens-1, randGens,
                  MDdGsolv, {'solPrefix':soluteName}, 
                  findMax=False, 
                  NeutralchainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/ctmSAM',
                  NegchainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/stmSAM',
                  PoschainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/ntmSAM')
  doOptSAMcharged(numCharged, 12*optGensModel+4*optGensMD+randGens-1, randGens,
                  MDdGsolv, {'solPrefix':soluteName}, 
                  findMax=True, 
                  NeutralchainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/ctmSAM',
                  NegchainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/stmSAM',
                  PoschainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/ntmSAM')

  #And refit the model, but first save it so we keep track of whether or not we're making it better
  shutil.move('dGsolv_LinRegModel.pkl', 'old_dGsolv_LinRegModel.pkl')
  model_at_par = dGsolvModel.build_model('structure_library.pkl')
  if not model_at_par:
    print("Model not passing criteria for doing well with test-set data, but ignoring warnings and using anyway.")
  with open('dGsolv_LinRegModel.pkl', 'w') as f_model:
    pickle.dump(dGsolvModel, f_model)
  
  #Now that the model has been updated, we need to re-evaluate the fitness of all model-generated individuals
  #i.e. we want to use the most accurate model to correct the previous, less accurate one
  with open('structure_library.pkl', 'r') as libfile:
    thislib = pickle.load(libfile)
  for struct in thislib:
    if struct.metricpredicted:
      newmetric = dGsolvModel.model_prediction(struct)
      struct.metric = newmetric
  with open('structure_library.pkl', 'w') as libfile:
    pickle.dump(thislib, libfile)
 
  #And polish off with another round of model optimizations
  doOptSAMcharged(numCharged, 14*optGensModel+4*optGensMD+randGens-1, randGens,
                  linregdGsolv, {'model':dGsolvModel}, 
                  findMax=False, 
                  NeutralchainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/ctmSAM',
                  NegchainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/stmSAM',
                  PoschainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/ntmSAM')
  doOptSAMcharged(numCharged, 16*optGensModel+4*optGensMD+randGens-1, randGens,
                  linregdGsolv, {'model':dGsolvModel}, 
                  findMax=True, 
                  NeutralchainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/ctmSAM',
                  NegchainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/stmSAM',
                  PoschainFilePrefix='/home/jmonroe/Surface_Affinities_Project/FFfiles/ntmSAM')


if __name__ == "__main__":
  main(sys.argv[1:])
 
