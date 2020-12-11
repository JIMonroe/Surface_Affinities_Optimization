#A library defining fitness functions to be used by the genetic algorithm
#This can include running of MD simulations and calculating values based on that
#OR, it can involve running a machine-learned function on a surface representation
#Each function must satisfy certain conditions, however.
#The first and only non-keyword argument it must take is a list of structure objects.
#The next arguments should ALL be keyword arguments.
#A given fitness function must also return a list of fitness metrics the same length 
#as the list of structure objects.

import os
import shutil
from datetime import datetime
import time
import subprocess
import multiprocessing
import glob
import pickle
from genetic_lib import *


def MDdGsolv(currStructs, solPrefix='methane'):
  #Uses MD simulations to calculate solvation free energies for a solute on a number of surfaces
  #Currently set up to run on syrah only, as each evaluation uses the queuing system to submit
  #MD simulations and check if they are done.

  #Set up array to hold metrics we will return
  returnMetrics = np.zeros(len(currStructs))

  gencount = currStructs[0].gen

  print("\nStarting MD simulations for generation %i at: %s" % (gencount, str(datetime.now())))

  #Set up directory reference
  baseDir = os.getcwd()

  #Need to create MD_work if not already present
  try:    
    os.mkdir('MD_work') #Holds directories for each set of MD simulations - note that won't save trajectories!
  except OSError:
    pass

  #Prepare each structure for MD simulation (add water, minimize)
  os.chdir('MD_work')
  toremove = glob.glob('./*')
  for r in toremove:
    shutil.rmtree(r)

  for k, struct in enumerate(currStructs):
    
    os.mkdir('gen%i_struct%i' % (struct.gen, k))
    os.chdir('gen%i_struct%i' % (struct.gen, k))
    shutil.copy(baseDir+'/'+struct.topFile, os.getcwd())
    shutil.copy(baseDir+'/'+struct.strucFile, os.getcwd())
    if solPrefix == 'boricacid': #Have to treat specially because of force-field used
      subprocess.call('sed -e \'s/CUDA_VISIBLE_DEVICES=0/CUDA_VISIBLE_DEVICES=%i/\' -e \'s/methane/%s/\' -e \'s/run_expanded/ba_run_expanded/\' %s > %s' \
                      % (k, solPrefix, '/home/jmonroe/Surface_Affinities_Project/all_code/submit_expanded.sh', 
                         os.getcwd()+'/submit_expanded.sh'), shell=True)
    else:
      subprocess.call('sed -e \'s/CUDA_VISIBLE_DEVICES=0/CUDA_VISIBLE_DEVICES=%i/\' -e \'s/methane/%s/\' %s > %s' \
                      % (k, solPrefix, '/home/jmonroe/Surface_Affinities_Project/all_code/submit_expanded.sh', 
                         os.getcwd()+'/submit_expanded.sh'), shell=True)
    

    #And get MD simulation running (using qsub, though!)
    subprocess.call('qsub submit_expanded.sh', shell=True)

    os.chdir('../')

  os.chdir(baseDir)

  #Now wait until all simulations are finished...
  qstatcheck = subprocess.Popen('qstat', stdout=subprocess.PIPE, shell=True).communicate()[0]
  while qstatcheck != '':
    time.sleep(60)
    qstatcheck = subprocess.Popen('qstat', stdout=subprocess.PIPE, shell=True).communicate()[0]

  print("MD simulations for generation %i finished at: %s" % (gencount, str(datetime.now())))

  #Calculate all the free energies on current surfaces
  currDirs = sorted(glob.glob('MD_work/*'))

  defaultSimDirs = ['Quad_0.25X_0.25Y', 'Quad_0.25X_0.75Y', 'Quad_0.75X_0.25Y', 'Quad_0.75X_0.75Y']
  defaultKXY = [10.0, 10.0, 10.0, 10.0]
  defaultRefX = [7.4550, 7.4550, 22.3650, 22.3650]
  defaultRefY = [8.6083, 25.8249, 8.6083, 25.8249]
  defaultDistRefX = [7.4550, 7.4550, 7.4550, 7.4550]
  defaultDistRefY = [8.6083, 8.6083, 8.6083, 8.6083]
  numLambdaStates = 19

  #For efficiency, do this in parallel with multiprocessing
  manager = multiprocessing.Manager()
  return_dict = manager.dict()
  jobs = []
  for k in range(len(currStructs)):
    thisdirs = [currDirs[k]+'/'+a for a in defaultSimDirs]
    p = multiprocessing.Process(target=doJobCalcdGsolv, 
                                args=(k, return_dict, thisdirs, defaultKXY, 
                                      defaultRefX, defaultRefY, defaultDistRefX, defaultDistRefY, numLambdaStates))
    jobs.append(p)
    p.start()
  for proc in jobs:
    proc.join()

  #Free energies are now stored in return_dict, so get them out
  for k in range(len(currStructs)):

    thisMetric = return_dict[k]
    print("For structure %i in generation %i:" % (k, gencount))
    print("  dGsolv = %f" % thisMetric[0])
    print("  dGsolvError = %f" % thisMetric[1])
    print("  Samples at all states: %s" % str(thisMetric[2].tolist()))
    print("  First state total samples = %i" % thisMetric[3])
    print("  Last state total samples = %i" % thisMetric[4])
    print("  First state min samples in XY bin = %f" % thisMetric[5])
    print("  First state max samples in XY bin = %f" % thisMetric[6])
    print("  Last state min samples in XY bin = %f" % thisMetric[7])
    print("  Last state max samples in XY bin = %f" % thisMetric[8])

    #Also report average temperature and box volume
    thisdirs = [currDirs[k]+'/'+a for a in defaultSimDirs]
    thisT = 0.0
    thisV = 0.0
    for adir in thisdirs: 
      aT, aV = getAvgTV(adir+'/prod_out.txt')
      thisT += aT
      thisV += aV
    thisT /= float(len(thisdirs))
    thisV /= float(len(thisdirs))
    print("  Average temperature = %f" % thisT)
    print("  Average box volume = %f" % thisV)

    returnMetrics[k] = thisMetric[0]

    #At this point, also want to store a representative final, solvated structure and topology
    thisSolvStruc = thisdirs[0]+'/prod.gro'
    copySolvStruc = currStructs[k].topFile.split('.top')[0] + '_solvated.gro'
    shutil.copy(baseDir+'/'+thisSolvStruc, baseDir+'/'+copySolvStruc)
    thisSolvTop = currDirs[k]+'/sol_surf.top'
    copySolvTop = currStructs[k].topFile.split('.top')[0] + '_solvated.top'
    shutil.copy(baseDir+'/'+thisSolvTop, baseDir+'/'+copySolvTop)

  print("All solvation free energies calculated for generation %i at: %s" % (gencount, str(datetime.now())))

  return returnMetrics  


def linregdGsolv(currStructs, model=None, modelfile='dGsolv_LinRegModel.pkl'):
  #Uses a machine-learned model to predict a fitness metric, in this case solvation free energy
  #The code is based on Sally Jiao's work, but adapted to this situation

  #Set up array to hold metrics we will return
  returnMetrics = np.zeros(len(currStructs))

  gencount = currStructs[0].gen

  print("\nStarting predictions based on model for generation %i at: %s" % (gencount, str(datetime.now())))

  #Load the model to use for prediction, but only if not provided to us
  if model is None:
    with open(modelfile, 'r') as f_metricModel:
      metricModel = pickle.load(f_metricModel)
  else:
    metricModel = model

  #Loop over structures
  for k, struct in enumerate(currStructs):

    thismetric = metricModel.model_prediction(struct)
    returnMetrics[k] = thismetric

    #Important to keep track of where the metric came from... MD or a model
    #Do this inside this function so don't have to keep track of in main genetic algorithm code
    #Python is pass-by-reference within functions, so if change this value here, should also change outside as well
    struct.metricpredicted = True

  print("All solvation free energies predicted for generation %i at: %s" % (gencount, str(datetime.now())))

  return returnMetrics

    
def MDselect(currStructs, solNames=['methanol', 'boricacid']):
  #Runs two sets of MD simulations, one for each solute, to calculate solvation free energies at interfaces
  #Then approximates selectivity by taking the difference between the two
  #Note that takes the second (or last) solute and subtracts the first
  #If maximize this quantity, get high selectivity for the second solute compared to the first
  #If minimize this quantity, will make the surface as selective as possible for the first solute
  #This could mean preferential binding of the first solute, or just low relative selectivity for the second
  #Or all this vice versa
  #This should allow more diverse behavior than taking absolute value of difference (maybe important on charged interfaces)
 
  gencount = currStructs[0].gen

  #Will call MDdGsolv for one solute, then the other
  dGvals = np.zeros((len(solNames), len(currStructs)))

  for k, aname in enumerate(solNames):
    print("\nComputing solvation free energy for solute %i, %s"%(k, aname))
    dGvals[k,:] = MDdGsolv(currStructs, solPrefix=aname)

  returnMetrics = dGvals[-1,:] - dGvals[0,:]

  print("\nSelectivity proxies (dGsolv2 - dGsolv1) for %s (1) and %s (2) at generation %i:"%(solNames[0], solNames[-1], gencount))

  for k in range(len(currStructs)):
    print("\tStructure %i in generation %i: %f kB*T" % (k, gencount, returnMetrics[k]))

  return returnMetrics


def linregselect(currStructs, model=None, modelfile='select_LinRegModel.pkl'):
  #Now instead of modelling dGsolv, modelling selectivity, so do a little differently than MDselectivity function
  #Could have re-used linRegdGsolv, but nice to have accurate print statements and different default file names
  
  #Set up array to hold metrics we will return
  returnMetrics = np.zeros(len(currStructs))

  gencount = currStructs[0].gen

  print("\nStarting predictions based on model for generation %i at: %s" % (gencount, str(datetime.now())))

  #Load the model to use for prediction, but only if not provided to us
  if model is None:
    with open(modelfile, 'r') as f_metricModel:
      metricModel = pickle.load(f_metricModel)
  else:
    metricModel = model

  #Loop over structures
  for k, struct in enumerate(currStructs):

    thismetric = metricModel.model_prediction(struct)
    returnMetrics[k] = thismetric

    #Important to keep track of where the metric came from... MD or a model
    #Do this inside this function so don't have to keep track of in main genetic algorithm code
    #Python is pass-by-reference within functions, so if change this value here, should also change outside as well
    struct.metricpredicted = True

  print("All selectivities predicted for generation %i at: %s" % (gencount, str(datetime.now())))

  return returnMetrics

 
