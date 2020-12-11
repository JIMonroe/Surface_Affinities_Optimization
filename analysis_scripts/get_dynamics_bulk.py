#!/usr/bin/env python

from __future__ import division, print_function

import sys, os
import copy
import numpy as np
import simtk.openmm as mm
import simtk.openmm.app as app
import simtk.unit as u
import parmed as pmd
from parmed.openmm.reporters import NetCDFReporter
from pymbar import mbar
from openmm_surface_affinities_lib import *
import waterlib as wl
from scipy import optimize

#Given a topology and structure file, this script sets up a simulation of a solvated solute
#(this code is for just in bulk!) and periodically kicks off NVE simulations from the NPT
#configurations and temperatures. These NVE simulations are then used to assess dynamics, while
#the trajectory in the NPT can be used to evaluate solute properties in the fully coupled 
#ensemble. The below script applies to bulk systems. 


def normalExponential(t, A, Tau):
  #A function to define a normal exponential for fitting decay of water residency in shells 
  return A*np.exp(-(t/Tau))


def stretchedExponential(t, A, Tau, B):
  #A function to define a stretched exponential for fitting the dipole vector autocorrelation function
  return A*np.exp(-(t/Tau)**B)


def doSimDynamics(top, systemRef, integratorRef, platform, prop, temperature, scalexy=False, inBulk=False, coupled=True, state=None, pos=None, vels=None, nSteps=10000000):
  #Input a topology object, reference system, integrator, platform, platform properties, 
  #and optionally state file, positions, or velocities
  #If state is specified including positions and velocities and pos and vels are not None, the 
  #positions and velocities from the provided state will be overwritten
  #Does NPT, stopping periodically to run NVE to compute dynamics
  #Only the NPT simulation will be saved, not the NVE 

  #Copy the reference system and integrator objects
  system = copy.deepcopy(systemRef)
  integrator = copy.deepcopy(integratorRef)

  #For NPT, add the barostat as a force
  #If not in bulk, use anisotropic barostat
  if not inBulk:
    system.addForce(mm.MonteCarloAnisotropicBarostat((1.0, 1.0, 1.0)*u.bar,
                                                     temperature, #Temperature should be SAME as for thermostat
                                                     scalexy, #Set with flag for flexibility
                                                     scalexy,
                                                     True, #Only scale in z-direction
                                                     250 #Time-steps between MC moves
                                                    )
    )
  #If in bulk, have to use isotropic barostat to avoid any weird effects with box changing dimensions
  else:
    system.addForce(mm.MonteCarloBarostat(1.0*u.bar,
                                          temperature,
                                          250
                                         )
    )

  #Create new simulation object for NPT simulation
  sim = app.Simulation(top.topology, system, integrator, platform, prop, state)

  #Also create copies and simulation object for the NVE we will be running
  systemNVE = copy.deepcopy(systemRef)
  integratorNVE = mm.VerletIntegrator(2.0*u.femtoseconds)
  integratorNVE.setConstraintTolerance(1.0E-08)
  simNVE = app.Simulation(top.topology, systemNVE, integratorNVE, platform, prop)

  #Set the particle positions in the NPT simulation
  if pos is not None:
    sim.context.setPositions(pos)

  #Apply constraints before starting the simulation
  sim.context.applyConstraints(1.0E-08)

  #Check starting energy decomposition if want
  #decompEnergy(sim.system, sim.context.getState(getPositions=True))

  #Initialize velocities if not specified
  if vels is not None:
    sim.context.setVelocities(vels)
  else:
    try:
      testvel = sim.context.getState(getVelocities=True).getVelocities()
      print("Velocities included in state, starting with 1st particle: %s"%str(testvel[0]))
      #If all the velocities are zero, then set them to the temperature
      if not np.any(testvel.value_in_unit(u.nanometer/u.picosecond)):
        print("Had velocities, but they were all zero, so setting based on temperature.")
        sim.context.setVelocitiesToTemperature(temperature)
    except:
      print("Could not find velocities, setting with temperature")
      sim.context.setVelocitiesToTemperature(temperature)

  #Set up the reporter to output energies, volume, etc.
  sim.reporters.append(app.StateDataReporter(
                                             'prod_out.txt', #Where to write - can be stdout or file name (default .csv, I prefer .txt)
                                             500, #Number of steps between writes
                                             step=True, #Write step number
                                             time=True, #Write simulation time
                                             potentialEnergy=True, #Write potential energy
                                             kineticEnergy=True, #Write kinetic energy
                                             totalEnergy=True, #Write total energy
                                             temperature=True, #Write temperature
                                             volume=True, #Write volume
                                             density=False, #Write density
                                             speed=True, #Estimate of simulation speed
                                             separator='  ' #Default is comma, but can change if want (I like spaces)
                                            )
  )

  #Set up reporter for printing coordinates (trajectory)
  sim.reporters.append(NetCDFReporter(
                                      'prod.nc', #File name to write trajectory to
                                      500, #Number of steps between writes
                                      crds=True, #Write coordinates
                                      vels=True, #Write velocities
                                      frcs=False #Write forces
                                     )
  )

  #Identify solute indices and water oxygen indices (only use solute heavy atoms, no hydrogens)
  soluteInds = []
  owInds = []
  hw1Inds = []
  hw2Inds = []
  for res in top.residues:
    if res.name not in ['OTM', 'CTM', 'STM', 'NTM', 'SOL']:
      for atom in res.atoms:
        if 'H' not in atom.name[0]:
          soluteInds.append(atom.idx)
    elif res.name == 'SOL':
      for atom in res.atoms:
        if atom.name == 'OW':
          owInds.append(atom.idx)
        elif atom.name == 'HW1':
          hw1Inds.append(atom.idx)
        elif atom.name == 'HW2':
          hw2Inds.append(atom.idx)

  print("Solute indices:")
  print(soluteInds)
  #print("Water oxygen indices:")
  #print(owInds)
  #print("Water hydrogen (1st) indices:")
  #print(hw1Inds)
  #print("Water hydrogen (2nd) indices:")
  #print(hw2Inds)

  #Define cutoffs for solute solvation shells
  solShell1Cut = 0.55 #nanometers from all solute atoms (including hydrogens)
  solShell2Cut = 0.85

  #Create array to store the dynamic information of interest every 0.2 ps (100 steps) for 50 ps
  calcSteps = 100
  calcTotSteps = 25000
  numWats = np.zeros((int(calcTotSteps/calcSteps)+1, 2)) #Number waters that started in shell that are in shell at later time
  dipCorrs = np.zeros((int(calcTotSteps/calcSteps)+1, 2)) #Dipole correlation in both solute shells

  #Start running dynamics
  print("\nRunning NPT simulation with interspersed NVE to find dynamics...")
  sim.context.setTime(0.0)

  stepChunk = 5000 #Run NVE for 50 ps to find dynamics every 10 ps
  countSteps = 0
  
  while countSteps < nSteps:
  
    countSteps += stepChunk
    sim.step(stepChunk)

    #Record the simulation state so can kick off the NVE simulation
    thisState = sim.context.getState(getPositions=True, getVelocities=True)

    #Get solute and water oxygen coordinates after wrapping around the solute
    coords = thisState.getPositions(asNumpy=True)
    boxDims = np.diagonal(thisState.getPeriodicBoxVectors(asNumpy=True))
    wrapCOM = np.average(coords[soluteInds], axis=0)
    coords = wl.reimage(coords, wrapCOM, boxDims) - wrapCOM
    solCoords = coords[soluteInds]
    owCoords = coords[owInds]
    hw1Coords = coords[hw1Inds]
    hw2Coords = coords[hw2Inds]

    #Also store reference solute coordinates - will need if in decoupled state
    refSolCoords = coords[soluteInds]

    #Figure out which waters are in the solute solvation shells
    shell1BoolMat = wl.nearneighbors(solCoords, owCoords, boxDims, 0.0, solShell1Cut)
    shell1Bool = np.array(np.sum(shell1BoolMat, axis=0), dtype=bool)
    shell2BoolMat = wl.nearneighbors(solCoords, owCoords, boxDims, solShell1Cut, solShell2Cut)
    shell2Bool = np.array(np.sum(shell2BoolMat, axis=0), dtype=bool)

    #Count number of waters in each shell (will need for averaging)
    thisCount1 = int(np.sum(shell1Bool))
    thisCount2 = int(np.sum(shell2Bool))

    #print("Found %i waters in shell1"%thisCount1)
    #print("Found %i waters in shell2"%thisCount2)

    #Loop over waters in shells and compute dipole vectors as references
    refDipoles1 = np.zeros((thisCount1, 3))
    refDipoles2 = np.zeros((thisCount2, 3))
    for k, pos in enumerate(owCoords[shell1Bool]):
      thisOHvecs = wl.reimage([hw1Coords[shell1Bool][k], hw2Coords[shell1Bool][k]], pos, boxDims) - pos
      thisDip = -0.5*(thisOHvecs[0] + thisOHvecs[1])
      refDipoles1[k] = thisDip / np.linalg.norm(thisDip)
    for k, pos in enumerate(owCoords[shell2Bool]):
      thisOHvecs = wl.reimage([hw1Coords[shell2Bool][k], hw2Coords[shell2Bool][k]], pos, boxDims) - pos
      thisDip = -0.5*(thisOHvecs[0] + thisOHvecs[1])
      refDipoles2[k] = thisDip / np.linalg.norm(thisDip)

    #Set up the NVE simulation
    simNVE.context.setState(thisState)
    simNVE.context.setTime(0.0)

    #Loop over taking steps to computed dynamics
    countStepsNVE = 0
    while countStepsNVE <= calcTotSteps:
      calcState = simNVE.context.getState(getPositions=True)
      #Get solute and water oxygen coordinates after wrapping around the solute
      coords = calcState.getPositions(asNumpy=True)
      #If solute is in the coupled state, update the solvation shell every pass
      #If it's decoupled, just use the original solute coordinates to look at waters
      #Prevents artifically high rates of waters leaving due to solute moving on its own as gas
      #Not EXACTLY the same as in coupled state, as solute volume doesn't fluctuate
      #However, we will still average over representative solute configurations and volumes
      if not coupled:
        wrapCOM = np.average(refSolCoords, axis=0)
        coords = wl.reimage(coords, wrapCOM, boxDims) - wrapCOM
        solCoords = refSolCoords
      else:
        wrapCOM = np.average(coords[soluteInds], axis=0)
        coords = wl.reimage(coords, wrapCOM, boxDims) - wrapCOM
        solCoords = coords[soluteInds]
      owCoords = coords[owInds]
      hw1Coords = coords[hw1Inds]
      hw2Coords = coords[hw2Inds]
      #Count waters that started in each shell that are now in the shell at this time
      #No absorbing boundaries
      thisbool1Mat = wl.nearneighbors(solCoords, owCoords, boxDims, 0.0, solShell1Cut)
      thisbool1 = np.array(np.sum(thisbool1Mat, axis=0), dtype=bool)
      thisbool2Mat = wl.nearneighbors(solCoords, owCoords, boxDims, solShell1Cut, solShell2Cut)
      thisbool2 = np.array(np.sum(thisbool2Mat, axis=0), dtype=bool)
      numWats[int(countStepsNVE/calcSteps),0] += int(np.sum(thisbool1*shell1Bool))
      numWats[int(countStepsNVE/calcSteps),1] += int(np.sum(thisbool2*shell2Bool))
      #Loop over waters in shells and compute dipole vectors for this configuration
      #Adding to sum that we will normalize to find average at each time point
      for k, pos in enumerate(owCoords[shell1Bool]):
        thisOHvecs = wl.reimage([hw1Coords[shell1Bool][k], hw2Coords[shell1Bool][k]], pos, boxDims) - pos
        thisDip = -0.5*(thisOHvecs[0] + thisOHvecs[1])
        thisDip /= np.linalg.norm(thisDip)
        dipCorrs[int(countStepsNVE/calcSteps),0] += (np.dot(thisDip, refDipoles1[k]) / float(thisCount1))
      for k, pos in enumerate(owCoords[shell2Bool]):
        thisOHvecs = wl.reimage([hw1Coords[shell2Bool][k], hw2Coords[shell2Bool][k]], pos, boxDims) - pos
        thisDip = -0.5*(thisOHvecs[0] + thisOHvecs[1])
        thisDip /= np.linalg.norm(thisDip)
        dipCorrs[int(countStepsNVE/calcSteps),1] += (np.dot(thisDip, refDipoles2[k]) / float(thisCount2))
      simNVE.step(calcSteps)
      countStepsNVE += calcSteps

  #Finish normalizing dipole correlations (really cosine of angle between dipole vector at different times)
  numWats /= float(int(nSteps/stepChunk))
  dipCorrs /= float(int(nSteps/stepChunk))
  print("Normalizing factor for finding averages: %f"%float(int(nSteps/stepChunk)))

  #And save the final state of the NPT simulation in case we want to extend it
  sim.saveState('nptDynamicsState.xml')

  #And return the dipole correlations and times at which they were computed
  timeVals = 0.002*np.arange(0.0, calcTotSteps+0.0001, calcSteps)

  return numWats, dipCorrs, timeVals


def main(args):
  #Get the structure and topology files from the command line
  #ParmEd accepts a wide range of file types (Amber, GROMACS, CHARMM, OpenMM... but not LAMMPS) 
  try:
    topFile = args[0]
    strucFile = args[1]
  except IndexError:
    print("Specify topology and structure files from the command line.")
    sys.exit(2)
  
  print("Using topology file: %s" % topFile)
  print("Using structure file: %s" % strucFile)
  
  print("\nSetting up system...")
  
  #Load in the files for initial simulations
  top = pmd.load_file(topFile)
  struc = pmd.load_file(strucFile)
  
  #Transfer unit cell information to topology object
  top.box = struc.box[:]

  #Set up some global features to use in all simulations
  temperature = 298.15*u.kelvin
  
  #Define the platform (i.e. hardware and drivers) to use for running the simulation
  #This can be CUDA, OpenCL, CPU, or Reference 
  #CUDA is for NVIDIA GPUs
  #OpenCL is for CPUs or GPUs, but must be used for old CPUs (not SSE4.1 compatible)
  #CPU only allows single precision (CUDA and OpenCL allow single, mixed, or double)
  #Reference is a clear, stable reference for other code development and is very slow, using double precision by default
  platform = mm.Platform.getPlatformByName('CUDA')
  prop = {#'Threads': '2', #number of threads for CPU - all definitions must be strings (I think)
          'Precision': 'mixed', #for CUDA or OpenCL, select the precision (single, mixed, or double)
          'DeviceIndex': '0', #selects which GPUs to use - set this to zero if using CUDA_VISIBLE_DEVICES
          'DeterministicForces': 'True' #Makes sure forces with CUDA and PME are deterministic
         }
  
  #Create the OpenMM system that can be used as a reference
  systemRef = top.createSystem(
                               nonbondedMethod=app.PME, #Uses PME for long-range electrostatics, simple cut-off for LJ
                               nonbondedCutoff=12.0*u.angstroms, #Defines cut-off for non-bonded interactions
                               rigidWater=True, #Use rigid water molecules
                               constraints=app.HBonds, #Constrains all bonds involving hydrogens
                               flexibleConstraints=False, #Whether to include energies for constrained DOFs
                               removeCMMotion=True, #Whether or not to remove COM motion (don't want to if part of system frozen)
  )

  #Set up the integrator to use as a reference
  integratorRef = mm.LangevinIntegrator(
                                        temperature, #Temperature for Langevin
                                        1.0/u.picoseconds, #Friction coefficient
                                        2.0*u.femtoseconds, #Integration timestep
  )
  integratorRef.setConstraintTolerance(1.0E-08)

  #Get solute atoms 
  soluteIndices = []
  for res in top.residues:
    if res.name not in ['OTM', 'CTM', 'STM', 'NTM', 'SOL']:
      for atom in res.atoms:
        soluteIndices.append(atom.idx)

  print("\nSolute indices: %s" % str(soluteIndices))

  #Setting up the alchemical system so we can repeat the calculation with a decoupled particle
  #We need to add a custom non-bonded force for the solute being alchemically changed
  #Will be helpful to have handle on non-bonded force handling LJ and coulombic interactions
  NBForce = None
  for frc in systemRef.getForces():
    if (isinstance(frc, mm.NonbondedForce)):
      NBForce = frc

  #Turn off dispersion correction since have interface
  NBForce.setUseDispersionCorrection(False)

  forceLabelsRef = getForceLabels(systemRef)

  decompEnergy(systemRef, struc.positions, labels=forceLabelsRef)

  #Separate out alchemical and regular particles using set objects
  alchemicalParticles = set(soluteIndices)
  chemicalParticles = set(range(systemRef.getNumParticles())) - alchemicalParticles

  #Define the soft-core function for turning on/off LJ interactions
  #In energy expressions for CustomNonbondedForce, r is a special variable and refers to the distance between particles
  #All other variables must be defined somewhere in the function.
  #The exception are variables like sigma1 and sigma2.
  #It is understood that a parameter will be added called 'sigma' and that the '1' and '2' are to specify the combining rule.
  softCoreFunction = '4.0*lambdaLJ*epsilon*x*(x-1.0); x = (1.0/reff_sterics);'
  softCoreFunction += 'reff_sterics = (0.5*(1.0-lambdaLJ) + ((r/sigma)^6));'
  softCoreFunction += 'sigma=0.5*(sigma1+sigma2); epsilon = sqrt(epsilon1*epsilon2)'
  #Define the system force for this function and its parameters
  SoftCoreForce = mm.CustomNonbondedForce(softCoreFunction)
  SoftCoreForce.addGlobalParameter('lambdaLJ', 1.0) #Throughout, should follow convention that lambdaLJ=1.0 is fully-interacting state
  SoftCoreForce.addPerParticleParameter('sigma')
  SoftCoreForce.addPerParticleParameter('epsilon')

  #Will turn off electrostatics completely in the original non-bonded force
  #In the end-state, only want electrostatics inside the alchemical molecule
  #To do this, just turn ON a custom force as we turn OFF electrostatics in the original force
  ONE_4PI_EPS0 = 138.935456 #in kJ/mol nm/e^2
  soluteCoulFunction = '(1.0-(lambdaQ^2))*ONE_4PI_EPS0*charge/r;'
  soluteCoulFunction += 'ONE_4PI_EPS0 = %.16e;' % (ONE_4PI_EPS0)
  soluteCoulFunction += 'charge = charge1*charge2'
  SoluteCoulForce = mm.CustomNonbondedForce(soluteCoulFunction)
  #Note this lambdaQ will be different than for soft core (it's also named differently, which is CRITICAL)
  #This lambdaQ corresponds to the lambda that scales the charges to zero
  #To turn on this custom force at the same rate, need to multiply by (1.0-lambdaQ**2), which we do
  SoluteCoulForce.addGlobalParameter('lambdaQ', 1.0) 
  SoluteCoulForce.addPerParticleParameter('charge')

  #Also create custom force for intramolecular alchemical LJ interactions
  #Could include with electrostatics, but nice to break up
  #We could also do this with a separate NonbondedForce object, but it would be a little more work, actually
  soluteLJFunction = '4.0*epsilon*x*(x-1.0); x = (sigma/r)^6;'
  soluteLJFunction += 'sigma=0.5*(sigma1+sigma2); epsilon=sqrt(epsilon1*epsilon2)'
  SoluteLJForce = mm.CustomNonbondedForce(soluteLJFunction)
  SoluteLJForce.addPerParticleParameter('sigma')
  SoluteLJForce.addPerParticleParameter('epsilon')
  
  #Loop over all particles and add to custom forces
  #As we go, will also collect full charges on the solute particles
  #AND we will set up the solute-solute interaction forces
  alchemicalCharges = [[0]]*len(soluteIndices)
  for ind in range(systemRef.getNumParticles()):
    #Get current parameters in non-bonded force
    [charge, sigma, epsilon] = NBForce.getParticleParameters(ind)
    #Make sure that sigma is not set to zero! Fine for some ways of writing LJ energy, but NOT OK for soft-core!
    if sigma/u.nanometer == 0.0:
      newsigma = 0.3*u.nanometer #This 0.3 is what's used by GROMACS as a default value for sc-sigma
    else:
      newsigma = sigma
    #Add the particle to the soft-core force (do for ALL particles)
    SoftCoreForce.addParticle([newsigma, epsilon])
    #Also add the particle to the solute only forces
    SoluteCoulForce.addParticle([charge])
    SoluteLJForce.addParticle([sigma, epsilon])
    #If the particle is in the alchemical molecule, need to set it's LJ interactions to zero in original force
    if ind in soluteIndices:
      NBForce.setParticleParameters(ind, charge, sigma, epsilon*0.0)
      #And keep track of full charge so we can scale it right by lambda
      alchemicalCharges[soluteIndices.index(ind)] = charge

  #Now we need to handle exceptions carefully
  for ind in range(NBForce.getNumExceptions()):
    [p1, p2, excCharge, excSig, excEps] = NBForce.getExceptionParameters(ind)
    #For consistency, must add exclusions where we have exceptions for custom forces
    SoftCoreForce.addExclusion(p1, p2)
    SoluteCoulForce.addExclusion(p1, p2)
    SoluteLJForce.addExclusion(p1, p2)

  #Only compute interactions between the alchemical and other particles for the soft-core force
  SoftCoreForce.addInteractionGroup(alchemicalParticles, chemicalParticles)

  #And only compute alchemical/alchemical interactions for other custom forces
  SoluteCoulForce.addInteractionGroup(alchemicalParticles, alchemicalParticles)
  SoluteLJForce.addInteractionGroup(alchemicalParticles, alchemicalParticles)

  #Set other soft-core parameters as needed
  SoftCoreForce.setCutoffDistance(12.0*u.angstroms)
  SoftCoreForce.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
  SoftCoreForce.setUseLongRangeCorrection(False) 
  systemRef.addForce(SoftCoreForce)

  #Set other parameters as needed - note that for the solute force would like to set no cutoff
  #However, OpenMM won't allow a bunch of potentials with cutoffs then one without...
  #So as long as the solute is smaller than the cut-off, won't have any problems!
  SoluteCoulForce.setCutoffDistance(12.0*u.angstroms)
  SoluteCoulForce.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
  SoluteCoulForce.setUseLongRangeCorrection(False) 
  systemRef.addForce(SoluteCoulForce)

  SoluteLJForce.setCutoffDistance(12.0*u.angstroms)
  SoluteLJForce.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
  SoluteLJForce.setUseLongRangeCorrection(False) 
  systemRef.addForce(SoluteLJForce)

  #First do simulation with fully coupled state
  SoftCoreForce.setGlobalParameterDefaultValue(0, 1.0)
  SoluteCoulForce.setGlobalParameterDefaultValue(0, 1.0)

  for k, ind in enumerate(soluteIndices):
    [charge, sig, eps] = NBForce.getParticleParameters(ind)
    NBForce.setParticleParameters(ind, alchemicalCharges[k]*1.0, sig, eps)

  forceLabelsRef = getForceLabels(systemRef)
  decompEnergy(systemRef, struc.positions, labels=forceLabelsRef)

  os.mkdir('coupled')
  os.chdir('coupled')

  #Do NVT simulation
  stateFileNVT, stateNVT = doSimNVT(top, systemRef, integratorRef, platform, prop, temperature, pos=struc.positions)

  #And do NPT simulation using state information from NVT
  stateFileNPT, stateNPT = doSimNPT(top, systemRef, integratorRef, platform, prop, temperature, inBulk=True, state=stateFileNVT)

  #Now perform dynamics simulation to get dynamics - this is defined here, NOT in openmm_surface_affinities_lib.py
  numShellWaters, dipoleCosAng, timePoints = doSimDynamics(top, systemRef, integratorRef, platform, prop, temperature, inBulk=True, coupled=True, state=stateFileNPT)

  #Finally, want to now save the water residency over time and then also fit to exponential decay
  np.savetxt("shell_watCounts_coupled.txt", np.hstack((np.array([timePoints]).T, numShellWaters)),
             header="Time (ps)  Number waters in the 1st and 2nd solvation shells")

  opt1, pcov1 = optimize.curve_fit(normalExponential, timePoints, numShellWaters[:,0]/numShellWaters[0,0])
  decayTime1 = opt1[1]
  opt2, pcov2 = optimize.curve_fit(normalExponential, timePoints, numShellWaters[:,1]/numShellWaters[0,1])
  decayTime2 = opt2[1]

  print("\nIn the fully coupled ensemble:")
  print("\tWater residency correlation time for 1st shell waters: %f"%decayTime1)
  print("\tWater residency correlation time for 2nd shell waters: %f"%decayTime2)

  #Finally, want to now save the dipoles over time and then also fit to stretched exponential
  np.savetxt("rotational_timeCorr_coupled.txt", np.hstack((np.array([timePoints]).T, dipoleCosAng)),
             header="Time (ps)  Cos(angle) between starting dipole and dipole for 1st and 2nd solvation shells")

  opt1, pcov1 = optimize.curve_fit(stretchedExponential, timePoints, dipoleCosAng[:,0])
  decayTime1 = opt1[1]
  opt2, pcov2 = optimize.curve_fit(stretchedExponential, timePoints, dipoleCosAng[:,1])
  decayTime2 = opt2[1]

  print("\tRotational correlation time for 1st shell waters: %f"%decayTime1)
  print("\tRotational correlation time for 2nd shell waters: %f"%decayTime2)

  os.chdir('../')

  #Next simulate with decoupled state, but do same analysis
  #At least this way the volumes considered will be similar
  SoftCoreForce.setGlobalParameterDefaultValue(0, 0.0)
  SoluteCoulForce.setGlobalParameterDefaultValue(0, 0.0)

  for k, ind in enumerate(soluteIndices):
    [charge, sig, eps] = NBForce.getParticleParameters(ind)
    NBForce.setParticleParameters(ind, alchemicalCharges[k]*0.0, sig, eps)

  forceLabelsRef = getForceLabels(systemRef)
  decompEnergy(systemRef, struc.positions, labels=forceLabelsRef)

  os.mkdir('decoupled')
  os.chdir('decoupled')

  #Do NVT simulation
  stateFileNVT, stateNVT = doSimNVT(top, systemRef, integratorRef, platform, prop, temperature, pos=struc.positions)

  #And do NPT simulation using state information from NVT
  stateFileNPT, stateNPT = doSimNPT(top, systemRef, integratorRef, platform, prop, temperature, inBulk=True, state=stateFileNVT)

  #Now perform dynamics simulation to get dynamics - this is defined here, NOT in openmm_surface_affinities_lib.py
  numShellWaters, dipoleCosAng, timePoints = doSimDynamics(top, systemRef, integratorRef, platform, prop, temperature, inBulk=True, coupled=False, state=stateFileNPT)

  #Finally, want to now save the water residency over time and then also fit to exponential decay
  np.savetxt("shell_watCounts_decoupled.txt", np.hstack((np.array([timePoints]).T, numShellWaters)),
             header="Time (ps)  Number waters in the 1st and 2nd solvation shells")

  opt1, pcov1 = optimize.curve_fit(normalExponential, timePoints, numShellWaters[:,0]/numShellWaters[0,0])
  decayTime1 = opt1[1]
  opt2, pcov2 = optimize.curve_fit(normalExponential, timePoints, numShellWaters[:,1]/numShellWaters[0,1])
  decayTime2 = opt2[1]

  print("\nIn the perfectly decoupled ensemble:")
  print("\tWater residency correlation time for 1st shell waters: %f"%decayTime1)
  print("\tWater residency correlation time for 2nd shell waters: %f"%decayTime2)

  #Finally, want to now save the dipoles over time and then also fit to stretched exponential
  np.savetxt("rotational_timeCorr_decoupled.txt", np.hstack((np.array([timePoints]).T, dipoleCosAng)),
             header="Time (ps)  Cos(angle) between starting dipole and dipole for 1st and 2nd solvation shells")

  opt1, pcov1 = optimize.curve_fit(stretchedExponential, timePoints, dipoleCosAng[:,0])
  decayTime1 = opt1[1]
  opt2, pcov2 = optimize.curve_fit(stretchedExponential, timePoints, dipoleCosAng[:,1])
  decayTime2 = opt2[1]

  print("\tRotational correlation time for 1st shell waters: %f"%decayTime1)
  print("\tRotational correlation time for 2nd shell waters: %f"%decayTime2)

  os.chdir('../')


if __name__ == "__main__":
  main(sys.argv[1:])

