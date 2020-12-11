#!/usr/bin/env python

from __future__ import division, print_function

import sys, os
import copy
import glob
import numpy as np
import simtk.openmm as mm
import simtk.openmm.app as app
import simtk.unit as u
import parmed as pmd
import pytraj as pt


Usage="""python analyze_energies_bulk.py topFile strucFile trajFile startFrame restraintBool

Determines solute-system potential energies decomposed into electrostatic
and LJ components (and WCA) for ANY system, bulk or interfacial (or even 
for boric acid, which is only different from the other solutes due to an 
intra-molecular potential energy term). This works because we are only 
changing the electrostatic and LJ interactions, meaining we don't need to 
compute the restraint potential, which will cancel out. This is accomplished 
by setting up an OpenMM system and providing it positions from a trajectory 
to evaluate the desired potential energy functions.

  Inputs:
    topFile - system topology file
    strucFile - system structure file, do define system with OpenMM
    trajFile - trajectory file name to loop over configurations
    startFrame - (default 0) starting frame to load from trajectory
    restraintBool - (default False) True/False for turning on/off restraints to surface
                    This is actually needed for the expanded ensemble simulations of the 
                    solute restrained to be at the interface. Though the restraint potential
                    energy is the same in both the coupled and decoupled states, the way
                    it gets weighted when computing free energies is different, so the 
                    restraint actually makes a contribution to the free energy difference.
  Outputs:
    pot_energy_decomp.txt - file containing coupled, no electrostatics, and
                            decoupled potential energies at each frame.
    Will also print differences, which give the solute-system interaction
    energies and can be recomputed from the output file. Note that this
    information is meaningless in the expanded ensemble, where proper
    reweighting is necessary to compute these averages.
"""


def printInfo(decompDat, mbarFile='mbar_object.pkl', alchFile=None):
  """Prints free energy decomposition information based.
     Need to provide the decomposed potential energies (coupled, no electrostatics,
     WCA, and fully decoupled as the 4 columns, either as numpy array or text file)
     and the name of a pickled MBAR object that can be used to correctly weight everything.
     To make the free energy decomposition more accurate for full free energy of solvation,
     the free energy to turn on LJ fully, and the free energy to turn on electrostatics with
     LJ already on, can provide a file (probably alchemical_U_noXYres.txt) that specifies
     the potential energy at each lambda state without XY restraints, which can be used to
     compute more accurate perturbed free energies thatn from what's computed from the
     energy decomposition calculated in the main section of this script.
  """
  import pickle

  #Check if decompDat is a string, if it is, load the file, otherwise, it should be a numpy array
  if isinstance(decompDat, basestring):
    decompDat = np.loadtxt(decompDat)

  #Load in the mbar object - don't bother catching error if file not found, it's pretty clear
  with open(mbarFile, 'r') as infile:
    mbarObj = pickle.load(infile)
  
  #The decomposition file should be ordered as fully coupled, no electrostatics, WCA, then decoupled
  #Get free energies between these states (less accurate/higher error than from original MBAR)
  dGdecomp, dGdecompErr = mbarObj.computePerturbedFreeEnergies(decompDat.T)
  print('Raw free energies (and uncertainties) between states computed in energy decomp file:')
  print(dGdecomp[0])
  print(dGdecompErr[0])
  
  dGwca = dGdecomp[0][2] - dGdecomp[0][-1]
  dGwcaErr = np.sqrt(dGdecompErr[0][2]**2 + dGdecompErr[0][-1]**2)
  dGljattr = dGdecomp[0][1] - dGdecomp[0][2]
  dGljattrErr = np.sqrt(dGdecompErr[0][1]**2 + dGdecompErr[0][2]**2)
  dGlj = dGdecomp[0][1] - dGdecomp[0][-1]
  dGljErr = np.sqrt(dGdecompErr[0][1]**2 + dGdecompErr[0][-1]**2)
  dGq = dGdecomp[0][0] - dGdecomp[0][1]
  dGqErr = np.sqrt(dGdecompErr[0][0]**2 + dGdecompErr[0][1]**2)
  dGtot = dGdecomp[0][0] - dGdecomp[0][-1]
  dGtotErr = np.sqrt(dGdecompErr[0][0]**2 + dGdecompErr[0][-1]**2)
  print('Free energy decompositions (less accurate than from check_dGsolv.py):')
  print('\tdG_WCA = %f +/- %f (dG to turn on repulsive part of LJ)'%(dGwca, dGwcaErr))
  print('\tdG_LJ_attr = %f +/- %f (dG to turn on attractive part of LJ with replusive on)'%(dGljattr, dGljattrErr))
  print('\tdG_LJ = %f +/- %f (dG to fully turn on LJ)'%(dGlj, dGljErr))
  print('\tdG_Q = %f +/- %f (dG to turn on electrostatics with LJ on)'%(dGq, dGqErr))
  print('\tdG_tot = %f +/- %f (dG to turn on fully coupled from decoupled)'%(dGtot, dGtotErr))
 
  #Next want to compute average potential energy differences en route to relative entropies
  #In what's below, it's probably more accurate to use the potential energies from alchDat wherever possible
  #(i.e. everywhere except for WCA and LJ attractive contributions)
  #But for that to work nicely, need to make bulk file the same as for surface file
  #Or set a flag
  avgUwca, avgUwcaErr = mbarObj.computeExpectations(decompDat[:,2] - decompDat[:,-1], decompDat[:,2])
  avgUljattr, avgUljattrErr = mbarObj.computeExpectations(decompDat[:,1] - decompDat[:,2], decompDat[:,1])
  avgUlj, avgUljErr = mbarObj.computeExpectations(decompDat[:,1] - decompDat[:,-1], decompDat[:,1])
  avgUq, avgUqErr = mbarObj.computeExpectations(decompDat[:,0] - decompDat[:,1], decompDat[:,0])
  avgUtot, avgUtotErr = mbarObj.computeExpectations(decompDat[:,0] - decompDat[:,-1], decompDat[:,0])
  print('Average potential energies in appropriate ensembles:')
  print('\t<U_WCA> = %f +/- %f (dU for repulsive part of LJ)'%(avgUwca, avgUwcaErr))
  print('\t<U_LJ_attr> = %f +/- %f (dU for repulsive to attractive LJ)'%(avgUljattr, avgUljattrErr))
  print('\t<U_LJ> = %f +/- %f (dU for full LJ compared to decoupled)'%(avgUlj, avgUljErr))
  print('\t<U_Q> = %f +/- %f (dU for fully coupled compared to no electrostatics)'%(avgUq, avgUqErr))
  print('\t<U_tot> = %f +/- %f (dU from fully decoupled to coupled)'%(avgUtot, avgUtotErr))
  
  #Finally, based on what we've just calculated, back out relative entropies for each step
  sRwca = dGwca - avgUwca
  sRwcaErr = np.sqrt(dGwcaErr**2 + avgUwcaErr**2)
  sRljattr = dGljattr - avgUljattr
  sRljattrErr = np.sqrt(dGljattrErr**2 + avgUljattrErr**2)
  sRlj = dGlj - avgUlj
  sRljErr = np.sqrt(dGljErr**2 + avgUljErr**2)
  sRq = dGq - avgUq
  sRqErr = np.sqrt(dGqErr**2 + avgUqErr**2)
  sRtot = dGtot - avgUtot
  sRtotErr = np.sqrt(dGtotErr**2 + avgUtotErr**2)
  print('Relative entropy decompositions:')
  print('\tSrel_WCA = %f +/- %f (relative entropy for decoupled to WCA)'%(sRwca, sRwcaErr))
  print('\tSrel_LJ_attr = %f +/- %f (relative entropy for WCA to full LJ)'%(sRljattr, sRljattrErr))
  print('\tSrel_LJ = %f +/- %f (relative entropy for decoupled to full LJ)'%(sRlj, sRljErr))
  print('\tSrel_Q = %f +/- %f (relative entropy for LJ only to fully coupled)'%(sRq, sRqErr))
  print('\tSrel_tot = %f +/- %f (relative entropy for decoupled to coupled)'%(sRtot, sRtotErr))

  #Read in the alchemical information file, if asked to
  if alchFile is not None:

    alchDat = np.loadtxt(alchFile)

    #And use it for more accurate information on some free energy differences  
    print('***Alchemical file %s was supplied! Displaying more accurate results below.***'%alchFile)
    dGacc, dGaccErr = mbarObj.computePerturbedFreeEnergies(alchDat.T)
    dGlj = dGacc[0][4] - dGacc[0][-1]
    dGljErr = np.sqrt(dGaccErr[0][4]**2 + dGaccErr[0][-1]**2)
    dGq = dGacc[0][0] - dGacc[0][4]
    dGqErr = np.sqrt(dGaccErr[0][0]**2 + dGaccErr[0][4]**2)
    dGtot = dGacc[0][0] - dGacc[0][-1]
    dGtotErr = np.sqrt(dGaccErr[0][0]**2 + dGaccErr[0][-1]**2) 
    print('Free energy decompositions (more accurate):')
    print('\tdG_LJ = %f +/- %f (dG to fully turn on LJ)'%(dGlj, dGljErr))
    print('\tdG_Q = %f +/- %f (dG to turn on electrostatics with LJ on)'%(dGq, dGqErr))
    print('\tdG_tot = %f +/- %f (dG to turn on fully coupled from decoupled)'%(dGtot, dGtotErr))
    avgUlj, avgUljErr = mbarObj.computeExpectations(alchDat[:,4] - alchDat[:,-1], alchDat[:,4])
    avgUq, avgUqErr = mbarObj.computeExpectations(alchDat[:,0] - alchDat[:,4], alchDat[:,0])
    avgUtot, avgUtotErr = mbarObj.computeExpectations(alchDat[:,0] - alchDat[:,-1], alchDat[:,0])
    print('Average potential energies in appropriate ensembles:')
    print('\t<U_LJ> = %f +/- %f (dU for full LJ compared to decoupled)'%(avgUlj, avgUljErr))
    print('\t<U_Q> = %f +/- %f (dU for fully coupled compared to no electrostatics)'%(avgUq, avgUqErr))
    print('\t<U_tot> = %f +/- %f (dU from fully decoupled to coupled)'%(avgUtot, avgUtotErr))
    sRlj = dGlj - avgUlj
    sRljErr = np.sqrt(dGljErr**2 + avgUljErr**2)
    sRq = dGq - avgUq
    sRqErr = np.sqrt(dGqErr**2 + avgUqErr**2)
    sRtot = dGtot - avgUtot
    sRtotErr = np.sqrt(dGtotErr**2 + avgUtotErr**2)
    print('Relative entropy decompositions:')
    print('\tSrel_LJ = %f +/- %f (relative entropy for decoupled to full LJ)'%(sRlj, sRljErr))
    print('\tSrel_Q = %f +/- %f (relative entropy for LJ only to fully coupled)'%(sRq, sRqErr))
    print('\tSrel_tot = %f +/- %f (relative entropy for decoupled to coupled)'%(sRtot, sRtotErr))


def main(args):
  #Get the structure, topology, and trajectory files from the command line
  #ParmEd accepts a wide range of file types (Amber, GROMACS, CHARMM, OpenMM... but not LAMMPS) 
  try:
    topFile = args[0]
    strucFile = args[1]
    trajFile = args[2]
  except IndexError:
    print("Specify topology, structure, and trajectory files from the command line.")
    print(Usage)
    sys.exit(2)

  #And also allow user to specify start frame, but default to zero if no input
  try:
    startFrame = int(args[3])
  except IndexError:
    startFrame = 0

  #And get information on whether or not to use a restraint
  try:
    boolstr = args[4]
    if boolstr.lower() == 'true' or boolstr.lower() == 'yes':
      restraintBool = True
    else:
      restraintBool = False
  except IndexError:
    restraintBool = False
  
  print("Using topology file: %s" % topFile)
  print("Using structure file: %s" % strucFile)
  print("Using trajectory file: %s" % trajFile)
  
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
  prop = {#'Threads': '1', #number of threads for CPU - all definitions must be strings (I think)
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

  #Get solute atoms and solute heavy atoms separately
  soluteIndices = []
  heavyIndices = []
  for res in top.residues:
    if res.name not in ['OTM', 'CTM', 'STM', 'NTM', 'SOL']:
      for atom in res.atoms:
        soluteIndices.append(atom.idx)
        if 'H' not in atom.name[0]:
          heavyIndices.append(atom.idx)

  #If working with expanded ensemble simulation of solute near the interface, need to include restraint to keep close
  if restraintBool:
    #Also get surface SU atoms and surface CU atoms at top and bottom of surface
    surfIndices = []
    for atom in top.atoms:
      if atom.type == 'SU':
        surfIndices.append(atom.idx)

    print("\nSolute indices: %s" % str(soluteIndices))
    print("Solute heavy atom indices: %s" % str(heavyIndices))
    print("Surface SU atom indices: %s" % str(surfIndices))

    #Will now add a custom bonded force between heavy atoms of each solute and surface SU atoms
    #Should be in units of kJ/mol*nm^2, but should check this
    #Also, note that here we are using a flat-bottom restraint to keep close to surface
    #AND to keep from penetrating into surface when it's in the decoupled state
    refZlo = 1.4*u.nanometer #in nm, the distance between the SU atoms and the solute centroid
    refZhi = 1.7*u.nanometer
    restraintExpression = '0.5*k*step(refZlo - (z2 - z1))*(((z2 - z1) - refZlo)^2)'
    restraintExpression += '+ 0.5*k*step((z2 - z1) - refZhi)*(((z2 - z1) - refZhi)^2)'
    restraintForce = mm.CustomCentroidBondForce(2, restraintExpression)
    restraintForce.addPerBondParameter('k')
    restraintForce.addPerBondParameter('refZlo') 
    restraintForce.addPerBondParameter('refZhi') 
    restraintForce.addGroup(surfIndices, np.ones(len(surfIndices))) #Don't weight with masses
    #To assign flat-bottom restraint correctly, need to know if each solute is above or below interface
    #Will need surface z-positions for this
    suZpos = np.average(struc.coordinates[surfIndices, 2])
    restraintForce.addGroup(heavyIndices, np.ones(len(heavyIndices)))
    solZpos = np.average(struc.coordinates[heavyIndices, 2])
    if (solZpos - suZpos) > 0:
      restraintForce.addBond([0, 1], [10000.0, refZlo, refZhi])
    else:
      #A little confusing, but have to negate and switch for when z2-z1 is always going to be negative
      restraintForce.addBond([0, 1], [10000.0, -refZhi, -refZlo])
    systemRef.addForce(restraintForce)

  #And define lambda states of interest
  lambdaVec = np.array(#electrostatic lambda - 1.0 is fully interacting, 0.0 is non-interacting
                       [[1.00, 0.75, 0.50, 0.25, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], 
                       #LJ lambdas - 1.0 is fully interacting, 0.0 is non-interacting
                        [1.00, 1.00, 1.00, 1.00, 1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.00] 
                       ])

  #We need to add a custom non-bonded force for the solute being alchemically changed
  #Will be helpful to have handle on non-bonded force handling LJ and coulombic interactions
  NBForce = None
  for frc in systemRef.getForces():
    if (isinstance(frc, mm.NonbondedForce)):
      NBForce = frc

  #Turn off dispersion correction since have interface
  NBForce.setUseDispersionCorrection(False)

  #Separate out alchemical and regular particles using set objects
  alchemicalParticles = set(soluteIndices)
  chemicalParticles = set(range(systemRef.getNumParticles())) - alchemicalParticles

  #Define the soft-core function for turning on/off LJ interactions
  #In energy expressions for CustomNonbondedForce, r is a special variable and refers to the distance between particles
  #All other variables must be defined somewhere in the function.
  #The exception are variables like sigma1 and sigma2.
  #It is understood that a parameter will be added called 'sigma' and that the '1' and '2' are to specify the combining rule.
  #Have also added parameter to switch the soft-core interaction to a WCA potential
  softCoreFunctionWCA = '(step(x-0.5))*(4.0*lambdaLJ*epsilon*x*(x-1.0) + (1.0-lambdaWCA)*lambdaLJ*epsilon) '
  softCoreFunctionWCA += '+ (1.0 - step(x-0.5))*lambdaWCA*(4.0*lambdaLJ*epsilon*x*(x-1.0));'
  softCoreFunctionWCA += 'x = (1.0/reff_sterics);'
  softCoreFunctionWCA += 'reff_sterics = (0.5*(1.0-lambdaLJ) + ((r/sigma)^6));'
  softCoreFunctionWCA += 'sigma=0.5*(sigma1+sigma2); epsilon = sqrt(epsilon1*epsilon2)'
  #Define the system force for this function and its parameters
  SoftCoreForceWCA = mm.CustomNonbondedForce(softCoreFunctionWCA)
  SoftCoreForceWCA.addGlobalParameter('lambdaLJ', 1.0) #Throughout, should follow convention that lambdaLJ=1.0 is fully-interacting state
  SoftCoreForceWCA.addGlobalParameter('lambdaWCA', 1.0) #When 1, attractions included; setting to 0 turns off attractions
  SoftCoreForceWCA.addPerParticleParameter('sigma')
  SoftCoreForceWCA.addPerParticleParameter('epsilon')

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
    SoftCoreForceWCA.addParticle([newsigma, epsilon])
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
    SoftCoreForceWCA.addExclusion(p1, p2)
    SoluteCoulForce.addExclusion(p1, p2)
    SoluteLJForce.addExclusion(p1, p2)

  #Only compute interactions between the alchemical and other particles for the soft-core force
  SoftCoreForceWCA.addInteractionGroup(alchemicalParticles, chemicalParticles)

  #And only compute alchemical/alchemical interactions for other custom forces
  SoluteCoulForce.addInteractionGroup(alchemicalParticles, alchemicalParticles)
  SoluteLJForce.addInteractionGroup(alchemicalParticles, alchemicalParticles)

  #Set other soft-core parameters as needed
  SoftCoreForceWCA.setCutoffDistance(12.0*u.angstroms)
  SoftCoreForceWCA.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
  SoftCoreForceWCA.setUseLongRangeCorrection(False) 
  systemRef.addForce(SoftCoreForceWCA)

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

  #Need to add integrator and context in order to evaluate potential energies
  #Integrator is arbitrary because won't use it
  integrator = mm.VerletIntegrator(1.0*u.femtoseconds)
  context = mm.Context(systemRef, integrator, platform, prop)

  ##########################################################################

  print("\nStarting analysis...")

  kBT = u.AVOGADRO_CONSTANT_NA * u.BOLTZMANN_CONSTANT_kB * temperature

  #Set up arrays to hold potential energies
  #First row will be coupled, then no electrostatics, then no electrostatics with WCA, then decoupled
  allU = np.array([[]]*4).T

  #We've now set everything up like we're going to run a simulation
  #But now we will use pytraj to load a trajectory to get coordinates
  #With those coordinates, we will evaluate the energies we want
  #Just need to figure out if we have a surface or a bulk system
  trajFiles = glob.glob('Quad*/%s'%trajFile)
  if len(trajFiles) == 0:
    trajFiles = [trajFile]

  print("Using following trajectory files: %s"%str(trajFiles))

  for aFile in trajFiles:

    trajtop = copy.deepcopy(top)
    trajtop.rb_torsions = pmd.TrackedList([]) #Necessary for SAM systems so doesn't break pytraj
    trajtop = pt.load_parmed(trajtop, traj=False)
    traj =  pt.iterload(aFile, top=trajtop, frame_slice=(startFrame, -1))

    thisAllU = np.zeros((len(traj), 4))

    #Loop over the lambda states of interest, looping over whole trajectory each time
    for i, lstate in enumerate([[1.0, 1.0, 1.0], #Fully coupled 
                                [1.0, 1.0, 0.0], #Charge turned off
                                [1.0, 0.0, 0.0], #Charged turned off, attractions turned off (so WCA)
                                [0.0, 1.0, 0.0]]): #Decoupled (WCA still included, though doesn't matter)

      #Set the lambda state
      context.setParameter('lambdaLJ', lstate[0])
      context.setParameter('lambdaWCA', lstate[1])
      context.setParameter('lambdaQ', lstate[2])
      for k, ind in enumerate(soluteIndices):
        [charge, sig, eps] = NBForce.getParticleParameters(ind)
        NBForce.setParticleParameters(ind, alchemicalCharges[k]*lstate[2], sig, eps)
      NBForce.updateParametersInContext(context)

      #And loop over trajectory
      for t, frame in enumerate(traj):

        thisbox = np.array(frame.box.values[:3])
        context.setPeriodicBoxVectors(np.array([thisbox[0], 0.0, 0.0]) * u.angstrom,
                                      np.array([0.0, thisbox[1], 0.0]) * u.angstrom,
                                      np.array([0.0, 0.0, thisbox[2]]) * u.angstrom )

        thispos = np.array(frame.xyz) * u.angstrom
        context.setPositions(thispos)

        thisAllU[t, i] = context.getState(getEnergy=True).getPotentialEnergy() / kBT
    
    #Add this trajectory information
    allU = np.vstack((allU, thisAllU))

  #And that should be it, just need to save files and print information
  #avgUq = np.average(allU[0,:] - allU[1,:])
  #stdUq = np.std(allU[0,:] - allU[1,:], ddof=1)
  #avgUlj = np.average(allU[1,:] - allU[2,:])
  #stdUlj = np.std(allU[1,:] - allU[2,:], ddof=1)

  #print("\nAverage solute-water electrostatic potential energy: %f +/- %f"%(avgUq, stdUq))
  #print("Average solute-water LJ potential energy: %f +/- %f"%(avgUlj, stdUlj))

  np.savetxt('pot_energy_decomp.txt', allU, 
             header='U_coupled (kBT)    U_noQ (kBT)    U_noQ_WCA (kBT)    U_decoupled (kBT) ')

  #Print some meaningful information
  #Just make sure we do so as accurately as possible using alchemical information
  alchFile = glob.glob('alchemical_U*.txt')[0]
  print('Using alchemical information file: %s'%alchFile)
  printInfo(allU, mbarFile='mbar_object.pkl', alchFile=alchFile)


if __name__ == "__main__":
  main(sys.argv[1:])

