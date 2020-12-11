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


#Given a topology and structure file, this script sets up an alchemical system and runs an expanded ensemble simulation


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
                               removeCMMotion=False, #Whether or not to remove COM motion (don't want to if part of system frozen)
  )

  #Set up the integrator to use as a reference
  integratorRef = mm.LangevinIntegrator(
                                        temperature, #Temperature for Langevin
                                        1.0/u.picoseconds, #Friction coefficient
                                        2.0*u.femtoseconds, #Integration timestep
  )
  integratorRef.setConstraintTolerance(1.0E-08)

  #To freeze atoms, set mass to zero (does not apply to virtual sites, termed "extra particles" in OpenMM)
  #Here assume (correctly, I think) that the topology indices for atoms correspond to those in the system
  for i, atom in enumerate(top.atoms):
    if atom.type in ('SU'): #, 'CU', 'CUO'):
      systemRef.setParticleMass(i, 0*u.dalton)

  #Get solute atoms and solute heavy atoms separately
  #Do this for as many solutes as we have so get list for each
  soluteIndices = []
  heavyIndices = []
  for res in top.residues:
    if res.name not in ['OTM', 'CTM', 'STM', 'NTM', 'SOL']:
      thissolinds = []
      thisheavyinds = []
      for atom in res.atoms:
        thissolinds.append(atom.idx)
        if 'H' not in atom.name[0]:
          thisheavyinds.append(atom.idx)
      soluteIndices.append(thissolinds)
      heavyIndices.append(thisheavyinds)

  #For convenience, also create flattened version of soluteIndices
  allSoluteIndices = []
  for inds in soluteIndices:
    allSoluteIndices += inds

  #Also get surface SU atoms and surface CU atoms at top and bottom of surface
  surfIndices = []
  for atom in top.atoms:
    if atom.type == 'SU':
      surfIndices.append(atom.idx)

  print("\nSolute indices: %s" % str(soluteIndices))
  print("(all together - %s)" % str(allSoluteIndices))
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
  for i in range(len(heavyIndices)):
    restraintForce.addGroup(heavyIndices[i], np.ones(len(heavyIndices[i])))
    solZpos = np.average(struc.coordinates[heavyIndices[i], 2])
    if (solZpos - suZpos) > 0:
      restraintForce.addBond([0, i+1], [10000.0, refZlo, refZhi])
    else:
      #A little confusing, but have to negate and switch for when z2-z1 is always going to be negative
      restraintForce.addBond([0, i+1], [10000.0, -refZhi, -refZlo])
  systemRef.addForce(restraintForce)

  #We also will need to force the solute to sample the entire surface
  #We will do this with flat-bottom restraints in x and y
  #But NOT on the centroid, because this is hard with CustomExternalForce
  #Instead, just apply it to the first heavy atom of each solute
  #Won't worry about keeping each solute in the same region of the surface on both faces...
  #Actually better if define regions on both faces differently
  initRefX = 0.25*top.box[0]*u.angstrom #Used parmed to load, and that program uses angstroms
  initRefY = 0.25*top.box[1]*u.angstrom
  refDistX = 0.25*top.box[0]*u.angstrom
  refDistY = 0.25*top.box[1]*u.angstrom
  print('Default X reference distance: %s'%str(initRefX))
  print('Default Y reference distance: %s'%str(initRefY))
  restraintExpressionXY = '0.5*k*step(periodicdistance(x,0,0,refX,0,0) - distX)*((periodicdistance(x,0,0,refX,0,0) - distX)^2)'
  restraintExpressionXY += '+ 0.5*k*step(periodicdistance(0,y,0,0,refY,0) - distY)*((periodicdistance(0,y,0,0,refY,0) - distY)^2)'
  restraintXYForce = mm.CustomExternalForce(restraintExpressionXY)
  restraintXYForce.addPerParticleParameter('k')
  restraintXYForce.addPerParticleParameter('distX')
  restraintXYForce.addPerParticleParameter('distY')
  restraintXYForce.addGlobalParameter('refX', initRefX)
  restraintXYForce.addGlobalParameter('refY', initRefY)
  for i in range(len(heavyIndices)):
    restraintXYForce.addParticle(heavyIndices[i][0], [1000.0, refDistX, refDistY]) #k in units kJ/mol*nm^2
  systemRef.addForce(restraintXYForce)

  #And define lambda states of interest
  lambdaVec = np.array(#electrostatic lambda - 1.0 is fully interacting, 0.0 is non-interacting
                       [[1.00, 0.75, 0.50, 0.25, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], 
                       #LJ lambdas - 1.0 is fully interacting, 0.0 is non-interacting
                        [1.00, 1.00, 1.00, 1.00, 1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.00] 
                       ])

  #JUST for boric acid, add a custom bonded force
  #Couldn't find a nice, compatible force field, but did find A forcefield, so using it
  #But has no angle terms on O-B-O and instead a weird bond repulsion term
  #This term also prevents out of plane bending
  #Simple in our case because boric acid is symmetric, so only need one parameter
  #Parameters come from Otkidach and Pletnev, 2001
  #Here, Ad = (A^2) / (d^6) since Ai and Aj and di and dj are all the same
  #In the original paper, B-OH bond had A = 1.72 and d = 0.354
  #Note that d is dimensionless and A should have units of (Angstrom^3)*(kcal/mol)^(1/2)
  #These units are inferred just to make things work out with kcal/mol and the given distance dependence
  bondRepulsionFunction = 'Ad*(1.0/r)^6'
  BondRepulsionForce = mm.CustomBondForce(bondRepulsionFunction)
  BondRepulsionForce.addPerBondParameter('Ad') #Units are technically kJ/mol * nm^6
  solOxInds = []
  for solInds in soluteIndices:
    baOxInds = []
    for aind in solInds:
      if top.atoms[aind].type == 'oh':
        baOxInds.append(aind)
    solOxInds.append(baOxInds)
  for baOxInds in solOxInds:
    for i in range(len(baOxInds)):
      for j in range(i+1, len(baOxInds)):
        BondRepulsionForce.addBond(baOxInds[i], baOxInds[j], [0.006289686]) 

  systemRef.addForce(BondRepulsionForce)

  #We need to add a custom non-bonded force for the solute being alchemically changed
  #Will be helpful to have handle on non-bonded force handling LJ and coulombic interactions
  NBForce = None
  for frc in systemRef.getForces():
    if (isinstance(frc, mm.NonbondedForce)):
      NBForce = frc

  #Turn off dispersion correction since have interface
  NBForce.setUseDispersionCorrection(False)

  forceLabelsRef = getForceLabels(systemRef)

  decompEnergy(systemRef, struc.positions, labels=forceLabelsRef, verbose=True)

  #Separate out alchemical and regular particles using set objects
  alchemicalParticles = set(allSoluteIndices)
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
  alchemicalCharges = [[0]]*len(allSoluteIndices)
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
    if ind in allSoluteIndices:
      NBForce.setParticleParameters(ind, charge, sigma, epsilon*0.0)
      #And keep track of full charge so we can scale it right by lambda
      alchemicalCharges[allSoluteIndices.index(ind)] = charge

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

  forceLabelsRef = getForceLabels(systemRef)

  decompEnergy(systemRef, struc.positions, labels=forceLabelsRef)

  #Run set of simulations for particle restrained in x and y to each quadrant of the surface
  #For best results, the configuration read in should have the solute right in the center of the surface
  #Will store the lambda biasing weights as we got to help speed convergence (hopefully)
  for xfrac in [0.25, 0.75]:
    for yfrac in [0.25, 0.75]:

      os.mkdir('Quad_%1.2fX_%1.2fY'%(xfrac, yfrac))
      os.chdir('Quad_%1.2fX_%1.2fY'%(xfrac, yfrac))

      restraintXYForce.setGlobalParameterDefaultValue(0, xfrac*top.box[0]*u.angstrom)
      restraintXYForce.setGlobalParameterDefaultValue(1, yfrac*top.box[1]*u.angstrom)

      decompEnergy(systemRef, struc.positions, labels=forceLabelsRef)

      #Do NVT simulation
      stateFileNVT, stateNVT = doSimNVT(top, systemRef, integratorRef, platform, prop, temperature, pos=struc.positions)

      #And do NPT simulation using state information from NVT
      stateFileNPT, stateNPT = doSimNPT(top, systemRef, integratorRef, platform, prop, temperature, state=stateFileNVT)

      #And do production run in expanded ensemble!
      stateFileProd, stateProd, weightsVec = doSimExpanded(top, systemRef, integratorRef, platform, prop, temperature, 0, lambdaVec, allSoluteIndices, alchemicalCharges, state=stateFileNPT)

      #Save final simulation configuration in .gro format (mainly for genetic algorithm)
      finalStruc = copy.deepcopy(struc)
      boxVecs = np.array(stateProd.getPeriodicBoxVectors().value_in_unit(u.angstrom)) # parmed works with angstroms
      finalStruc.box = np.array([boxVecs[0,0], boxVecs[1,1], boxVecs[2,2], 90.0, 90.0, 90.0])
      finalStruc.coordinates = np.array(stateProd.getPositions().value_in_unit(u.angstrom))
      finalStruc.save('prod.gro')

      os.chdir('../')


if __name__ == "__main__":
  main(sys.argv[1:])

