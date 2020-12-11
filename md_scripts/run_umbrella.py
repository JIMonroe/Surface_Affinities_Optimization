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
import pytraj as pt


#Given a topology and structure file, this script sets up and runs umbrella-sampling simulations
#For the structure file given, the solute should be far from the surface so it can be pulled towards it
#This will generate starting points for the umbrella simulations


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

  #Track non-bonded force, mainly to turn off dispersion correction
  NBForce = None
  for frc in systemRef.getForces():
    if (isinstance(frc, mm.NonbondedForce)):
      NBForce = frc

  #Turn off dispersion correction since have interface
  NBForce.setUseDispersionCorrection(False)

  #Get solute atoms and solute heavy atoms separately
  soluteIndices = []
  heavyIndices = []
  for res in top.residues:
    if res.name not in ['OTM', 'CTM', 'STM', 'NTM', 'SOL']:
      for atom in res.atoms:
        soluteIndices.append(atom.idx)
        if 'H' not in atom.name[0]:
          heavyIndices.append(atom.idx)

  #Also get surface SU atoms
  surfIndices = []
  for atom in top.atoms:
    if atom.type == 'SU':
      surfIndices.append(atom.idx)

  startPos = np.array(struc.positions.value_in_unit(u.nanometer))

  print("\nSolute indices: %s" % str(soluteIndices))
  print("Solute heavy atom indices: %s" % str(heavyIndices))
  print("Surface SU atom indices: %s" % str(surfIndices))

  #Solute should already be placed far from the surface
  #If this is not done right, or it is too close to half the periodic box distance, will have issues
  #Either way, set this as the starting reference z distance
  initRefZ = np.average(startPos[heavyIndices, 2]) - np.average(startPos[surfIndices, 2])

  print(initRefZ)

  #Will now add a custom bonded force between solute heavy atoms and surface SU atoms
  #Should be in units of kJ/mol*nm^2, but should check this
  #For expanded ensemble, fine to assume solute is less than half the box distance from the surface
  #But for umbrella sampling, want to apply pulling towards surface regardless of whether pull from above or below
  #Also allows us to get further from the surface with our umbrellas without worrying about PBCs
  restraintExpression = '0.5*k*(abs(z2 - z1) - refZ)^2'
  restraintForce = mm.CustomCentroidBondForce(2, restraintExpression)
  restraintForce.addPerBondParameter('k')
  restraintForce.addGlobalParameter('refZ', initRefZ) #Make global so can modify during simulation
  restraintForce.addGroup(surfIndices, np.ones(len(surfIndices))) #Don't weight with masses
  restraintForce.addGroup(heavyIndices, np.ones(len(heavyIndices)))
  restraintForce.addBond([0, 1], [1000.0])
  restraintForce.setUsesPeriodicBoundaryConditions(True) #Only when doing umbrella sampling
  systemRef.addForce(restraintForce)

  forceLabelsRef = getForceLabels(systemRef)

  decompEnergy(systemRef, struc.positions, labels=forceLabelsRef, verbose=False)

  #Do NVT simulation
  stateFileNVT1, stateNVT1 = doSimNVT(top, systemRef, integratorRef, platform, prop, temperature, pos=struc.positions)

  #Do pulling simulation - really I'm just slowly changing the equilibrium bond distance between the surface and the solute
  stateFilePull, statePull = doSimPull(top, systemRef, integratorRef, platform, prop, temperature, state=stateFileNVT1)

  decompEnergy(systemRef, statePull, labels=forceLabelsRef, verbose=False)

  #Load in pulling restraint data and pulling trajectory to identify starting structures for each umbrella
  pullData = np.loadtxt('pull_restraint.txt')
  trajtop = copy.deepcopy(top)
  trajtop.rb_torsions = pmd.TrackedList([])
  trajtop = pt.load_parmed(trajtop, traj=False)
  pulltraj = pt.iterload('pull.nc', trajtop)
  frameTimes = np.array([frame.time for frame in pulltraj])

  #Define some umbrella distances based on a fixed spacing between initial and final pulling coordinate
  #Actually for final pulling coordinate, close to the surface, using average of actual coordinate and reference
  zSpace = 0.1 #nm
  zUmbs = np.arange(0.5*(pullData[-1,1] + pullData[-1,2]), pullData[0,2], zSpace)

  print("\nUsing following umbrellas:")
  print(zUmbs)

  #Then loop over umbrellas and run simulation for each
  for i, zRefDist in enumerate(zUmbs):
    
    os.mkdir("umbrella%i"%i)
    os.chdir("umbrella%i"%i)

    #Find where in the pulling trajectory the solute came closest to this umbrella
    pullDatInd = np.argmin(abs(pullData[:,2] - zRefDist))
    frameInd = np.argmin(abs(frameTimes - pullData[pullDatInd,0]))

    #Get starting coordinates
    #Making sure to assign the units that will be returned by pytraj
    thisCoords = np.array(pulltraj[frameInd].xyz) * u.angstrom

    #Set reference value for harmonic force
    restraintForce.setGlobalParameterDefaultValue(0, zRefDist)

    print("\nUmbrella %i:"%i)
    print("\tReference distance: %f"%zRefDist)
    print("\tFrame chosen from trajectory: %i (%f ps)"%(frameInd, frameTimes[frameInd]))
    
    #And run simulations, first equilibrating in NVT, then NPT, then production in NPT
    stateFileNVT, stateNVT = doSimNVT(top, systemRef, integratorRef, platform, prop, temperature, pos=thisCoords)

    stateFileNPT, stateNPT = doSimNPT(top, systemRef, integratorRef, platform, prop, temperature, state=stateFileNVT)
 
    stateFileProd, stateProd = doSimUmbrella(top, systemRef, integratorRef, platform, prop, temperature, state=stateFileNPT)

    os.chdir("../")



if __name__ == "__main__":
  main(sys.argv[1:])

