#!/usr/bin/env python

import sys, os
import shutil
import subprocess
import glob
import copy
import numpy as np
import parmed as pmd
import waterlib as wl

#Given topology and structure files for a surface and a solute, places the solute just above the surface
#Will use similar procedure later to identify appropriate umbrellas

#Read in template structure and topology files
surfTop = sys.argv[1]
surfGro = sys.argv[2]
solTop = sys.argv[3]
solGro = sys.argv[4]

#Load files in with parmed
surf = pmd.load_file(surfTop, xyz=surfGro)
sol = pmd.load_file(solTop, xyz=solGro)

#Get box dimensions as defined by surface
boxdims = surf.box[:3]

#Define z-distances from surface CENTER to place solute (and umbrella centers)
#Will assume surface is all in one piece, i.e. not wrapping around box in z-dimension
surfCoords = np.array(surf.coordinates)
surfMaxZ = np.max(surfCoords[:,2])
surfAvgZ = np.average(surfCoords[:,2])
#minDist = np.ceil(surfMaxZ - surfAvgZ) + 1.0

#Better to place solvent in solution because takes up more space there
#AND easier to pull towards surface if it likes the surface (unlikely to dislike surface more than water...)
maxDist = np.floor(0.5*boxdims[2])
if surfAvgZ+maxDist >= boxdims[2]:
  maxDist = boxdims[2] - surfAvgZ #Not general... just make sure you put the surface in the center of the box!

#Want to place z-component of solute COM at maxDist from surface
solCoords = np.array(sol.coordinates)
solCoords -= np.average(solCoords, axis=0)
solCoords += np.array([0.5*boxdims[0], 0.5*boxdims[1], surfAvgZ+maxDist])
sol.coordinates = solCoords

thisConf = surf + sol

#Save structure (.gro) file 
thisConf.save('sol_surf_init.gro')

#Can't save topology because parmed will mangle the output
#Instead, manually read in both topologies and merge them together
#This should be as easy as taking last line of solute file and adding it to end of surface
#This works because SHOULD have generic topology header that loads all possible parameters for
#the surface, solute, and solvent, so just need to change the [ molecules ] header.
with open(surfTop, 'r') as infile:
  surfLines = infile.readlines()

with open(solTop, 'r') as infile:
  solLines = infile.readlines()

#Find last line that isn't blank for solvent
lastInd = 1
for i, aline in enumerate(solLines[::-1]):
  thisline = aline.strip()
  if len(thisline) != 0:
    lastInd = i+1
    break

sysLines = surfLines + [solLines[-lastInd]]

#And save the new topology
with open('sol_surf.top', 'w') as outfile:
  outfile.writelines(sysLines)

#Print the distance from the surface center that we placed the solute
#This way can capture this value from bash script, etc.
print maxDist


