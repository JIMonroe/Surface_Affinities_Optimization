#Script to just quickly use parmed to set the solute coordinates and box to what we want

import sys
import numpy as np
import parmed as pmd

rstfile = sys.argv[1]
zcoord = float(sys.argv[2]) #In ANGSTROMS for parmed

struc = pmd.load_file(rstfile)

#Really should use only heavy-atom center of geometry, but close enough to just use all-atoms
solcent = np.average(struc.coordinates, axis=0)

newcent = np.array([14.91, 17.22, zcoord])

newcoords = struc.coordinates - solcent + newcent

struc.coordinates = newcoords

struc.box = [29.8200, 34.4332, 60.0000, 90.0, 90.0, 90.0]

struc.save(rstfile, overwrite=True)


