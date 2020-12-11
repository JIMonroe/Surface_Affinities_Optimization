#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -l gpu=1
#$ -N dynamicsSurf

date

#Manually select GPU to use
export CUDA_VISIBLE_DEVICES=0

source /usr/local/gromacs-2018.2/bin/GMXRC

#For convenience, define the force-field file directory
ffdir=/home/jmonroe/Surface_Affinities_Project/FFfiles

#First need to add the solute to the surface
#To do this, just combine the .gro and .top files for the solute with the provided surface .gro and .top files
#For this to work, MUST modify the solute .gro file in the FFfiles directory so the solute is where you want it to be
#The workflow is then to run umbrella sampling, find a good z-coordinate, then manually set the .gro file in FFfiles to this
#Also, make sure that the only .gro and .top files in the working directory this is run from are for the bare surface!
solfile=${ffdir}/methane.gro
surffile=(*.gro)
surffile=${surffile[0]}
python -c "import copy
import numpy as np
import parmed as pmd
solute = pmd.load_file('${solfile}')
surf = pmd.load_file('${surffile}')
solZpos = np.average(solute.coordinates[:,2])
suZpos = np.average(surf['@SU'].coordinates[:,2])
solsurf = surf + solute
solsurf.save('sol_surf_init.gro')
"
cat *.top > sol_surf.top
tail -n 1 ${ffdir}/methane.top >> sol_surf.top

#Must copy over vdwradii.dat for solvation to work right
cp ${ffdir}/vdwradii.dat ./

#Next, solvate
gmx -quiet solvate -cp sol_surf_init.gro -cs tip4p.gro -o solvated.gro -p sol_surf.top -box 2.98200 3.44332 6.00000 -scale 0.55

#And run simulations with OpenMM!
python /home/jmonroe/Surface_Affinities_Project/all_code/analysis_scripts/get_dynamics_surf.py sol_surf.top solvated.gro

date


