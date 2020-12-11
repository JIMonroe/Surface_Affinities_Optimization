#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -l gpu=1
#$ -N expRunBulk

date

export CUDA_VISIBLE_DEVICES=0

source /usr/local/gromacs-2018.2/bin/GMXRC

#For convenience, define the force-field file directory
ffdir=/home/jmonroe/Surface_Affinities_Project/FFfiles

#And copy over desired solute from this directory
cp ${ffdir}/methane.gro ./
cp ${ffdir}/methane.top ./

#First make sure the solute is centered in a box of appropriate size
gmx -quiet editconf -f methane.gro -o centered.gro -box 3.50000 3.50000 3.50000 -c

#Must copy over vdwradii.dat for solvation to work right
cp ${ffdir}/vdwradii.dat ./

#Next, solvate
gmx -quiet solvate -cp centered.gro -cs tip4p.gro -o solvated.gro -p methane.top -box 3.50000 3.50000 3.50000 -scale 0.55

#And run simulations with OpenMM!
python /home/jmonroe/Surface_Affinities_Project/all_code/md_scripts/run_expanded_bulk.py methane.top solvated.gro

date


