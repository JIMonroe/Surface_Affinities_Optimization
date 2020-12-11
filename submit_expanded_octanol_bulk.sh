#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -l gpu=1
#$ -N expRunBulk

date

export CUDA_VISIBLE_DEVICES=7

source /usr/local/gromacs-2018.2/bin/GMXRC

#For convenience, define the force-field file directory
ffdir=/home/jmonroe/Surface_Affinities_Project/FFfiles

#And copy over desired solute from this directory
cp ${ffdir}/boricacid.gro ./
cp ${ffdir}/boricacid.top ./

#First make sure the solute is centered in a box of appropriate size
gmx -quiet editconf -f boricacid.gro -o centered.gro -box 3.50000 3.50000 3.50000 -c

#Must copy over vdwradii.dat for solvation to work right
cp ${ffdir}/vdwradii.dat ./

#Next, solvate
gmx insert-molecules -f centered.gro -ci octanol.gro -o solvated.gro -box 3.50000 3.50000 3.50000 -nmol 1000 -scale 0.66 &>> solv_info.txt

#Manually add line for solvent in topology file!
#Not anymore...
NOCT=`grep 'Added ' solv_info.txt | awk '{print $2}'`
echo 'octanol           '$NOCT >> boricacid.top

#Add in water parameters for octanol 
sed -i '/\[ system/i \
; Octanol as a solvent as well \
#include "octanol.itp"
     '  boricacid.top

#And run simulations with OpenMM!
python run_expanded_octanol.py boricacid.top solvated.gro
#python test_octanol_bulk.py octanol_box.top solvated.gro


date


