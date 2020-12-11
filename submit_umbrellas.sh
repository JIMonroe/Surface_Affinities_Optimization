#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -l gpu=1
#$ -N umbRun

date

export CUDA_VISIBLE_DEVICES=0

source /usr/local/gromacs-2018.2/bin/GMXRC

#First we run add_solute.py
ffdir=/home/jmonroe/Surface_Affinities_Project/FFfiles
python /home/jmonroe/Surface_Affinities_Project/all_code/umbrella_scripts/add_solute.py ${ffdir}/fullCH3_6-8.top ${ffdir}/fullCH3_6-8.gro ${ffdir}/methane.top ${ffdir}/methane.gro

#Must copy over vdwradii.dat for solvation to work right
cp ${ffdir}/vdwradii.dat ./

#Next, solvate
gmx -quiet solvate -cp sol_surf_init.gro -cs tip4p.gro -o solvated.gro -p sol_surf.top -box 2.98200 3.44332 6.00000 -scale 0.55

#And run simulations with OpenMM!
python /home/jmonroe/Surface_Affinities_Project/all_code/md_scripts/run_umbrella.py sol_surf.top solvated.gro

#Compute perturbed potential energies (solute decoupled, no electrostatics, WCA solute, etc.)
for dir in umbrella*
do
  cd $dir
  python ~/Surface_Affinities_Project/all_code/analysis_scripts/analyze_energies.py ../sol_surf.top ../solvated.gro prod.nc 0 False
  #Last number is starting frame - adjust as needed
  cd ../
done

#Clean up trajectories since don't really need but take up a lot of space
rm umbrella*/*.nc

python /home/jmonroe/Surface_Affinities_Project/all_code/umbrella_scripts/pmf_calc.py umbrella 1000.0

date


