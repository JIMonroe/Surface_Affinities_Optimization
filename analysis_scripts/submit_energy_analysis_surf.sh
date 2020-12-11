#!/bin/bash
#
#$ -V
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -N energy_decomp

date

module load python/anaconda

countdir=1

for dir in Quad*

do

  cd $dir

  python ~/Surface_Affinities_Project/all_code/analysis_scripts/analyze_energies.py ../sol_surf.top ../solvated.gro prod.nc 2000 True
  #Last number is starting frame - adjust as needed

  if [ $countdir -eq 1 ]
  then
    cp pot_energy_decomp.txt ../
  else
    tail -n +2 pot_energy_decomp.txt >> ../pot_energy_decomp.txt
  fi

  countdir=$((countdir+1))

  cd ../

done

date

