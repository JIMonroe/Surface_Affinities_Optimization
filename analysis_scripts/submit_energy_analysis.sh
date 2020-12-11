#!/bin/bash
#
#$ -V
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -N energy_decomp

date

module load python/anaconda

python ~/Surface_Affinities_Project/all_code/analysis_scripts/analyze_energies.py *.top solvated.gro prod.nc 2000
#Last number is starting frame - adjust as needed

date

