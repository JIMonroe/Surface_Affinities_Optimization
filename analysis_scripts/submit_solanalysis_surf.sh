#!/bin/bash
#
#$ -V
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -N sol_structure

date

module load amber/amber16

python ~/Surface_Affinities_Project/all_code/analysis_scripts/solute_water_structure.py sol_surf.top False True

date

