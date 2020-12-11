#!/bin/bash
#
#$ -V
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -N wat_structure

date

module load amber/amber16

python ~/Surface_Affinities_Project/all_code/analysis_scripts/surf_water_structure.py sol_surf.top Quad*

date

