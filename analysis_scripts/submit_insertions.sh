#!/bin/bash
#
#$ -V
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -N HS_insert

date

module load amber/amber16

python ~/Surface_Affinities_Project/all_code/analysis_scripts/solute_insertions.py ../*.top prod.nc

date

