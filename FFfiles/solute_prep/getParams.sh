#!/bin/bash 

#Get the input file
INFILE=${1}

#Get the output file prefix desired
PREFIX=${2}

#Get the tleap input file (so that doesn't have to be in working directory)
#Will uses sed to adjust this file given the prefix
TLEAPIN=${3}

echo ${PREFIX}

#In below, assigns AMBER atom types, not GAFF - should be about the same, but can try both if want
antechamber -i ${INFILE} -fi mol2 -o ${PREFIX}.ac -fo ac -c bcc -at gaff2 -s 2

prepgen -i ${PREFIX}.ac -o ${PREFIX}.prepin 

parmchk2 -i ${PREFIX}.prepin -f prepi -o ${PREFIX}.frcmod 

#Generate AMBER topology files with tleap
sed -e "s/prefix/${PREFIX}/g" ${TLEAPIN} > ./tleap_input_${PREFIX}.in

tleap -f tleap_input_${PREFIX}.in

#Switch to GROMACS topology files with parmed
python -c "
import parmed as pmd
solute = pmd.load_file('${PREFIX}.parm7', xyz='${PREFIX}.rst7')
solute.save('${PREFIX}.top', parameters='${PREFIX}.itp')
solute.save('${PREFIX}.gro')
"

