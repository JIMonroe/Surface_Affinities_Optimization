#!/bin/bash 

#Get the output file prefix desired
PREFIX=${1}

echo ${PREFIX}

tleap -f tleap_input_${PREFIX}.in

#Switch to GROMACS topology files with parmed
python -c "
import parmed as pmd
solute = pmd.load_file('${PREFIX}.parm7', xyz='${PREFIX}.rst7')
solute.save('${PREFIX}.top', parameters='${PREFIX}.itp')
solute.save('${PREFIX}.gro')
"

