#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -l gpu=1
#$ -N energy_decomp

#Need to select GPUs to use
##############################################################################

#Get number of GPUs requested by job
numgpu=$(qstat -j $JOB_ID | grep "gpu" | sed 's/^.*=//' | sed 's/,.*//')

#Get availability of all GPUs as array
allusage=($(nvidia-smi -q -d UTILIZATION | grep 'Gpu' | awk '{print $3}'))

#Find the set of GPUs with the most load (fill out first)
set1=0
set1gpus=()
for i in {0..3}
do
  if [ ${allusage[$i]} -lt 3 ]; then
    set1=$(expr $set1 + 1)
    set1gpus+=($i)
  fi
done

set2=0
set2gpus=()
for i in {4..7}
do
  if [ ${allusage[$i]} -lt 3 ]; then
    set2=$(expr $set2 + 1)
    set2gpus+=($i)
  fi
done

if [ "$set1" -lt "$set2" ]; then
  if [ "$set1" -ge "$numgpu" ]; then
    printf -v touse "%s," "${set1gpus[@]:0:$numgpu}"
  elif [ "$set2" -ge "$numgpu" ]; then
    printf -v touse "%s," "${set2gpus[@]:0:$numgpu}"
  else touse='None'
  fi
else
  if [ "$set2" -ge "$numgpu" ]; then
    printf -v touse "%s," "${set2gpus[@]:0:$numgpu}"
  elif [ "$set1" -ge "$numgpu" ]; then
    printf -v touse "%s," "${set1gpus[@]:0:$numgpu}"
  else touse='None'
  fi
fi

if [ "$touse" == 'None' ]; then
  echo "Not enough GPUs... queue should not have let this through, hopefully fails."

else export CUDA_VISIBLE_DEVICES=$touse
fi

echo "Requested "$numgpu" GPUs"
echo "Using GPUs: "$touse

############################################################################

date

#export CUDA_VISIBLE_DEVICES=0

countdir=1

for dir in umbrella*

do

  cd $dir

  python ~/Surface_Affinities_Project/all_code/analysis_scripts/analyze_energies.py ../sol_surf.top ../solvated.gro prod.nc 2000 False
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

