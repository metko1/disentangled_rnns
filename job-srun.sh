#!/bin/bash
# submit options
#SBATCH --job-name=disRNN-test
##SBATCH --nodes=25
##SBATCH --ntasks-per-node=2
#SBATCH --ntasks=50
#SBATCH --time=1:00:00
#SBATCH -o output.%j
#SBATCH -e error.%j
#SBATCH --array=100,200,400,800,1600,3200,4300

#declare -ia array=(100 200 400 800 1600 3200 4300)
source venv/bin/activate
for i in $(seq 1 2); do
  #srun bash -c 'echo "process $SLURM_PROCID \
  #(out of $SLURM_NPROCS total) on $(hostname -s)" >> results.txt'
  srun python3 disRNN_test.py ${SLURM_ARRAY_TASK_ID} &
  #for n in ${array[@]}; do
    #srun python3 disRNN_test.py $n &
  #done
done
wait

exit
