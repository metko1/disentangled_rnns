#!/bin/bash
# submit options
#SBATCH --job-name=test
##SBATCH --nodes=40
##SBATCH --ntasks-per-node=2
##SBATCH --ntasks=241
#SBATCH --mem=8G              # Limit memory to allow parallelism
#SBATCH --cpus-per-task=1     # Allocate 1 CPU per task
##SBATCH --time=1:00:00
#SBATCH -o output.%j
#SBATCH -e error.%j
##SBATCH --array=100,200,400,800,1600,3200,4300
#SBATCH --array=0-240
##40

#declare -ia array=(100 200 400 800 1600 3200 4300)
#source .venv/bin/activate
for i in $(seq 1 2); do
  echo "process $SLURM_PROCID (out of $SLURM_NPROCS total) on $(hostname -s) with index $SLURM_ARRAY_TASK_ID" >> results.txt
  #srun bash -c 'echo "process $SLURM_PROCID (out of $SLURM_NPROCS total) on $(hostname -s) with index $SLURM_ARRAY_TASK_ID" >> results.txt'
  #srun python3 disRNN_test.py ${SLURM_ARRAY_TASK_ID} &
done
wait

exit
