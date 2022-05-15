#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=c
#SBATCH --output=c.out

module purge
module load anaconda3/2020.07
eval #$(conda shell.bash hook)"
conda activate dl_env

srun python ./Model1.py