#!/bin/bash

#SBATCH -N 1                    # Number of nodes
#SBATCH -c 1                  # Number of cores
#SBATCH --mem=24GB              # Amount of RAM
#SBATCH -p htc                  # Partition
#SBATCH -G a100:1
#SBATCH -q public               # Quality of Service (QoS)
#SBATCH -o slurm.%j.out         # File to save STDOUT (%j = JobID)
#SBATCH -e slurm.%j.err         # File to save STDERR (%j = JobID)
#SBATCH --mail-type=ALL         # Send email on job start, end, and fail
#SBATCH --export=NONE           # Purge the job-submitting shell environment
#SBATCH -t 0-04:00:00

# Load necessary modules or software
module load mamba/latest

# Activate your Python environment
source activate PytorchTransformer

python train.py
python inference_3_cases.py
