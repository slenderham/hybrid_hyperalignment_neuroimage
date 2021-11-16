#!/bin/bash

# Name of the job
#SBATCH --job-name=compmethods

# Number of compute nodes
#SBATCH --nodes=1

# Number of tasks per node
#SBATCH --ntasks-per-node=4

# Number of CPUs per task
#SBATCH --cpus-per-task=1

# Request memory
#SBATCH --mem=128G

# Walltime (job duration)
#SBATCH --time=01:00:00

# Email notifications (comma-separated options: BEGIN,END,FAIL)
#SBATCH --mail-type=FAIL

./all_h2a_benchmarks.sh
