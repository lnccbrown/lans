#!/bin/bash

# Default resources are 1 core with 2.8GB of memory per core.

# job name:
#SBATCH -J full_ddm_mle

# output file
#SBATCH --output /users/afengler/batch_job_out/full_ddm_mle_%A_%a.out

# Request runtime, memory, cores:
#SBATCH --time=15:00:00
#SBATCH --mem=4G
#SBATCH -c 2
#SBATCH -N 1
#SBATCH --array=1-10

# Run a command
python -u /users/afengler/git_repos/nn_likelihoods/kde_mle_parallel.py
