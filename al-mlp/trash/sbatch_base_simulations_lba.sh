#!/bin/bash

# Default resources are 1 core with 2.8GB of memory per core.

# job name:
#SBATCH -J lba_base_sim

# priority
#SBATCH --account=bibs-frankmj-condo

# output file
#SBATCH --output /users/afengler/batch_job_out/lba_base_sim_%A_%a.out

# Request runtime, memory, cores:
#SBATCH --time=20:00:00
#SBATCH --mem=16G
#SBATCH -c 14
#SBATCH -N 1
#SBATCH --array=1-1

# Run a command
python -u /users/afengler/git_repos/nn_likelihoods/lba_base_simulations.py
