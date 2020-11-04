#!/bin/bash

# Default resources are 1 core with 2.8GB of memory per core.

# job name:
#SBATCH -J bin_sim


# priority
#SBATCH --account=bibs-frankmj-condo

# output file
#SBATCH --output /users/afengler/batch_job_out/bin_sim_%A_%a.out

# Request runtime, memory, cores:
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH -c 14
#SBATCH -N 1
#SBATCH --array=1-100

# Run a command

python -u /users/afengler/git_repos/nn_likelihoods/bin_data.py $SLURM_ARRAY_TASK_ID
