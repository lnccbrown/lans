#!/bin/bash

# Default resources are 1 core with 2.8GB of memory per core.

# job name:
#SBATCH -J p_exp_ddm

# priority
#SBATCH --account=bibs-frankmj-condo

# output file
#SBATCH --output /users/afengler/batch_job_out/pertubration_exp_%A_%a.out

# Request runtime, memory, cores:
#SBATCH --time=36:00:00
#SBATCH --mem=24G
#SBATCH -c 14
#SBATCH -N 1
##SBATCH -p gpu --gres=gpu:1
#SBATCH --array=1-100

# Run a command
#source /users/afengler/miniconda3/etc/profile.d/conda.sh
#conda activate tony
python -u /users/afengler/git_repos/nn_likelihoods/method_comparison_sim.py $SLURM_ARRAY_TASK_ID 500
