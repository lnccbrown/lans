#!/bin/bash

# Default resources are 1 core with 2.8GB of memory per core.

# job name:
#SBATCH -J timings

# priority
##SBATCH --account=bibs-frankmj-condo
#SBATCH --account=carney-frankmj-condo

# email error reports
#SBATCH --mail-user=alexander_fengler@brown.edu 
#SBATCH --mail-type=ALL

# output file
#SBATCH --output /users/afengler/batch_job_out/timings_%A.out

# Request runtime, memory, cores
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH -c 14
#SBATCH -N 1

#SBATCH --constraint='quadrortx'
##SBATCH --constraint='cascade'
#SBATCH -p gpu --gres=gpu:1
#SBATCH --array=1-1

source /users/afengler/.bashrc
conda deactivate
conda activate tf-gpu-py37
#conda activate tf-cpu

gpu=0

python -u /users/afengler/git_repos/nn_likelihoods/keras_timing.py --machine ccv --nreps 100 --method ddm --gpu $gpu


# python -u /users/afengler/OneDrive/git_repos/nn_likelihoods/keras_timing.py --machine ccv --nreps 100 --method ddm --gpu $gpu