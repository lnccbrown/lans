#!/bin/bash

# Default resources are 1 core with 2.8GB of memory per core.

# job name:
#SBATCH -J echo_env

# priority
#SBATCH --account=bibs-frankmj-condo

# output file
#SBATCH --output /users/afengler/batch_job_out/echo_env.out

# Request runtime, memory, cores:
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH -c 12
#SBATCH -N 1
##SBATCH -p gpu --gres=gpu:1
#SBATCH --array=1-2

echo $my_fancy_var
echo 'test output'
