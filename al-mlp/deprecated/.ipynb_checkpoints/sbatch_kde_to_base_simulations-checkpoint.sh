#!/bin/bash

# Default resources are 1 core with 2.8GB of memory per core.

# job name:
#SBATCH -J tpl_3_weibull

# priority
#SBATCH --account=bibs-frankmj-condo

# output file
#SBATCH --output /users/afengler/batch_job_out/tpl3_%A_%a.out

# Request runtime, memory, cores:
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH -c 14
#SBATCH -N 1
#SBATCH --array=28,60

# Run a command
method='ornstein'
python -u kde_train_test.py --machine ccv --method $method --simfolder training_data_binned_0_nbins_0_n_20000 --fileprefix ${method}_nchoices_2_train_data_binned_0_nbins_0_n_20000 --outfolder training_data_binned_0_nbins_0_n_20000 --nbyparam 1000 --mixture 0.8 0.1 0.1 --fileid $SLURM_ARRAY_TASK_ID

#--fileid $SLURM_ARRAY_TASK_ID
#python -u /users/afengler/git_repos/nn_likelihoods/navarro_fuss_train_test.py
