#!/bin/bash

# Default resources are 1 core with 2.8GB of memory per core.

# job name:
#SBATCH -J ddm_a_a_4

# priority
##SBATCH --account=bibs-frankmj-condo
#SBATCH --account=carney-frankmj-condo
##SBATCH --account=bibs-frankmj-condo

# output file
#SBATCH --output /users/afengler/batch_job_out/mc_ddm_a_a_4_%A_%a.out

# Request runtime, memory, cores:
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH -c 1
#SBATCH -N 1
##SBATCH -p gpu --gres=gpu:1
#SBATCH --array=1-100

# Run a command
#source /users/afengler/miniconda3/etc/profile.d/conda.sh
#conda activate tony

source /users/afengler/.bashrc
conda deactivate
conda activate tf-cpu

# source /users/afengler/.bashrc
# conda deactivate
# conda activate tf-gpu-py37

# NNBATCH RUNS

nmcmcsamples=25000
nbyarrayjob=10
ncpus=1
nsamples=( 1024 4096 ) #( 1024 2048 4096 ) # 2048 4096 ) #( 1024 2048 4096 )
method="weibull_cdf2" #'ddm_sdv_analytic'   #'ddm_sdv_analytic'  #"full_ddm2"
modelidentifier='_100k'
ids=( -1 )
machine='ccv'
samplerinit='mle'
outfilesignature='elife_diffevo_'
infilesignature=None
analytic=0
sampler='diffevo'
#SLURM_ARRAY_TASK_ID=1

for n in "${nsamples[@]}"
do
    for id in "${ids[@]}"
    do 
        python -u method_comparison_sim.py --machine $machine \
                                                      --method $method \
                                                      --modelidentifier $modelidentifier \
                                                      --nsamples $n \
                                                      --nmcmcsamples $nmcmcsamples \
                                                      --datatype parameter_recovery \
                                                      --sampler $sampler \
                                                      --infilesignature $infilesignature  \
                                                      --outfileid $SLURM_ARRAY_TASK_ID \
                                                      --activedims 0 1 2 3 4 5 6 \
                                                      --samplerinit $samplerinit \
                                                      --ncpus $ncpus \
                                                      --nbyarrayjob $nbyarrayjob \
                                                      --nnbatchid $id \
                                                      --analytic $analytic \
                                                      --outfilesig $outfilesignature
    done
done

# --outfileid $SLURM_ARRAY_TASK_ID