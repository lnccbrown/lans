#!/bin/bash

# Default resources are 1 core with 2.8GB of memory per core.

# job name:
#SBATCH -J mcmc_data_handler

# priority
#SBATCH --account=carney-frankmj-condo

# output file
#SBATCH --output /users/afengler/batch_job_out/data_handler

#Request runtime, memory, cores:
#SBATCH --time=36:00:00
#SBATCH --mem=64G
#SBATCH -c 2
#SBATCH -N 1
#SBATCH --array=1-1

# machine="ccv"
# ids=( -1 ) #( 0 1 2 3 4 5 6 7 8 9)
# ndata=( 1024 4096 ) #  4096 )
# method="ddm"
# analytic=0
# initmode='mle'

# for n in "${ndata[@]}"
# do
#     for id in "${ids[@]}"
#     do 
#        python -u  /users/afengler/git_repos/nn_likelihoods/mcmc_data_handler.py --machine $machine --method $method --initmode $initmode --ndata $n --nsubsample 20000 --nnbatchid $id --analytic $analytic
#     done
# done


# machine="ccv"
# ids=( -1 ) #( 0 1 2 3 4 5 6 7 8 9)
# ndata=( 1024 4096 ) #  4096 )
# method="ddm_analytic"
# analytic=0
# initmode='mle'

# for n in "${ndata[@]}"
# do
#     for id in "${ids[@]}"
#     do 
#        python -u  /users/afengler/git_repos/nn_likelihoods/mcmc_data_handler.py --machine $machine --method $method --initmode $initmode --ndata $n --nsubsample 20000 --nnbatchid $id --analytic $analytic
#     done
# done


# machine="ccv"
# ids=( -1 ) #( 0 1 2 3 4 5 6 7 8 9)
# ndata=( 1024 4096 ) #  4096 )
# method="ddm_analytic"
# analytic=1
# initmode='mle'

# for n in "${ndata[@]}"
# do
#     for id in "${ids[@]}"
#     do 
#        python -u  /users/afengler/git_repos/nn_likelihoods/mcmc_data_handler.py --machine $machine --method $method --initmode $initmode --ndata $n --nsubsample 20000 --nnbatchid $id --analytic $analytic
#     done
# done


# machine="ccv"
# ids=( -1 ) #( 0 1 2 3 4 5 6 7 8 9)
# ndata=( 1024 4096 ) #  4096 )
# method="ddm_sdv"
# analytic=0
# initmode='mle'

# for n in "${ndata[@]}"
# do
#     for id in "${ids[@]}"
#     do 
#        python -u  /users/afengler/git_repos/nn_likelihoods/mcmc_data_handler.py --machine $machine --method $method --initmode $initmode --ndata $n --nsubsample 20000 --nnbatchid $id --analytic $analytic
#     done
# done


# machine="ccv"
# ids=( -1 ) #( 0 1 2 3 4 5 6 7 8 9)
# ndata=( 1024 4096 ) #  4096 )
# method="ddm_sdv_analytic"
# analytic=0
# initmode='mle'

# for n in "${ndata[@]}"
# do
#     for id in "${ids[@]}"
#     do 
#        python -u  /users/afengler/git_repos/nn_likelihoods/mcmc_data_handler.py --machine $machine --method $method --initmode $initmode --ndata $n --nsubsample 20000 --nnbatchid $id --analytic $analytic
#     done
# done


# machine="ccv"
# ids=( -1 ) #( 0 1 2 3 4 5 6 7 8 9)
# ndata=( 1024 4096 ) #  4096 )
# method="ddm_sdv_analytic"
# analytic=1
# initmode='mle'

# for n in "${ndata[@]}"
# do
#     for id in "${ids[@]}"
#     do 
#        python -u  /users/afengler/git_repos/nn_likelihoods/mcmc_data_handler.py --machine $machine --method $method --initmode $initmode --ndata $n --nsubsample 20000 --nnbatchid $id --analytic $analytic
#     done
# done


# machine="ccv"
# ids=( -1 ) #( 0 1 2 3 4 5 6 7 8 9)
# ndata=( 1024 4096 ) #  4096 )
# method="ornstein"
# analytic=0
# initmode='mle'

# for n in "${ndata[@]}"
# do
#     for id in "${ids[@]}"
#     do 
#        python -u  /users/afengler/git_repos/nn_likelihoods/mcmc_data_handler.py --machine $machine --method $method --initmode $initmode --ndata $n --nsubsample 20000 --nnbatchid $id --analytic $analytic
#     done
# done


# machine="ccv"
# ids=( -1 ) #( 0 1 2 3 4 5 6 7 8 9)
# ndata=( 1024 4096 ) #  4096 )
# method="levy"
# analytic=0
# initmode='mle'

# for n in "${ndata[@]}"
# do
#     for id in "${ids[@]}"
#     do 
#        python -u  /users/afengler/git_repos/nn_likelihoods/mcmc_data_handler.py --machine $machine --method $method --initmode $initmode --ndata $n --nsubsample 20000 --nnbatchid $id --analytic $analytic
#     done
# done


# machine="ccv"
# ids=( -1 ) #( 0 1 2 3 4 5 6 7 8 9)
# ndata=( 1024 4096 ) #  4096 )
# method="full_ddm2"
# analytic=0
# initmode='mle'

# for n in "${ndata[@]}"
# do
#     for id in "${ids[@]}"
#     do 
#        python -u  /users/afengler/git_repos/nn_likelihoods/mcmc_data_handler.py --machine $machine --method $method --initmode $initmode --ndata $n --nsubsample 20000 --nnbatchid $id --analytic $analytic
#     done
# done


# machine="ccv"
# ids=( -1 ) #( 0 1 2 3 4 5 6 7 8 9)
# ndata=( 1024 4096 ) #  4096 )
# method="weibull_cdf2"
# analytic=0
# initmode='mle'

# for n in "${ndata[@]}"
# do
#     for id in "${ids[@]}"
#     do 
#        python -u  /users/afengler/git_repos/nn_likelihoods/mcmc_data_handler.py --machine $machine --method $method --initmode $initmode --ndata $n --nsubsample 20000 --nnbatchid $id --analytic $analytic
#     done
# done


# machine="ccv"
# ids=( -1 ) #( 0 1 2 3 4 5 6 7 8 9)
# ndata=( 1024 4096 ) #  4096 )
# method="angle2"
# analytic=0
# initmode='mle'

# for n in "${ndata[@]}"
# do
#     for id in "${ids[@]}"
#     do 
#        python -u  /users/afengler/git_repos/nn_likelihoods/mcmc_data_handler.py --machine $machine --method $method --initmode $initmode --ndata $n --nsubsample 20000 --nnbatchid $id --analytic $analytic
#     done
# done

# machine="ccv"
# ids=( -1 ) #( 0 1 2 3 4 5 6 7 8 9)
# ndata=( 1024 4096 ) #  4096 )
# method="levy"
# analytic=0
# sampler='diffevo'
# fileprefix='elife_diffevo_'
# initmode='mle'
# nsubsample=10000
# modelidentifier='_100k'

# for n in "${ndata[@]}"
# do
#     for id in "${ids[@]}"
#     do 
#        python -u  /users/afengler/git_repos/nn_likelihoods/mcmc_data_handler.py --machine $machine \
#                                                                                 --method $method \
#                                                                                 --initmode $initmode \
#                                                                                 --ndata $n \
#                                                                                 --nsubsample $nsubsample \
#                                                                                 --nnbatchid $id \
#                                                                                 --analytic $analytic \
#                                                                                 --modelidentifier $modelidentifier \
#                                                                                 --fileprefix $fileprefix \
#                                                                                 --sampler $sampler
#     done
# done

machine="ccv"
ids=( -1 ) #( 0 1 2 3 4 5 6 7 8 9)
ndata=( 1024 4096 ) #  4096 )
method="ornstein"
analytic=0
sampler='diffevo'
fileprefix='elife_diffevo_'
initmode='mle'
nsubsample=10000
modelidentifier='_100k'

for n in "${ndata[@]}"
do
    for id in "${ids[@]}"
    do 
       python -u  /users/afengler/git_repos/nn_likelihoods/mcmc_data_handler.py --machine $machine \
                                                                                --method $method \
                                                                                --initmode $initmode \
                                                                                --ndata $n \
                                                                                --nsubsample $nsubsample \
                                                                                --nnbatchid $id \
                                                                                --analytic $analytic \
                                                                                --modelidentifier $modelidentifier \
                                                                                --fileprefix $fileprefix \
                                                                                --sampler $sampler
    done
done

machine="ccv"
ids=( -1 ) #( 0 1 2 3 4 5 6 7 8 9)
ndata=( 1024 4096 ) #  4096 )
method="full_ddm2"
analytic=0
sampler='diffevo'
fileprefix='elife_diffevo_'
initmode='mle'
nsubsample=10000
modelidentifier='_100k'

for n in "${ndata[@]}"
do
    for id in "${ids[@]}"
    do 
       python -u  /users/afengler/git_repos/nn_likelihoods/mcmc_data_handler.py --machine $machine \
                                                                                --method $method \
                                                                                --initmode $initmode \
                                                                                --ndata $n \
                                                                                --nsubsample $nsubsample \
                                                                                --nnbatchid $id \
                                                                                --analytic $analytic \
                                                                                --modelidentifier $modelidentifier \
                                                                                --fileprefix $fileprefix \
                                                                                --sampler $sampler
    done
done

machine="ccv"
ids=( -1 ) #( 0 1 2 3 4 5 6 7 8 9)
ndata=( 1024 4096 ) #  4096 )
method="weibull_cdf2"
analytic=0
sampler='diffevo'
fileprefix='elife_diffevo_'
initmode='mle'
nsubsample=10000
modelidentifier='_100k'

for n in "${ndata[@]}"
do
    for id in "${ids[@]}"
    do 
       python -u  /users/afengler/git_repos/nn_likelihoods/mcmc_data_handler.py --machine $machine \
                                                                                --method $method \
                                                                                --initmode $initmode \
                                                                                --ndata $n \
                                                                                --nsubsample $nsubsample \
                                                                                --nnbatchid $id \
                                                                                --analytic $analytic \
                                                                                --modelidentifier $modelidentifier \
                                                                                --fileprefix $fileprefix \
                                                                                --sampler $sampler
    done
done

machine="ccv"
ids=( -1 ) #( 0 1 2 3 4 5 6 7 8 9)
ndata=( 1024 4096 ) #  4096 )
method="angle2"
analytic=0
sampler='diffevo'
fileprefix='elife_diffevo_'
initmode='mle'
nsubsample=10000
modelidentifier='_100k'

for n in "${ndata[@]}"
do
    for id in "${ids[@]}"
    do 
       python -u  /users/afengler/git_repos/nn_likelihoods/mcmc_data_handler.py --machine $machine \
                                                                                --method $method \
                                                                                --initmode $initmode \
                                                                                --ndata $n \
                                                                                --nsubsample $nsubsample \
                                                                                --nnbatchid $id \
                                                                                --analytic $analytic \
                                                                                --modelidentifier $modelidentifier \
                                                                                --fileprefix $fileprefix \
                                                                                --sampler $sampler
    done
done

machine="ccv"
ids=( -1 ) #( 0 1 2 3 4 5 6 7 8 9)
ndata=( 1024 4096 ) #  4096 )
method="ddm_sdv"
analytic=0
sampler='diffevo'
fileprefix='elife_diffevo_'
initmode='mle'
nsubsample=10000
modelidentifier='_100k'

for n in "${ndata[@]}"
do
    for id in "${ids[@]}"
    do 
       python -u  /users/afengler/git_repos/nn_likelihoods/mcmc_data_handler.py --machine $machine \
                                                                                --method $method \
                                                                                --initmode $initmode \
                                                                                --ndata $n \
                                                                                --nsubsample $nsubsample \
                                                                                --nnbatchid $id \
                                                                                --analytic $analytic \
                                                                                --modelidentifier $modelidentifier \
                                                                                --fileprefix $fileprefix \
                                                                                --sampler $sampler
    done
done

machine="ccv"
ids=( -1 ) #( 0 1 2 3 4 5 6 7 8 9)
ndata=( 1024 4096 ) #  4096 )
method="ddm"
analytic=0
sampler='diffevo'
fileprefix='elife_diffevo_'
initmode='mle'
nsubsample=10000
modelidentifier='_100k'

for n in "${ndata[@]}"
do
    for id in "${ids[@]}"
    do 
       python -u  /users/afengler/git_repos/nn_likelihoods/mcmc_data_handler.py --machine $machine \
                                                                                --method $method \
                                                                                --initmode $initmode \
                                                                                --ndata $n \
                                                                                --nsubsample $nsubsample \
                                                                                --nnbatchid $id \
                                                                                --analytic $analytic \
                                                                                --modelidentifier $modelidentifier \
                                                                                --fileprefix $fileprefix \
                                                                                --sampler $sampler
    done
done


# CNN

# machine="ccv"
# ids=( -1 ) #( 0 1 2 3 4 5 6 7 8 9)
# ndata=( 1024 4096 ) #  4096 )
# method="ddm"
# analytic=0

# for n in "${ndata[@]}"
# do
#     for id in "${ids[@]}"
#     do 
#        python -u  /users/afengler/git_repos/nn_likelihoods/is_data_handler.py --machine $machine --method $method --ndata $n --nsubsample 20000
#     done
# done

# machine="ccv"
# ids=( -1 ) #( 0 1 2 3 4 5 6 7 8 9)
# ndata=( 1024 4096 ) #  4096 )
# method="weibull_cdf"
# analytic=0

# for n in "${ndata[@]}"
# do
#     for id in "${ids[@]}"
#     do 
#        python -u  /users/afengler/git_repos/nn_likelihoods/is_data_handler.py --machine $machine --method $method --ndata $n --nsubsample 20000
#     done
# done

# machine="ccv"
# ids=( -1 ) #( 0 1 2 3 4 5 6 7 8 9)
# ndata=( 1024 4096 ) #  4096 )
# method="race_model_3"
# analytic=0

# for n in "${ndata[@]}"
# do
#     for id in "${ids[@]}"
#     do 
#        python -u  /users/afengler/git_repos/nn_likelihoods/is_data_handler.py --machine $machine \
#                                                                               --method $method \
#                                                                               --ndata $n \
#                                                                               --nsubsample $nsubsample \
#                                                                               --fileprefix $fileprefix \
#                                                                               --initmode $initmode \
#                                                                               --sampler $sampler \
#                                                                               --analytic $analytic \
#     done
# done

# machine="ccv"
# ids=( -1 ) #( 0 1 2 3 4 5 6 7 8 9)
# ndata=( 1024 4096 ) #  4096 )
# method="race_model_4"
# analytic=0

# for n in "${ndata[@]}"
# do
#     for id in "${ids[@]}"
#     do 
#        python -u  /users/afengler/git_repos/nn_likelihoods/is_data_handler.py --machine $machine --method $method --ndata $n --nsubsample 20000
#     done
# done

# machine="ccv"
# ids=( -1 ) #( 0 1 2 3 4 5 6 7 8 9)
# ndata=( 1024 4096 ) #  4096 )
# method="lca_3"
# analytic=0

# for n in "${ndata[@]}"
# do
#     for id in "${ids[@]}"
#     do 
#        python -u  /users/afengler/git_repos/nn_likelihoods/is_data_handler.py --machine $machine --method $method --ndata $n --nsubsample 20000
#     done
# done

# machine="ccv"
# ids=( -1 ) #( 0 1 2 3 4 5 6 7 8 9)
# ndata=( 1024 4096 ) #  4096 )
# method="lca_4"
# analytic=0

# for n in "${ndata[@]}"
# do
#     for id in "${ids[@]}"
#     do 
#        python -u  /users/afengler/git_repos/nn_likelihoods/is_data_handler.py --machine $machine --method $method --ndata $n --nsubsample 20000
#     done
# done