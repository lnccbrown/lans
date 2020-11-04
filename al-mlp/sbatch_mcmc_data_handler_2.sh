#!/bin/bash

# Default resources are 1 core with 2.8GB of memory per core.

# job name:
#SBATCH -J mcmc_data_handler

# priority
##SBATCH --account=carney-frankmj-condo

# output file
#SBATCH --output /users/afengler/batch_job_out/data_handler

# Request runtime, memory, cores:
#SBATCH --time=36:00:00
#SBATCH --mem=64G
#SBATCH -c 10
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
# method="ornstein_pos"
# analytic=0
# initmode='mle'

# for n in "${ndata[@]}"
# do
#     for id in "${ids[@]}"
#     do 
#        python -u  /users/afengler/git_repos/nn_likelihoods/mcmc_data_handler.py --machine $machine --method $method --initmode $initmode --ndata $n --nsubsample 20000 --nnbatchid $id --analytic $analytic
#     done
# done



# CNN

machine="x7"
ids=( -1 ) #( 0 1 2 3 4 5 6 7 8 9)
ndata=( 1024 4096 ) #  4096 )
method="ddm_sdv"
analytic=0

for n in "${ndata[@]}"
do
    for id in "${ids[@]}"
    do 
       python -u  is_data_handler.py --machine $machine \
                                                                              --method $method \
                                                                              --ndata $n \
                                                                              --nsubsample 20000 \
                                                                              --isfolder 'results/time_benchmark_eLIFE_exps_round2'
    done
done

machine="x7"
ids=( -1 ) #( 0 1 2 3 4 5 6 7 8 9)
ndata=( 1024 4096 ) #  4096 )
method="ddm"
analytic=0

for n in "${ndata[@]}"
do
    for id in "${ids[@]}"
    do 
       python -u  is_data_handler.py --machine $machine \
                                                                              --method $method \
                                                                              --ndata $n \
                                                                              --nsubsample 20000 \
                                                                              --isfolder 'results/time_benchmark_eLIFE_exps_round2'
    done
done



# # python -u /users/afengler/git_repos/nn_likelihoods/mcmc_data_handler.py --machine ccv --method ddm --nburnin 5000 --ndata 2048 --nsubsample 10000 --nnbatchid -1
# python -u /users/afengler/git_repos/nn_likelihoods/mcmc_data_handler.py --machine ccv --method ddm --nburnin 5000 --ndata 4096 --nsubsample 10000 --nnbatchid -1
# #python -u /users/afengler/git_repos/nn_likelihoods/mcmc_data_handler.py --machine ccv --method ornstein --nburnin 5000 --ndata 2048 --nsubsample 10000 --nnbatchid -1
# python -u /users/afengler/git_repos/nn_likelihoods/mcmc_data_handler.py --machine ccv --method ornstein --nburnin 5000 --ndata 4096 --nsubsample 10000 --nnbatchid -1
# #python -u /users/afengler/git_repos/nn_likelihoods/mcmc_data_handler.py --machine ccv --method full_ddm --nburnin 5000 --ndata 2048 --nsubsample 10000 --nnbatchid -1
# python -u /users/afengler/git_repos/nn_likelihoods/mcmc_data_handler.py --machine ccv --method full_ddm --nburnin 5000 --ndata 4096 --nsubsample 10000 --nnbatchid -1
# #python -u /users/afengler/git_repos/nn_likelihoods/mcmc_data_handler.py --machine ccv --method levy --nburnin 5000 --ndata 2048 --nsubsample 10000 --nnbatchid -1
# python -u /users/afengler/git_repos/nn_likelihoods/mcmc_data_handler.py --machine ccv --method levy --nburnin 5000 --ndata 4096 --nsubsample 10000 --nnbatchid -1
# #python -u /users/afengler/git_repos/nn_likelihoods/mcmc_data_handler.py --machine ccv --method weibull_cdf --nburnin 5000 --ndata 2048 --nsubsample 10000 --nnbatchid -1
# python -u /users/afengler/git_repos/nn_likelihoods/mcmc_data_handler.py --machine ccv --method weibull_cdf --nburnin 5000 --ndata 4096 --nsubsample 10000 --nnbatchid -1
# #python -u /users/afengler/git_repos/nn_likelihoods/mcmc_data_handler.py --machine ccv --method angle --nburnin 5000 --ndata 2048 --nsubsample 10000 --nnbatchid -1
# python -u /users/afengler/git_repos/nn_likelihoods/mcmc_data_handler.py --machine ccv --method angle --nburnin 5000 --ndata 4096 --nsubsample 10000 --nnbatchid -1
