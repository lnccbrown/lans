#!/bin/bash

# Default resources are 1 core with 2.8GB of memory per core.

# job name:
#SBATCH -J train_dat

# priority
##SBATCH --account=bibs-frankmj-condo
#SBATCH --account=carney-frankmj-condo

# output file
#SBATCH --output /users/afengler/batch_job_out/tpl_1_%A_%a.out

# Request runtime, memory, cores
#SBATCH --time=6:00:00
#SBATCH --mem=128G
#SBATCH -c 12
#SBATCH -N 1
#SBATCH --array=1-300

# --------------------------------------------------------------------------------------
# Sequentially run different kind of models

#declare -a dgps=( "ddm" "full_ddm" "angle" "weibull_cdf" "ornstein" "lca" "race_model" "ddm_seq2" "ddm_par2" "ddm_mic2" "ddm_seq2_angle" "ddm_par2_angle" "ddm_mic2_angle") 
declare -a dgps=( "race_model_3" )  # ( "angle2" )  #( "weibull_cdf" ) # ( "ddm_seq2_angle" "ddm_mic2_angle" "ddm_par2_angle" )
n_samples=( 10000 )   # ( 128 256 512 1024 2048 4096 8192 50000 100000 200000 400000 )
n_choices=( 3 ) #( 4 5 6 )
n_parameter_sets=100   #20000
n_bins=( 0 )
binned=0
machine="home" #"ccv"
datatype="cnn_train" #"cnn_train" # "parameter_recovery"
mode="mlp"
analytic=0
maxt=20
nproc=12

# params concerning training data generation
nbyparam=1000

# outer -------------------------------------
for bins in "${n_bins[@]}"
do
    for n in "${n_samples[@]}"
    do
    # inner ----------------------------tmux---------
        for dgp in "${dgps[@]}"
        do
            if [[ "$dgp" = "lca" ]] || [[ "$dgp" = "race_model" ]];
            then
                for n_c in "${n_choices[@]}"
                    do
                       echo "$dgp"
                       echo $n_c
                       python -u dataset_generator.py --machine $machine \
                                                      --dgplist $dgp \
                                                      --datatype $datatype \
                                                      --nreps 1 \
                                                      --binned $binned \
                                                      --nbins $bins \
                                                      --maxt $maxt \
                                                      --nchoices $n_c \
                                                      --nsamples $n \
                                                      --mode $mode \
                                                      --nparamsets $n_parameter_sets \
                                                      --save 1 \
                                                      --deltat 0.001 \
                                                      --fileid 'TEST'
                       
                       python -u simulator_get_stats.py --machine $machine \
                                                        --method $dgp \
                                                        --simfolder training_data_binned_${binned}_nbins_${bins}_n_${n} \
                                                        --fileprefix ${dgp}_nchoices_${n_c}_train_data_binned_${binned}_nbins_${bins}_n_${n} \
                                                        --fileid 'TEST'
                       
                       python -u kde_train_test.py --machine $machine \
                                                   --method $dgp \
                                                   --simfolder training_data_binned_${binned}_nbins_${bins}_n_${n} \
                                                   --fileprefix ${dgp}_nchoices_${n_c}_train_data_binned_${binned}_nbins_${bins}_n_${n} \
                                                   --outfolder training_data_binned_${binned}_nbins_${bins}_n_${n} \
                                                   --nbyparam $nbyparam \
                                                   --mixture 0.8 0.1 0.1 \
                                                   --fileid 'TEST' \ 
                                                   --nproc $nproc \
                                                   --analytic $analytic
                    #--fileid $SLURM_ARRAY_TASK_ID \
           
                done
            else
                 echo "$dgp"
                 echo ${n_choices[0]}
                 python -u dataset_generator.py --machine $machine \
                                                --dgplist $dgp \
                                                --datatype $datatype \
                                                --nreps 1 \
                                                --binned $binned \
                                                --nbins $bins \
                                                --maxt $maxt \
                                                --nchoices ${n_choices[0]} \
                                                --nsamples $n \
                                                --mode $mode \
                                                --nparamsets $n_parameter_sets \
                                                --save 1 \
                                                --deltat 0.001 \
                                                --fileid 'TEST'
                                                
                 python -u simulator_get_stats.py --machine $machine \
                                                  --method $dgp \
                                                  --simfolder training_data_binned_${binned}_nbins_${bins}_n_${n} \
                                                  --fileprefix ${dgp}_nchoices_${n_choices[0]}_train_data_binned_${binned}_nbins_${bins}_n_${n} \
                                                  --fileid 'TEST' 
                                                  
                 python -u kde_train_test.py --machine $machine \
                                             --method $dgp \
                                             --simfolder training_data_binned_${binned}_nbins_${bins}_n_${n} \
                                             --fileprefix ${dgp}_nchoices_${n_choices[0]}_train_data_binned_${binned}_nbins_${bins}_n_${n} \
                                             --outfolder training_data_binned_${binned}_nbins_${bins}_n_${n} \
                                             --nbyparam $nbyparam \
                                             --mixture 0.8 0.1 0.1 \
                                             --fileid 'TEST' \
                                             --nproc $nproc \
                                             --analytic $analytic
            fi
        done
                # normal call to function
    done
done
#---------------------------------------------------------------------------------------