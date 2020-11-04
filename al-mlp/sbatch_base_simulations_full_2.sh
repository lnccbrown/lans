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
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH -c 10
#SBATCH -N 1
##SBATCH --array=1-300  # DO THIS FOR TRAINING DATA GENERATION
#SBATCH --array=1-200

# --------------------------------------------------------------------------------------
# Sequentially run different kind of models
declare -a dgps=( "lca_4" ) # "race_model" "lca" ) #"ddm_sdv_analytic" "ddm_sdv_red_analytic" ) #( "ddm" "full_ddm" "angle" "weibull_cdf" "ornstein" "levy" )  #( "ddm_mic2_angle" "ddm_par2_angle" ) # ( "ddm_seq2_angle" )
n_samples=( 100000 )   # ( 128 256 512 1024 2048 4096 8192 50000 100000 200000 400000 )
n_choices=( 4 ) #( 4 5 6 )
nparamsets=5000
nparamsetsrej=20 # cnn 20000 but 150 array   # mlp 10000 but 300 array # KRISHN: 10
n_bins=( 0 ) # KRISHN: n_bins=0
machine="ccv" #"ccv" "home" "x7"
datatype="training"  #"parameter_recovery" "training" "parameter_recovery_hierarchical"
nsubjects=1 #10
#mode="test" # "test" #"test"  #'mlp' 'cnn' # KRISHN: 'test'
maxt=20 # 20 for mlp # KRISHN: 20
save_output=1
# outer -------------------------------------
for bins in "${n_bins[@]}"
do
    for n in "${n_samples[@]}"
    do
    # inner ----------------------------tmux---------
        for dgp in "${dgps[@]}"
        do
#             if [[ "$dgp" = "lca" ]] || [[ "$dgp" = "race_model" ]];
#             then
#                 for n_c in "${n_choices[@]}"
#                     do
#                        echo "$dgp"
#                        echo $n_c
#                        python -u full_training_data_generator.py --machine $machine \
#                                                                  --dgplist $dgp \
#                                                                  --datatype $datatype \
#                                                                  --nsubjects $nsubjects \
#                                                                  --nreps 1 \
#                                                                  --binned $binned \
#                                                                  --nbins $bins \
#                                                                  --maxt $maxt \
#                                                                  --nchoices $n_c \
#                                                                  --nsamples $n \
#                                                                  --mode $mode \
#                                                                  --nparamsets $nparamsets \
#                                                                  --nparamsetsrej $nparamsetsrej \
#                                                                  --save $save_output \
#                                                                  --deltat 0.001 \
#                                                                  --fileid 999 #$SLURM_ARRAY_TASK_ID 
#                 done
#             else
             echo "$dgp"
             #echo $n_c
             python -u full_training_data_generator.py --machine $machine \
                                                          --dgplist $dgp \
                                                          --nsubjects $nsubjects \
                                                          --datatype $datatype \
                                                          --nreps 1 \
                                                          --nbins $bins \
                                                          --maxt $maxt \
                                                          --nchoices ${n_choices[0]} \
                                                          --nsamples $n \
                                                          --nparamsets $nparamsets \
                                                          --nparamsetsrej $nparamsetsrej \
                                                          --save $save_output  \
                                                          --deltat 0.001 \
                                                          --fileid $SLURM_ARRAY_TASK_ID

                                                              # --binned $binned \

            # fi
        done
                # normal call to function
    done
done
#---------------------------------------------------------------------------------------