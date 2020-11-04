# KDE
models=( "ddm" "ddm_sdv" "full_ddm2" "levy" "ornstein" "weibull_cdf" "angle2")

for model in "${models[@]}"
    do
        scp -r afengler@transfer.ccv.brown.edu:/users/afengler/data/kde/$model/parameter_recovery_data_binned_0* /users/afengler/OneDrive/project_nn_likelihoods/data/kde/$model/
    done







# ANALYTIC
models=( "ddm" "ddm_sdv" )

for model in "${models[@]}"
    do
        scp -r afengler@transfer.ccv.brown.edu:/users/afengler/data/analytic/$model/parameter_recovery_data_binned_0* /users/afengler/OneDrive/project_nn_likelihoods/data/analytic/$model/
    done
