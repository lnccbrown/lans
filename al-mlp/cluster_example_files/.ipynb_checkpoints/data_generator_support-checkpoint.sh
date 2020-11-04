#!/bin/bash
n_data_points=( 100 250 500 1000 2000 3000 )
#file_id=( 1 2 3 4 5 6 7 8 9 10 )
n_experiments=1
# outer -------------------------------------
for n in "${n_data_points[@]}"
do 
# inner ---------------------------------
#for id in "${file_id[@]}"
for id in {1..100}
do
    python dataset_generator.py ccv weibull_cdf_ndt uniform 1000 1 $n 1
#         echo $n
#         echo $id
done
# ---------------------------------------
#done
# -------------------------------------------