#!/bin/bash
# bash batch_importance_sampler.sh angle gpu N
model=$1
for sample in {0..1000}
do
CUDA_VISIBLE_DEVICES=$2 python sampler_inference_v2.py --model $model --nsample $sample --nbin 512 --N $3 --proposal tdist
done
