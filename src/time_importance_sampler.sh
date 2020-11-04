#!/bin/bash
# bash batch_importance_sampler.sh angle gpu N
for sample in {0..99}
do
CUDA_VISIBLE_DEVICES=0 python sampler_inference_v2.py --model weibull --nsample $((1 + RANDOM % 1000)) --nbin 512 --N 1024 --proposal tdist
CUDA_VISIBLE_DEVICES=0 python sampler_inference_v2.py --model weibull --nsample $((1 + RANDOM % 1000)) --nbin 512 --N 4096 --proposal tdist
CUDA_VISIBLE_DEVICES=0 python sampler_inference_v2.py --model lca_3 --nsample $((1 + RANDOM % 1000)) --nbin 512 --N 1024 --proposal tdist
CUDA_VISIBLE_DEVICES=0 python sampler_inference_v2.py --model lca_3 --nsample $((1 + RANDOM % 1000)) --nbin 512 --N 4096 --proposal tdist
CUDA_VISIBLE_DEVICES=0 python sampler_inference_v2.py --model lca_4 --nsample $((1 + RANDOM % 1000)) --nbin 512 --N 1024 --proposal tdist
CUDA_VISIBLE_DEVICES=0 python sampler_inference_v2.py --model lca_4 --nsample $((1 + RANDOM % 1000)) --nbin 512 --N 4096 --proposal tdist
#CUDA_VISIBLE_DEVICES=0 python sampler_inference_v2.py --model levy --nsample $((1 + RANDOM % 1000)) --nbin 512 --N 1024 --proposal tdist
#CUDA_VISIBLE_DEVICES=0 python sampler_inference_v2.py --model levy --nsample $((1 + RANDOM % 1000)) --nbin 512 --N 4096 --proposal tdist
#CUDA_VISIBLE_DEVICES=0 python sampler_inference_v2.py --model ddm_sdv --nsample $((1 + RANDOM % 1000)) --nbin 512 --N 1024 --proposal tdist
#CUDA_VISIBLE_DEVICES=0 python sampler_inference_v2.py --model ddm_sdv --nsample $((1 + RANDOM % 1000)) --nbin 512 --N 4096 --proposal tdist
done
