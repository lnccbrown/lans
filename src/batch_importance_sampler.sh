model='ddm_sdv'
for sample in {0..100}
do
CUDA_VISIBLE_DEVICES=7 python sampler_inference.py --model $model --nsample $sample --nbin 512 --N 4096 --proposal tdist
done
