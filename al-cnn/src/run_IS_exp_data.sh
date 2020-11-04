for sample in {1..3}
do
CUDA_VISIBLE_DEVICES=0,1 python importance_sampler_tdistributionV2.py --model ddm_par2 --nsample $sample --nbin 512 --N 1024
done

for sample in {1..3}
do
CUDA_VISIBLE_DEVICES=0,1 python importance_sampler_tdistributionV2.py --model ddm_seq2 --nsample $sample --nbin 512 --N 1024
done

for sample in {0..3}
do
CUDA_VISIBLE_DEVICES=0,1 python importance_sampler_tdistributionV2.py --model ddm_mic2 --nsample $sample --nbin 512 --N 1024
done
