#CUDA_VISIBLE_DEVICES=0,1 python importance_sampler.py --model ddm --nbin 256 --N 1024
#CUDA_VISIBLE_DEVICES=0,1 python importance_sampler.py --model angle --nbin 256 --N 1024
#CUDA_VISIBLE_DEVICES=0,1 python importance_sampler.py --model weibull --nbin 256 --N 1024
#CUDA_VISIBLE_DEVICES=0,1 python importance_sampler.py --model fullddm --nbin 256 --N 1024
#CUDA_VISIBLE_DEVICES=0,1 python importance_sampler.py --model ornstein --nbin 256 --N 1024
#CUDA_VISIBLE_DEVICES=2,3 python importance_sampler.py --model race_model_3 --nbin 256 --N 1024
#CUDA_VISIBLE_DEVICES=3,3 python importance_sampler.py --model race_model_4 --nbin 256 --N 1024
#CUDA_VISIBLE_DEVICES=3,2 python importance_sampler.py --model lca_3 --nbin 256 --N 1024

model='fullddm'
for sample in {0..4}
do
CUDA_VISIBLE_DEVICES=2 python sampler_inference.py --model $model --nsample $sample --nbin 256 --N 1024 --proposal tdist
done
