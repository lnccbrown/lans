# projectABC
This repository houses code for training deep neural networks to approximate the likelihood function of a family of Sequential Sampling Models. We provide two separate strategies for this: the pointwise and the dataset approach. More details regarding the ideas behind this work can be found at https://elifesciences.org/articles/65074

# al-mlpL The pointwise approach
Provides implementation for the "pointwise" approach. 

# al-cnn: The dataset approach
`al-cnn/src/config.py` contains the path to model weights/checkpoints. Also used to specify the various hyperparameter settings, including the type of SSM (and related options such as number of choices, boundary values, etc.), bin size and dataset size. 

After simulating the neccesary dataset, we prepare for training by creating tfrecords.

`cd al-cnn/src`

`python create_tfrecords.py`

To train the forward model, we run

`CUDA_VISIBLE_DEVICES=0 python run_pipeline.py --train`

To run the Importance Sampler, we run

`CUDA_VISIBLE_DEVICES=0 python sampler_inference.py --model <ddm> --nsample <1> --nbin <256> --N <1024> --proposal <normal/tdist>`

Just to run MLE for parameter recovery

`CUDA_VISIBLE_DEVICES=0 python inference.py --model <ddm> --nbin <256> --N <1024>`

# hddmnn\_tutorial
Provides a demo for using the above two approaches in the context of hierarchical parameter estimation using the HDDM package.
