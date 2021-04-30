# projectABC
This repository houses code for training deep neural networks to approximate the likelihood function of a family of Sequential Sampling Models. We provide two separate strategies for this: the pointwise and the dataset approach. More details regarding the ideas behind this work can be found at https://elifesciences.org/articles/65074

# al-mlp: The pointwise approach
Provides implementations for the "pointwise" approach. 

**Note** that this code base is not self-contained. We suggest to use the tutorial you can find in the `hddmnn_tutorial` folder to use this method through an extension to the `HDDM` Python package.
You can also check the `hddmnn_tutorial` under <https://github.com/AlexanderFengler/hddmnn_tutorial> for recent updates.

`al-mlp/basic_simulator.py` holds a python function to call the cython simulators (can be called directly as well) defined in `al-mlp/cddm_data_simulation.pyx`.
Run the `al-mlp/setup.py` to load compile the cython simulators (needs the `cython` package installed).

`al-mlp/kde_class.py` holds the class which we use to define KDE based likelihood functions from simulator data.

`al-mlp/keras_fit_model.py` holds the code used to train the mlps given fully pre-pocessed training data.

`al-mlp/full_training_data_generator.py` holds the training data generators (internally making use of the `basic_simulator.py` function mentioned above).

The folder `al-mlp/networks` holds pre-trained networks for the models shown in the paper.

The folder `al-mlp/samplers` holds the mcmc-samplers (including DE-MCMC) used in the paper.
They were called through `al-mlp/method_comparison_sim.py`.
We however suggest to instead use `HDDM` at this point.

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
