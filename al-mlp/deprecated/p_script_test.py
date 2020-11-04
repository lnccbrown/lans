# Load packages
import tensorflow as tf
import pandas as pd
from tensorflow import keras
import numpy as np
import pandas as pd
import os
import scipy as scp
import scipy.stats as scps

# Load my own functions
import dnnregressor_train_eval_keras as dnnk
from kde_training_utilities import kde_load_data
import make_data_wfpt as mdw

#os.environ['CUDA_VISIBLE_DEVICES'] = ['0','1','2','3']

# Make dnnk class (cpm for choice probability model)
cpm = dnnk.dnn_trainer()

# Load data
data_folder = os.getcwd() + '/data_storage/kde/kde_training_dat/ddm_final_train_test'

# rt_choice
cpm.data['train_features'], cpm.data['train_labels'], cpm.data['test_features'], cpm.data['test_labels'] = kde_load_data(folder = data_folder)

# If necessary, specify new set of parameters here:
# Model params
cpm.model_params['output_activation'] = 'linear'
cpm.model_params['input_shape'] = cpm.data['train_features'].shape[1]

# Data params
cpm.data_params['data_type'] = 'wfpt'
cpm.data_params['data_type_signature'] = '_kde_ddm_'
cpm.data_params['training_data_size'] = cpm.data['train_features'].shape[0]

# Make model
with tf.device('/gpu:0'):
    cpm.keras_model_generate(save_model = True)
    
# Train model
with tf.device('/cpu:0'):
    cpm.run_training(save_history = True, 
                     warm_start = False)