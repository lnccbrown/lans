import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow import keras
import numpy as np
import yaml
import pandas as pd
from itertools import product
import pickle
import uuid
import os

import boundary_functions as bf
import multiprocessing as mp
import cddm_data_simulation as cd
from cdwiener import batch_fptd
import clba

# INITIALIZATIONS -------------------------------------------------------------
# print(device_lib.list_local_devices())

machine = 'x7'
method = 'ddm'
#param_origin = 'previous'

#analytic = True
#file_signature = '_start_true_'
#n_data_samples = 2500  # should be hardwired to 2500 for deep inference
#n_sims = 500
fcn_custom_objects = {"heteroscedastic_loss": tf.losses.huber_loss}

# Get list of files
method_comparison_folder = "/media/data_cifs/afengler/data/analytic/ddm/method_comparison/"
file_signature  = 'base_data_param_recov_unif_reps_10_n_3000'
file_signature_len = len(file_signature)
files = os.listdir(method_comparison_folder)

# Get data in desired format
dats = []
for file_ in files:
    if file_[:file_signature_len] == file_signature:
        dats.append(pickle.load(open(method_comparison_folder + file_ , 'rb')))

dat_tmp_0 = []
dat_tmp_1 = []
for dat in dats:
    dat_tmp_0.append(dat[0])
    dat_tmp_1.append(dat[1])
    print(dat_tmp_0)

dat_total = [dat_tmp_0, dat_tmp_1]
data_grid = dat_total[0]
param_grid = dat_total[1]
#dat_total = [np.concatenate(dat_tmp_0, axis = 0), np.concatenate(dat_tmp_1, axis = 0)]
#data_grid = dat_total[1]
#param_grid = dat_total[0]
    
# Get network hyperparameters
dnn_params = yaml.load(open("hyperparameters.yaml"))
print(' GPU I AM ASKING FOR: ', dnn_params['gpu_x7'])

if machine == 'x7':
    stats = pickle.load(open("/media/data_cifs/afengler/git_repos/nn_likelihoods/kde_stats.pickle", "rb"))
    method_params = stats[method]
    output_folder = method_params['output_folder_x7']
    os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = dnn_params['gpu_x7'] 

    with open("model_paths_x7.yaml") as tmp_file:
        network_path = yaml.load(tmp_file)['fcn_' + method]

if machine == 'ccv':
    stats = pickle.load(open("/users/afengler/git_repos/nn_likelihoods/kde_stats.pickle", "rb"))
    method_params = stats[method]
    output_folder = method_params['output_folder']
    with open("model_paths.yaml") as tmp_file:
        network_path = yaml.load(tmp_file)['fcn_' + method]
        
#print(stats)
#print(method_params)

# MAKE PARAMETER / DATA GRID -------------------------------------------------------------------------

fcn = keras.models.load_model(network_path + 'model_final.h5', custom_objects = fcn_custom_objects)
fcn_results = fcn.predict(data_grid)
print('sucessfully executed')
#pickle.dump((param_grid, data_grid, fcn_results), open(output_folder + "deep_inference_rep_10_n_3000_{}.pickle".format(uuid.uuid1()), "wb"))