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
import re

import boundary_functions as bf
import multiprocessing as mp
import cddm_data_simulation as cd
from cdwiener import batch_fptd
import clba

# SUPPORT FUNCTIONS -------------------------------------------------------------
def load_data_perturbation_experiment(file_ = '..'):
    tmp = pickle.load(open(file_, 'rb'))
    data_grid = tmp[1]
    param_grid = tmp[0]
    return (param_grid, data_grid)

def load_data_parameter_recovery(file_ == '..'):
    tmp = pickle.load(open(file_, 'rb'))
    data_grid = tmp[1]
    param_grid = tmp[0]
    return (param_grid, data_grid)

def run_inference(file_list = ['.', '.'],
                  machine = 'x7',
                  method = 'ddm_ndt',
                  datatype = ,
                  save = True):
    
    # Load Model
    fcn = keras.models.load_model(network_path + 'model_final.h5', custom_objects = fcn_custom_objects)
    print('loaded model')
    # Run inference and store inference files for all files in file_list
    for file_ in file_list:
        print('current file: ')
        print(file_)
        
        if datatype == 'perturbation_experiment': 
            param_grid, data_grid = load_data_perturbation_experiment(file_ = file_)
        if datatype == 'parameter_recovery':
            param_grid, data_grid = load_data_parameter_recoery(file_ = file_)
        if datatype == 'real':
            #param_grid, data_grid = load_data_parameter_recovery(file_ = file_)
            
        # Predictions
        fcn_results = fcn.predict(data_grid)
        
        # Save
        if save == True:
            # Make out file name
            pattern_span = re.search('base_data', file_).span()
            tmp_file_name = file_[:pattern_span[0]] + 'deep_inference' + file_[pattern_span[1]:]
        
            pickle.dump((param_grid, data_grid, fcn_results), 
                        open(tmp_file_name, "wb"))
        return fcn_results
# ----------------------------------------------------------------------------------
# Run    
if __name__ == "__main__":
    
    # Make command line interface
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--method",
                     type = str,
                     default = 'ddm')
    CLI.add_argument("--machine",
                     type = str, 
                     default = 'x7')
    CLI.add_argument("--gpu",
                     type = str, 
                     default = '1')
    CLI.add_argument("--save",
                     type = int,
                     default = 1)
    CLI.add_argument("--infileid",
                     type = str,
                     'base_data_param_recovery_unif_reps_10_n_3000')
    CLI.add_argument("--datatype",
                     type = str,
                     'perturbation_experiment')

    # Setup
    fcn_custom_objects = {"heteroscedastic_loss": tf.losses.huber_loss}
    
    if machine == 'x7':
        method_params = pickle.load(open("/media/data_cifs/afengler/git_repos/nn_likelihoods/kde_stats.pickle", "rb"))[args.method]
        method_comparison_folder = method_params['output_folder_x7']
        os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        
        print(' GPU I AM ASKING FOR: ', args.gpu)
        
        with open("model_paths_x7.yaml") as tmp_file:
            network_path = yaml.load(tmp_file)['fcn_' + method]

    if machine == 'ccv':
        method_params = pickle.load(open("/users/afengler/git_repos/nn_likelihoods/kde_stats.pickle", "rb"))[method]
        method_comparison_folder = method_params['output_folder']
        with open("model_paths.yaml") as tmp_file:
            network_path = yaml.load(tmp_file)['fcn_' + method]

    # Get list of files
    method_comparison_folder = "/media/data_cifs/afengler/data/analytic/ddm/method_comparison/"
    file_signature  = 'base_data_param_recov_unif_reps_10_n_3000'
    file_signature_len = len(file_signature)
    files_ = os.listdir(method_comparison_folder)

    file_list = []
    for file_ in file_list:
        if file_[:file_signature_len] == file_signature:
            file_list.append(file_)
    file_list = [method_comparison_folder + file_ for file_ in file_list]
    
    # Run inference
    run_inference(file_list = file_list,
                   machine = arg.machine,
                   method = args.method,
                   datatype = args.datatype,
                   save = args.save)