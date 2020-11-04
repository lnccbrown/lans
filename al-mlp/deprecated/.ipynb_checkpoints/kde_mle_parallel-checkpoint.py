# Load packages
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow import keras
import numpy as np
import pandas as pd
import os
import pickle
import time
import uuid
import scipy as scp
import scipy.stats as scps
from scipy.optimize import differential_evolution
from datetime import datetime
import yaml

# Load my own functions
import keras_to_numpy as ktnp
from kde_training_utilities import kde_load_data # Want to overcome
import cddm_data_simulation as cds
import boundary_functions as bf

# SUPPORT FUNCTIONS ------------
def make_params(param_bounds = [], param_bounds_epsilon =[]):
    params = np.zeros(len(param_bounds))
    
    for i in range(len(params)):
        params[i] = np.random.uniform(low = param_bounds[i][0] + param_bounds_epsilon[i], high = param_bounds[i][1] - param_bounds_epsilon[i])    
    return params

# def get_params_from_meta_data(file_path  = ''):
#     # Loading meta data file (,,) (simulator output at this point)
#     tmp = pickle.load(open(file_path, 'rb'))[2]
#     params = []
#     # for loop makes use of common structure of simulator outputs across models
#     for key in tmp.keys():
#         # delta_t signifies start of simulator parameters that we don't care about for our purposes here
#         if key == 'delta_t':
#             break
#         # variance parameter not used thus far, others added
#         if key != 's':
#             params.append(key)
#     return params 

def adjust_params_names_group(params = ['v', 'a', 'w'], 
                              params_bounds = [], 
                              params_bounds_epsilon= [],
                              param_varies = [0, 0, 1],
                              n_subjects = 3):
    params_adj = []
    params_bounds_adj = []
    params_bounds_epsilon_adj = []
    cnt = 0
    for p in params:
        if param_varies[cnt]:
            for i in range(n_subjects):
                params_adj.append(p + '_' + str(i))
                params_bounds_adj.append(params_bounds[cnt])
                params_bounds_epsilon_adj.append(params_bounds_epsilon[cnt])
        else:
            params_adj.append(p)
            params_bounds_adj.append(params_bounds[cnt])
            params_bounds_epsilon_adj.append(params_bounds_epsilon[cnt])
        cnt += 1
    return params_adj, params_bounds_adj, params_bounds_epsilon_adj

def make_data(param_bounds = [],
              param_bounds_epsilon = [],
              param_is_boundary_param = [0, 0, 1],
              param_names = ['v', 'a', 'w']):
    
    # Generate set of parameters
    tmp_params = make_params(param_bounds = param_bounds, param_bounds_epsilon = param_bounds_epsilon)

    # Define boundary parameters 
    boundary_params = {}
    cnt = 0
    
    for param in parameter_names:
        if param_is_boundary_param[cnt]:
            boundary_params[param] = tmp_params[cnt]
        cnt += 1

    # Run model simulations: MANUAL INTERVENTION TD: AUTOMATE
    ddm_dat_tmp = cds.ddm_flexbound(v = tmp_params[param_names.index('v')],
                                    a = tmp_params[param_names.index('a')],
                                    w = tmp_params[param_names.index('w')],
                                    s = 1,
                                    delta_t = 0.001,
                                    max_t = 20,
                                    n_samples = n_samples,
                                    boundary_fun = boundary, # function of t (and potentially other parameters) that takes in (t, *args)
                                    boundary_multiplicative = boundary_multiplicative, # CAREFUL: CHECK IF BOUND
                                    boundary_params = boundary_params)

    data_np = np.concatenate([ddm_dat_tmp[0], ddm_dat_tmp[1]], axis = 1)
    return data_np, tmp_params

def make_data_group(param_bounds = [],
                    param_bounds_epsilon = [],
                    param_is_boundary_param = [0, 0, 1], 
                    params_ordered = ['v', 'a', 'w', 'node', 'theta'],
                    param_varies = [0, 0, 1],
                    params_names = ['v', 'a', 'w_0', 'w_1', 'w_2'],
                    n_subjects = 3):
    
    # Generate set of parameters
    tmp_params_full = make_params(param_bounds = param_bounds, param_bounds_epsilon = param_bounds_epsilon)
    data = {}
    for i in range(n_subjects):
        tmp_params = ktnp.get_tmp_params(params = tmp_params_full,
                                         params_ordered = params_ordered,
                                         param_varies = param_varies,
                                         params_names = params_names,
                                         idx = i)
        
        # Define boundary parameters 
        boundary_params = {}
        cnt = 0

        for param in parameter_names:
            if param_is_boundary_param[cnt]:
                boundary_params[param] = tmp_params[cnt]
            cnt += 1

        # Run model simulations: MANUAL INTERVENTION TD: AUTOMATE
        ddm_dat_tmp = cds.ddm_flexbound(v = tmp_params[params_ordered.index('v')],
                                        a = tmp_params[params_ordered.index('a')],
                                        w = tmp_params[params_ordered.index('w')],
                                        s = 1,
                                        delta_t = 0.01,
                                        max_t = 20,
                                        n_samples = n_samples,
                                        boundary_fun = boundary, # function of t (and potentially other parameters) that takes in (t, *args)
                                        boundary_multiplicative = boundary_multiplicative, # CAREFUL: CHECK IF BOUND IS MULTIPLICATIVE
                                        boundary_params = boundary_params)

        data[str(i)] = np.concatenate([ddm_dat_tmp[0], ddm_dat_tmp[1]], axis = 1)
    return data, tmp_params_full

# -------------------------------

if __name__ == "__main__":
    
    # Initializations -------------
    print('Running intialization ....')
    
    # Get configuration from yaml file
    print('Reading config file .... ')
    yaml_config_path = os.getcwd() + '/kde_mle_parallel.yaml' # MANUAL INTERVENTION
    with open(yaml_config_path, 'r') as stream:
        config_data = yaml.unsafe_load(stream)
        
    # Handle cuda business if necessary
    #     Handle some cuda business (if desired to use cuda here..)
    if config_data['cuda_on']:
        print('Handle cuda business....')   
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"]="3"
        print(device_lib.list_local_devices())
    
    # Load Model
    print('Loading model .... ')
    model_dtype = config_data['model_dtype']
    
    if model_dtype == 'ckpt':
        model_path = config_data['model_path']
        ckpt_path = config_data['ckpt_path']
        model = keras.models.load_model(model_path)
        model.load_weights(ckpt_path)
    if model_dtype == 'h5':
        model_path = config_data['model_path']
        model = keras.models.load_model(model_path, custom_objects = {"huber_loss": tf.losses.huber_loss})
    
    # get network architecture for numpy forward pass (used in mle, coming from ktnp imported)
    weights, biases, activations = ktnp.extract_architecture(model)

    # LOAD CONFIG FILE INTO VARIABLES ----
    print('Setting parameters from config file .... ')
    n_runs = config_data['n_runs'] # number of mles to compute in main loop
    n_samples = config_data['n_samples'] # samples by run
    n_workers = config_data['n_workers'] # number of workers to choose for parallel mle
    save_mle_out = config_data['save_mle_out']
    mle_out_path = config_data['mle_out_path']
    param_bounds = config_data['param_bounds']
    param_bounds_epsilon = config_data['param_bounds_epsilon']
    parameter_names = config_data['param_names']
    param_varies = config_data['param_varies']
    param_is_boundary_param = config_data['param_is_boundary_param']
    meta_data_file_path = config_data['meta_data_file_path']
    n_subjects = config_data['n_subjects']
    boundary = eval(config_data['boundary'])
    boundary_multiplicative = config_data['boundary_multiplicative']
    network_trained_on_log = config_data['network_trained_on_log']
    
    # optimizer properties:
    de_optim_popsize = config_data['de_optim_popsize']
    
    # NOTE PARAMETERS: 
    # WEIBULL: [v, a, w, node, shape, scale]
    # LINEAR COLLAPSE: [v, a, w, node, theta]
    # DDM: [v, a, w]
    # FULL_DDM: [v, a, w, dw, sdv]
    # LBA: [v_0, ..., v_n, A, b, s]
    
    print('Finishing up initialization .... ')

    # MAKE COLUMNS FOR OPTIMIZER RESULT TABLE ---------
    
    if n_subjects == 1:
        p_sim = []
        p_mle = []

        for parameter_name in parameter_names:
            p_sim.append(parameter_name + '_sim')
            p_mle.append(parameter_name + '_mle')

        my_optim_columns = p_sim + p_mle + ['n_samples']

        # Initialize the data frame in which to store optimizer results
        optim_results = pd.DataFrame(np.zeros((n_runs, len(my_optim_columns))), columns = my_optim_columns)
        optim_results.iloc[:, 2 * len(parameter_names)] = n_samples
        
    else:
        # get adjusted parameter vector which takes into account multiple subjects in parameter space
        parameter_names_adj, parameter_bounds_adj, parameter_bounds_epsilon_adj = adjust_params_names_group(
                                                                              params = parameter_names,
                                                                              params_bounds = param_bounds,
                                                                              params_bounds_epsilon = param_bounds_epsilon,
                                                                              param_varies = param_varies,
                                                                              n_subjects = n_subjects)

        p_sim = []
        p_mle = []

        for parameter_name in parameter_names_adj:
            p_sim.append(parameter_name + '_sim')
            p_mle.append(parameter_name + '_mle')

        my_optim_columns = p_sim + p_mle + ['n_samples']

        # Initialize the data frame in which to store optimizer results
        optim_results = pd.DataFrame(np.zeros((n_runs, len(my_optim_columns))), columns = my_optim_columns)
        optim_results.iloc[:, 2 * len(parameter_names_adj)] = n_samples
    
    # -----------------------------------------------------
    
    print('Start of MLE procedure .... ')
    # Main loop -------------------------------------------------------------
    for i in range(0, n_runs, 1): 

        # Get start time
        start_time = time.time()
        
        if n_subjects == 1:
            # Make dataset
            data, tmp_params = make_data(param_bounds = param_bounds, 
                                         param_bounds_epsilon = param_bounds_epsilon,
                                         param_is_boundary_param = param_is_boundary_param, 
                                         param_names = parameter_names)
            
            # Print some info on run
            print('Parameters for run ' + str(i) + ': ')
            print(tmp_params)
            
            # Run optimizer
            out_parallel = differential_evolution(ktnp.log_p, 
                                                  bounds = param_bounds,
                                                  args = (weights, biases, activations, data, network_trained_on_log),
                                                  popsize = de_optim_popsize,
                                                  disp = True, 
                                                  workers = n_workers)
            
            # Store results
            optim_results.iloc[i, :len(parameter_names)] = tmp_params 
            optim_results.iloc[i, len(parameter_names):(2*len(parameter_names))] = out_parallel.x

        else:
            # Make dataset
            print(parameter_names)
            data, tmp_params = make_data_group(param_bounds = parameter_bounds_adj,
                                               param_bounds_epsilon = parameter_bounds_epsilon_adj,
                                               param_is_boundary_param = param_is_boundary_param,
                                               params_ordered = parameter_names,
                                               param_varies = param_varies,
                                               params_names =  parameter_names_adj,
                                               n_subjects = n_subjects)
            
            # Print some info on run
            print('Parameters for run ' + str(i) + ': ')
            print(tmp_params)

            # Run optimizer
            out_parallel = differential_evolution(ktnp.group_log_p, 
                                                  bounds = parameter_bounds_adj,
                                                  args = (weights, 
                                                          biases, 
                                                          activations, 
                                                          data, 
                                                          param_varies, 
                                                          parameter_names, 
                                                          parameter_names_adj,
                                                          network_trained_on_log),
                                                  popsize = de_optim_popsize,
                                                  disp = True, 
                                                  workers = n_workers)
            # Store results
            optim_results.iloc[i, :len(parameter_names_adj)] = tmp_params # KEEP OUTSIDE OF MAKE_DATA CALL
            optim_results.iloc[i, len(parameter_names_adj):(2*len(parameter_names_adj))] = out_parallel.x

        print('Solution vector of current run: ')
        print(out_parallel.x)

        print('The run took: ')
        elapsed = time.time() - start_time
        print(time.strftime("%H:%M:%S", time.gmtime(elapsed)))

    # ----------------------------------------------------------
    if save_mle_out:
        # Save optimization results to file
        optim_results.to_csv(mle_out_path + '/mle_results_' + uuid.uuid1().hex + '.csv')