# Import central packages
from tensorflow import keras
from tensorflow.python.client import device_lib
import numpy as np
import pandas as pd

# Import basic tools
import os
import re
import time
from datetime import datetime
import pickle
import yaml
import sys
import argparse

# My own functions
import dataset_generator as dg

def heteroscedastic_loss(true, pred):
    params = pred.shape[1] // 2
    point = pred[:, :params]
    var = pred[:, params:]
    precision = 1 / var
    return keras.backend.sum((precision * ((true - point) ** 2)) + keras.backend.log(var), - 1)

def make_fcn(n_params = 5,
             input_dims = (None, 2),
             conv_layers = [64, 64, 128, 128, 128], 
             kernel_sizes = [1, 3, 3, 3, 3], 
             strides = [1, 2, 2, 2, 2], 
             activations = ["relu", "relu", "relu", "relu", "relu"]):
    
    # Input layer
    inp = keras.Input(shape = input_dims)
    x = inp
    for layer in range(len(conv_layers)):
        x = keras.layers.Conv1D(conv_layers[layer], 
                                kernel_size = kernel_sizes[layer], 
                                strides = strides[layer], 
                                activation = activations[layer])(x)
    # Pooling layer
    x = keras.layers.GlobalAveragePooling1D()(x)
    
    # Final Layer 
    mean = keras.layers.Dense(n_params)(x)
    var = keras.layers.Dense(n_params, activation = "softplus")(x)
    out = keras.layers.Concatenate()([mean, var])
    model = keras.Model(inp, out)
    return model

# INITIALIZATIONS -------------------------------------------------------------
if __name__ == "__main__":
    # Build comand line interface -------
    CLI = argparse.ArgumentParser()
    
    CLI.add_argument("--machine",
                     type = str,
                     default = 'x7')
    CLI.add_argument("--dgp",
                     type = str,
                     default = 'x7')
    CLI.add_argument("--nparameters",
                     type = int,
                     default = 500000)
    CLI.add_argument("--nsamples",
                     type = int,
                     default = 10000)
    CLI.add_argument("--maxt",
                     type = float,
                     default = 20.0)
    CLI.add_argument("--mode",
                     type = str,
                     default = 'train')
    CLI.add_argument("--gpu",
                     type = str,
                     default = '1')
                     
    args = CLI.parse_args()
    # ------------------------------------
    
    # General inits ------------------------------------------------------------------------------
    machine = args.machine
    if machine == 'x7':
        folder_str = "/media/data_cifs/afengler/"
                                           
    if machine == 'ccv':
        folder_str = "/users/afengler/"
    
    data_generator_config = yaml.load(open(folder_str + "git_repos/nn_likelihoods/config_files/" + \
                                           "config_data_generator.yaml"))
        
    data_generator_config['n_parameter_sets'] = args.nparameters
    data_generator_config['n_samples'] = args.nsamples
    data_generator_config['method'] = args.dgp
    data_generator_config['n_reps'] = 1
    data_generator_config['mode'] = args.mode
    data_generator_config['binned'] = 0
    max_t = args.maxt 
    
    timestamp = datetime.now().strftime('%m_%d_%y_%H_%M_%S')

    # set up gpu to use
    if machine == 'x7':
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        print(device_lib.list_local_devices())
    
    dnn_params = yaml.load(open(folder_str + "git_repos/nn_likelihoods/hyperparameters.yaml"))                                   
    method_params = pickle.load(open(folder_str + "git_repos/nn_likelihoods/kde_stats.pickle", "rb"))[args.dgp]
                                           
    # Specify final model path
    if machine == 'x7':
        model_path_x7 = method_params['model_folder_x7'] + "deep_inference_" + str(args.nsamples) + "_" + timestamp
        if not os.path.exists(model_path_x7):
            os.makedirs(model_path_x7)
        
        model_paths_x7 = yaml.load(open(folder_str + 'git_repos/nn_likelihoods/model_paths_x7.yaml'))
        model_paths_x7['fcn_' + args.dgp + '_' + str(args.nsamples)] = model_path_x7
        yaml.dump(model_paths_x7, open(folder_str + 'git_repos/nn_likelihoods/model_paths_x7.yaml', "w"))
    
    if machine == 'ccv':
        model_path_ccv = method_params['model_folder'] + "deep_inference_" + str(args.nsamples) + "_" + timestamp
        if not os.path.exists(model_path_ccv):
            os.makedirs(model_path_ccv)
        
        model_paths_ccv = yaml.load(open(folder_str + 'git_repos/nn_likelihoods/model_paths.yaml'))
        model_paths_ccv['fcn_' + args.dgp + '_' + str(args.nsamples)] = model_path_ccv
        yaml.dump(model_paths_ccv, open(folder_str + 'git_repos/nn_likelihoods/model_paths.yaml', "w"))
    # ------------------------------------------------------------------------------------------------

    # Making training data ---------------------------------------------------------------------------
    print('Making dataset')
    my_dg = dg.data_generator(machine = machine,
                      max_t = max_t,
                      config = data_generator_config)
    param_grid, data_grid = my_dg.make_dataset_uniform(save = False)
    data_grid = np.squeeze(data_grid, axis = 0)
    # ------------------------------------------------------------------------------------------------
    #param_grid, boundary_param_grid = generate_param_grid(n_datasets = n_datasets) 
    #data_grid = generate_data_grid(param_grid, boundary_param_grid)

    print('size of datagrid: ', data_grid.shape)

    # MODEL TRAINING AND SAVE ------------------------------------------------------------------------
    # Create keras model structure 
    model = make_fcn(n_params = param_grid.shape[1])

    # Define keras callbacks
    if machine == 'x7':
        ckpt_filename = model_path_x7 + "/model.h5"
        csv_log_filename = model_path_x7 + "/history.csv"
    if machine == 'ccv':
        ckpt_filename = model_path_ccv + "/model.h5"
        csv_log_filename = model_path_ccv + "/history.csv"

    checkpoint = keras.callbacks.ModelCheckpoint(ckpt_filename, 
                                                 monitor = 'val_loss', 
                                                 verbose = 1, 
                                                 save_best_only = False)

    earlystopping = keras.callbacks.EarlyStopping(monitor = 'val_loss', 
                                                  min_delta = 0, 
                                                  verbose = 1, 
                                                  patience = 6)

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', 
                                                  factor = 0.1,
                                                  patience = 3, 
                                                  verbose = 1,
                                                  min_delta = 0.0001,
                                                  min_lr = 0.0000001)

    csv_logger = keras.callbacks.CSVLogger(csv_log_filename)
    
    # Fit model
    model.compile(loss = heteroscedastic_loss, 
                  optimizer = "adam")

    history = model.fit(data_grid, param_grid, 
                        validation_split = .01,
                        batch_size = 32, 
                        epochs = 250, 
                        callbacks = [checkpoint, reduce_lr, earlystopping, csv_logger])

    print(history)
    
    # Saving model
    print('saving model')
    if machine == 'x7':
        model.save(model_path_x7 + "/model_final.h5")
    if machine == 'ccv':
        model.save(model_path_ccv + "/model_final.h5")
    # ---------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------------------------------
# # REFORMULATE param bounds
# def generate_param_grid(n_datasets = 100):
#     param_upper_bnd = []
#     param_lower_bnd = []
#     boundary_param_upper_bnd = [] 
#     boundary_param_lower_bnd = []

#     for p in range(len(method_params['param_names'])):
#         param_upper_bnd.append(method_params['param_bounds_network'][p][1])
#         param_lower_bnd.append(method_params['param_bounds_network'][p][0])

#     if len(method_params['boundary_param_names']) > 0:
#         for p in range(len(method_params['boundary_param_names'])):
#             boundary_param_upper_bnd.append(method_params['boundary_param_bounds_network'][p][1])
#             boundary_param_lower_bnd.append(method_params['boundary_param_bounds_network'][p][0])                                    

#     param_grid = np.random.uniform(low = param_lower_bnd, 
#                                    high = param_upper_bnd, 
#                                    size = (n_datasets, len(method_params['param_names'])))

#     if len(method_params['boundary_param_names']) > 0:
#         boundary_param_grid = np.random.uniform(low = boundary_param_lower_bnd,
#                                                 high = boundary_param_upper_bnd,
#                                                 size = (n_datasets, len(method_params['boundary_param_bounds_network'])))
#     else:
#         boundary_param_grid = []
        
#     return (param_grid, boundary_param_grid)

# def generate_data_grid(param_grid, boundary_param_grid):
#     data_grid = np.zeros((param_grid.shape[0], n_data_samples, 2))
#     for i in range(param_grid.shape[0]):
#         param_dict_tmp = dict(zip(method_params["param_names"], param_grid[i]))
        
#         if len(method_params['boundary_param_names']) > 0:
#             boundary_dict_tmp = dict(zip(method_params["boundary_param_names"], boundary_param_grid[i]))
#         else:
#             boundary_dict_tmp = {}
            
#         rts, choices, _ = method_params["dgp"](**param_dict_tmp, 
#                                                boundary_fun = method_params["boundary"], 
#                                                n_samples = n_data_samples,
#                                                delta_t = 0.01, 
#                                                boundary_params = boundary_dict_tmp,
#                                                boundary_multiplicative = method_params['boundary_multiplicative'])
        
#         data_grid[i] = np.concatenate([rts, choices], axis = 1)
        
#         if i % 100 == 0:
#             print('datasets_generated: ', i)
#     return data_grid
        