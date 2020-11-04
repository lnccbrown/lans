# Load packages
import tensorflow as tf
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
from scipy.optimize import minimize
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load my own functions
import dnnregressor_train_eval_keras as dnnk
import make_data_wfpt as mdw
from kde_training_utilities import kde_load_data
import ddm_data_simulation as ddm_sim
import boundary_functions as bf

# Define the likelihood function
def log_p(params = [0, 1, 0.9], model = [], data = [], parameter_names = []):
    # Make feature array
    feature_array = np.zeros((data[0].shape[0], len(parameter_names) + 2))
    
    # Store parameters
    cnt = 0
    for i in range(0, len(parameter_names), 1):
        feature_array[:, i] = params[i]
        cnt += 1
    
    # Store rts and choices
    feature_array[:, cnt] = data[0].ravel() # rts
    feature_array[:, cnt + 1] = data[1].ravel() # choices
    
    # Get model predictions
    prediction = model.predict(feature_array)
    
    # Some post-processing of predictions
    prediction[prediction < 1e-29] = 1e-29
    
    return(- np.sum(np.log(prediction)))  

def make_params(param_bounds = []):
    params = np.zeros(len(param_bounds))
    
    for i in range(len(params)):
        params[i] = np.random.uniform(low = param_bounds[i][0], high = param_bounds[i][1])
        
    return params
# ---------------------


if __name__ == "__main__":
    
    # CUDA INIT (CAREFUL SPECIFY CPU HERE IF NO GPU AVAILABLE)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]="3"

    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())

    base_path = '/media/data_cifs/afengler/data/kde/full_ddm'
    # SPECIFY MODEL PATH
    model_path = base_path + '/keras_models/dnnregressor_full_ddm_06_28_19_00_48_00/model_0'
    ckpt_path = base_path + '/keras_models/dnnregressor_full_ddm_06_28_19_00_48_00/ckpt_0_10'

    # LOAD MODEL
    model = keras.models.load_model(model_path)
    model.load_weights(ckpt_path)

    # MLE RUN SETUP
    n_runs = 50 # number of mle's to compute
    n_samples = 5000 # number of samples as base
    feature_file_path = base_path + '/train_test_data/test_features.pickle' 
    mle_out_path = base_path + '/mle_runs'

    # PARAMETERS: DDM-CONSTANT: [v, a, w]
    # PARAMETERS: DDM-WEIBULL: [v, a, w, node, shape, scale]
    # PARAMETERS: FULL-DDM-CONSTANT [v, a, w, dw, sdv]
    # PARAMETERS: ORNSTEIN-UHLENBECK
    
    # SPECIFY PARAMETER BOUNDS FOR GENETIC ALGORITHM
    param_bounds = [(-2, 2), (0.5, 2), (0.3, 0.7), (0.0, 0.1), (0.0, 0.5)]
    
    # SPECIFY BOUNDARY USED
    boundary = bf.constant
    boundary_multiplicative = True
    
    # Get parameter names in correct ordering:
    dat = pickle.load(open(feature_file_path, 
                           'rb'))

    parameter_names = list(dat.keys())[:-2] # :-1 to get rid of 'rt' and 'choice' here

    # Make columns for optimizer result table
    p_sim = []
    p_mle = []

    for parameter_name in parameter_names:
        p_sim.append(parameter_name + '_sim')
        p_mle.append(parameter_name + '_mle')

    my_optim_columns = p_sim + p_mle + ['n_samples']

    # Initialize the data frame in which to store optimizer results
    optim_results = pd.DataFrame(np.zeros((n_runs, len(my_optim_columns))), columns = my_optim_columns)
    optim_results.iloc[:, 2 * len(parameter_names)] = n_samples

    # Main loop ----------- TD: Parallelize
    for i in range(n_runs): 

        # Get start time
        start_time = time.time()

        tmp_params = make_params(param_bounds = param_bounds)

        # Store in output file
        optim_results.iloc[i, :len(parameter_names)] = tmp_params

        # Print some info on run
        print('Parameters for run ' + str(i) + ': ')
        print(tmp_params)

        # Define boundary params
        # Linear Collapse
        #boundary_params = {'node': tmp_params[3],
        #                   'theta': tmp_params[4]}

        # Constant
        boundary_params = {}

        # Run model simulations
        ddm_dat_tmp = ddm_sim.full_ddm(v = tmp_params[0],
                                       a = tmp_params[1],
                                       w = tmp_params[2],
			               dw = tmp_params[3],
				       sdv = tmp_params[4],
                                       s = 1,
                                       delta_t = 0.001,
                                       max_t = 20,
                                       n_samples = n_samples,
                                       boundary_fun = boundary, # function of t (and potentially other parameters) that takes in (t, *args)
                                       boundary_multiplicative = boundary_multiplicative, # CAREFUL: CHECK IF BOUND
                                       boundary_params = boundary_params)

        # Print some info on run
        print('Mean rt for current run: ')
        print(np.mean(ddm_dat_tmp[0]))

        # Run optimizer
        out = differential_evolution(log_p, 
                                     bounds = param_bounds, 
                                     args = (model, ddm_dat_tmp, parameter_names), 
                                     popsize = 40,
                                     polish = False,
                                     disp = True,
				     maxiter = 100,
				     )

        # Print some info
        print('Solution vector of current run: ')
        print(out.x)

        print('The run took: ')
        elapsed_time = time.time() - start_time
        print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

        # Store result in output dataframe
        optim_results.iloc[i, len(parameter_names):(2*len(parameter_names))] = out.x
    # -----------------------

    # Save optimization results to file
    optim_results.to_csv(mle_out_path + '/mle_results_' + uuid.uuid1().hex + '.csv')
