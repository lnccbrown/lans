# IMPORTS --------------------------------------------------------------------
# We are not importing tensorflow or keras here
import os
import time
import re
#os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'

from numpy import ndarray
import numpy as np
import yaml
import pandas as pd
from itertools import product
import multiprocessing as mp
import pickle
import uuid

import sys
import argparse
import scipy as scp
from scipy.optimize import differential_evolution

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.models import load_model

# Sampler
from samplers import SliceSampler
from samplers import DifferentialEvolutionSequential

# Analytical Likelihood for ddm
from cdwiener import batch_fptd

# Analytical Likelihood for lba
#import clba

# Network converter
#import keras_to_numpy as ktnp
import ckeras_to_numpy as ktnp

#import keras_to_numpy_class as ktnpc

# Tensorflow 

# -----------------------------------------------------------------------------

# SUPPORT FUNCTIONS -----------------------------------------------------------
# Get full parameter vector including bounds
def make_parameter_bounds_for_sampler(method_params = []):
    
    param_bounds = method_params['param_bounds_network'] + method_params['boundary_param_bounds_network']

    # If model is lba, lca, race we need to expand parameter boundaries to account for
    # parameters that depend on the number of choices
    if method == 'lba' or method == 'lca' or method == 'race':
        param_depends_on_n = method_params['param_depends_on_n_choice']
        param_bounds_tmp = []

        n_process_params = len(method_params['param_names'])

        p_cnt = 0
        for i in range(n_process_params):
            if method_params['param_depends_on_n_choice'][i]:
                for c in range(method_params['n_choices']):
                    param_bounds_tmp.append(param_bounds[i])
                    p_cnt += 1
            else:
                param_bounds_tmp.append(param_bounds[i])
                p_cnt += 1

        param_bounds_tmp += param_bounds[n_process_params:]
        return np.array(param_bounds_tmp)
    else: 
        return np.array(param_bounds)
# -----------------------------------------------------------------------------
  
# INITIALIZATIONS -------------------------------------------------------------
if __name__ == "__main__":
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--machine",
                     type = str,
                     default = 'x7')
    CLI.add_argument("--method",
                     type = str,
                     default = 'ddm')
    CLI.add_argument("--datatype",
                     type = str,
                     default = 'parameter_recovery') # real, parameter_recovery, perturbation experiment
    CLI.add_argument("--nsamples",
                     type = int,
                     default = 1000)
    CLI.add_argument("--nmcmcsamples",
                     type = int,
                     default = 10000)
    CLI.add_argument("--sampler",
                    type = str,
                    default = 'slice')
    CLI.add_argument("--outfileid",
                     type = str,
                     default = 'TEST')
    CLI.add_argument("--infilesignature",
                     type = str,
                     default = '')
    CLI.add_argument("--outfilesignature",
                     type = str,
                     default = 'signature')
    CLI.add_argument("--nchoices",
                     type = int,
                     default = 2)
    CLI.add_argument("--activedims",
                     nargs = "*",
                     type = int,
                     default = [0, 1, 2, 3])
    CLI.add_argument("--frozendims",
                     nargs = "*",
                     type = int,
                     default = [])
    CLI.add_argument("--frozendimsinit",
                     nargs = '*',
                     type = float,
                     default = [])
    CLI.add_argument("--samplerinit",
                     type = str,
                     default = 'mle') # 'mle', 'random', 'true'
    CLI.add_argument("--nbyarrayjob",
                     type = int,
                     default = 10)
    CLI.add_argument("--ncpus",
                     type = int,
                     default = 10)
    CLI.add_argument("--nnbatchid",  # nnbatchid is used if we use the '_batch' parts of the model_path files (essentially to for pposterior sample runs that check if for the same model across networks we observe similar behavior)
                     type = int,
                     default = -1)
    CLI.add_argument("--analytic",
                     type = int,
                     default = 0)
    CLI.add_argument("--modelidentifier",
                     type = str,
                     default = None)
    
    args = CLI.parse_args()
    print(args)
    
    #mode = args.boundmode
    machine = args.machine
    method = args.method
    #analytic = ('analytic' in method)
    sampler = args.sampler
    data_type = args.datatype
    n_samples = args.nsamples
    nmcmcsamples = args.nmcmcsamples
    infilesignature = args.infilesignature
    if infilesignature == None or infilesignature == 'None':
        infilesignature = ''
    
    outfileid = args.outfileid
    outfilesignature = args.outfilesignature
    
    if outfilesignature == None or outfilesignature == 'None':
        outfilesignature = ''
    
    n_cpus = args.ncpus
    n_by_arrayjob = args.nbyarrayjob
    nnbatchid = args.nnbatchid
    analytic = args.analytic
    samplerinit = args.samplerinit
    
    if args.modelidentifier == None or args.modelidentifier == 'None':
        modelidentifier = ''
    else:
        modelidentifier = args.modelidentifier
    
    if machine == 'x7':
        os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = '3'
        import tensorflow as tf
        from tensorflow import keras

        if tf.__version__[0] == '2':
            print('DISABLING EAGER EXECUTION')
            tf.compat.v1.disable_eager_execution()

        print('Tensorflow version: ', tf.__version__)
        tf.test.is_gpu_available()
    else:
        import tensorflow as tf
        from tensorflow import keras

        if tf.__version__[0] == '2':
            print('DISABLING EAGER EXECUTION')
            tf.compat.v1.disable_eager_execution()

        print('Tensorflow version: ', tf.__version__)
        tf.test.is_gpu_available()
        
    global keras_model

    # Initialize the frozen dimensions
    if len(args.frozendims) >= 1:
        frozen_dims = [[args.frozendims[i], args.frozendimsinit[i]] for i in range(len(args.frozendims))]
        active_dims = args.activedims
    else:
        active_dims = 'all'
        frozen_dims = 'none'
   
    if machine == 'x7':
        method_params = pickle.load(open("/media/data_cifs/afengler/git_repos/nn_likelihoods/kde_stats.pickle", "rb"))[method]
        output_folder = method_params['output_folder_x7']
        method_folder = method_params['method_folder_x7']
        
        with open("model_paths_x7.yaml") as tmp_file:
            if nnbatchid == -1:
                network_path = yaml.load(tmp_file)[method + modelidentifier]
                network_id = network_path[list(re.finditer('/', network_path))[-2].end():]
            else:
                network_path = yaml.load(tmp_file)[method + '_batch' + modelidentifier][nnbatchid]
                network_id = network_path[list(re.finditer('/', network_path))[-2].end():]

            print('Loading network from: ')
            print(network_path)
            # model = load_model(network_path + 'model_final.h5', custom_objects = {"huber_loss": tf.losses.huber_loss})
            keras_model = keras.models.load_model(network_path + '/model_final.h5', compile = False)

    if machine == 'ccv':
        method_params = pickle.load(open("/users/afengler/git_repos/nn_likelihoods/kde_stats.pickle", "rb"))[method]
        output_folder = method_params['output_folder']
        method_folder = method_params['method_folder']
        
        with open("model_paths.yaml") as tmp_file:
            if nnbatchid == -1:
                network_path = yaml.load(tmp_file)[method + modelidentifier]
                network_id = network_path[list(re.finditer('/', network_path))[-2].end():]

            else:
                network_path = yaml.load(tmp_file)[method + '_batch' + modelidentifier][nnbatchid]
                network_id = network_path[list(re.finditer('/', network_path))[-2].end():]
                
            print('Loading network from: ')
            print(network_path)
            
            keras_model = keras.models.load_model(network_path + '/model_final.h5', compile = False)
            
    if machine == 'home':
        method_params = pickle.load(open("/users/afengler/OneDrive/git_repos/nn_likelihoods/kde_stats.pickle", "rb"))[method]
        output_folder = method_params['output_folder_home']
        method_folder = method_params['method_folder_home']
        
        with open("model_paths_home.yaml") as tmp_file:
            if nnbatchid == -1:
                network_path = yaml.load(tmp_file)[method + modelidentifier]
                network_id = network_path[list(re.finditer('/', network_path))[-2].end():]

            else:
                network_path = yaml.load(tmp_file)[method + '_batch' + modelidentifier][nnbatchid]
                network_id = network_path[list(re.finditer('/', network_path))[-2].end():]
                
            print('Loading network from: ')
            print(network_path)
            
#           Global keras_model
            keras_model = keras.models.load_model(network_path + '/model_final.h5', compile = False)
            
    if data_type == 'parameter_recovery':
        file_ = 'parameter_recovery_data_binned_0_nbins_0_n_' + str(n_samples) + '/' + infilesignature + method + \
                '_nchoices_2_parameter_recovery_binned_0_nbins_0_nreps_1_n_' + str(n_samples) + '.pickle'
        
        if analytic:
            if not os.path.exists(output_folder + 'analytic'):
                os.makedirs(output_folder + 'analytic')
        else:  
            if not os.path.exists(output_folder + network_id):
                os.makedirs(output_folder + network_id)
        
        outfilesignature = outfilesignature + 'post_samp_data_param_recov_unif_reps_1_n_' + \
                             str(n_samples) + '_init_' + samplerinit + '_' + infilesignature
    
    if data_type == 'real':                                                                        
        file_ = args.infilesignature
        if machine == 'x7':
            data_folder = '/media/data_cifs/afengler/data/real/'
        if machine == 'ccv':
            data_folder = '/users/afengler/data/real/'
     
    method_params['n_choices'] = args.nchoices
    print('METHOD PARAMETERS: \n')
    print(method_params)

    # Load weights, biases and activations of current network --------
    if analytic:
        pass
    else:
        pass
#         with open(network_path + "weights.pickle", "rb") as tmp_file:
#             weights = pickle.load(tmp_file)
#             #print(weights)
#             for weight in weights:
#                 print(weight.shape)
#         with open(network_path + 'biases.pickle', 'rb') as tmp_file:
#             biases = pickle.load(tmp_file)
#             #print(biases)
#         with open(network_path + 'activations.pickle', 'rb') as tmp_file:
#             activations = pickle.load(tmp_file)
#             #print(activations)
#         n_layers = int(len(weights))
# ----------------------------------------------------------------

# DEFINE TARGET LIKELIHOODS FOR CORRESPONDING MODELS -------------------------------------------------
 
# ----------------------------------------------------------------------------------------------------

# MAKE PARAMETER / DATA GRID -------------------------------------------------------------------------
    # REFORMULATE param bounds
    
    if data_type == 'real':
        print(data_folder + file_)
        data = pickle.load(open(data_folder + file_ , 'rb'))
        data_grid = data[0]
    elif data_type == 'parameter_recovery':
        print('We are reading in: ', method_folder + file_)
        data = pickle.load(open(method_folder + file_ , 'rb'))
        param_grid = data[0]
        print('param grid')
        print(param_grid)
        print(param_grid.shape)
        data_grid = np.squeeze(data[1], axis = 0)

        # subset data according to array id so that we  run the sampler only for those datasets
        data_grid = data_grid[((int(outfileid) - 1) * n_by_arrayjob) : (int(outfileid) * n_by_arrayjob), :, :]
        param_grid = param_grid[((int(outfileid) - 1) * n_by_arrayjob) : (int(outfileid) * n_by_arrayjob), :]
    else:
        print('Unknown Datatype, results will likely not make sense')   
    
    # Sampler initialization

    n_sampler_runs = data_grid.shape[0]

    if samplerinit == 'random':
        init_grid = ['random' for i in range(n_sampler_runs)]
    elif samplerinit == 'true':
        if not (data_type == 'parameter_recovery' or data_type == 'perturbation_experiment'):
            print('You cannot initialize true parameters if we are dealing with real data....')
        init_grid = data[0]
    elif samplerinit == 'mle':
        init_grid = ['mle' for i in range(n_sampler_runs)]
    
    # Parameter bounds to pass to sampler    
    sampler_param_bounds = make_parameter_bounds_for_sampler(method_params = method_params)

    # Apply epsilon correction
    epsilon_bound_correction = 0.005
    sampler_param_bounds[:, 0] = sampler_param_bounds[:, 0] + epsilon_bound_correction
    sampler_param_bounds[:, 1] = sampler_param_bounds[:, 1] - epsilon_bound_correction

    sampler_param_bounds = [sampler_param_bounds for i in range(n_by_arrayjob)]
    
    print('sampler_param_bounds: ' , sampler_param_bounds)
    print('shape sampler param bounds: ', sampler_param_bounds[0].shape)
    #print('active dims: ', active_dims)
    #print('frozen_dims: ', frozen_dims)
    print('param_grid: ', param_grid)
    print('shape of param_grid:', len(param_grid))
    print('shape of data_grid:', data_grid.shape)
# ----------------------------------------------------------------------------------------------------

# RUN POSTERIOR SIMULATIONS --------------------------------------------------------------------------
   # MLP TARGET
    n_params = sampler_param_bounds[0].shape[0]
    
#     if not analytic:
#         mlpt = ktnpc.mlp_target(weights = weights, biases = biases, activations = activations, n_datapoints = data_grid.shape[1])
    
    # Can probably cache this function with good defaults...
    def mlp_target(params, 
                   data, 
                   ll_min = -16.11809 # corresponds to 1e-7
                   ): 
        
        mlp_input_batch = np.zeros((data_grid.shape[1], n_params + 2), dtype = np.float32)
        mlp_input_batch[:, :n_params] = params
        mlp_input_batch[:, n_params:] = data
        #return np.sum(np.core.umath.maximum(ktnp.predict(mlp_input_batch, weights, biases, activations, n_layers), ll_min))
        return np.sum(np.core.umath.maximum(keras_model.predict_on_batch(mlp_input_batch), ll_min))

    # NAVARRO FUSS (DDM)
    if 'sdv' in method:
        def nf_target(params, data, likelihood_min = 1e-10):
            return np.sum(np.maximum(np.log(batch_fptd(data[:, 0] * data[:, 1] * (- 1),
                                                       params[0],
                                                       params[1] * 2, 
                                                       params[2],
                                                       params[3],
                                                       params[4])),
                                                       np.log(likelihood_min)))
    else:
        def nf_target(params, data, likelihood_min = 1e-10):
            return np.sum(np.maximum(np.log(batch_fptd(data[:, 0] * data[:, 1] * (- 1),
                                                       params[0],
                                                       params[1] * 2, 
                                                       params[2],
                                                       params[3])),
                                                       np.log(likelihood_min)))

    # Define posterior samplers for respective likelihood functions
    def mlp_posterior(args): # args = (data, true_params)
        scp.random.seed()
        if sampler == 'slice':
            model = SliceSampler(bounds = args[2], 
                                 target = mlp_target, 
                                 w = .4 / 1024, #w = .4 / 1024, 
                                 p = 8,
                                 print_interval = 100)
            
            model.sample(data = args[0],
                         min_samples = nmcmcsamples,
                         max_samples = 10000,
                         init = args[1],
                         active_dims = active_dims,
                         frozen_dim_vals = frozen_dims)
            
            return (model.samples, model.lp, -1, model.sample_time, model.optim_time)

        if sampler == 'diffevo':
            model = DifferentialEvolutionSequential(bounds = args[2],
                                                    target = mlp_target,
                                                    mode_switch_p = 0.1,
                                                    gamma = 'auto',
                                                    crp = 0.3)
        
            (samples, lps, gelman_rubin_r_hat, sample_time, optim_time) = model.sample(data = args[0],
                                                                                       max_samples = nmcmcsamples,
                                                                                       min_samples = 2000,
                                                                                       n_burn_in = 1000,
                                                                                       init = args[1],
                                                                                       active_dims = active_dims,
                                                                                       frozen_dim_vals = frozen_dims,
                                                                                       gelman_rubin_force_stop = True)
            return (samples, lps, gelman_rubin_r_hat, sample_time, optim_time) # random_seed) # random seed was just to check that we are not passing the same everytime

    # Test navarro-fuss
    def nf_posterior(args): # TODO add active and frozen dim vals
        scp.random.seed()
        
        if sampler == 'slice':
            model = SliceSampler(bounds = args[2], 
                                 target = nf_target, 
                                 w = .4 / 1024, #w = .4 / 1024, 
                                 p = 8,
                                 print_interval = 100)
            
            model.sample(data = args[0],
                         min_samples = nmcmcsamples,
                         max_samples = 10000,
                         init = args[1],
                         active_dims = active_dims,
                         frozen_dim_vals = frozen_dims)
            
            return (model.samples, model.lp, -1, model.sample_time, model.optim_time)
            
        if sampler == 'diffevo':
            model = DifferentialEvolutionSequential(bounds = args[2],
                                                    target = nf_target,
                                                    mode_switch_p = 0.1,
                                                    gamma = 'auto',
                                                    crp = 0.3)
        
            (samples, lps, gelman_rubin_r_hat, sample_time, optim_time) = model.sample(data = args[0],
                                                                                       max_samples = nmcmcsamples,
                                                                                       min_samples = 2000,
                                                                                       n_burn_in = 1000,
                                                                                       init = args[1],
                                                                                       active_dims = active_dims,
                                                                                       frozen_dim_vals = frozen_dims,
                                                                                       gelman_rubin_force_stop = True)
       
            return (samples, lps, gelman_rubin_r_hat, sample_time, optim_time) # random_seed)

    # Make available the specified amount of cpus
    if n_cpus == 'all':
        p = mp.Pool(mp.cpu_count())
    else: 
        p = mp.Pool(n_cpus)

    # Subset parameter and data grid
    timings = []
    # Run the sampler with correct target as specified above
    if n_cpus != 1:
        if method == 'lba_analytic':
            posterior_samples = np.array(p.map(lba_posterior, zip(data_grid,
                                                                  init_grid,
                                                                  sampler_param_bounds)))
        elif analytic and 'ddm' in method:
            posterior_samples = p.map(nf_posterior, zip(data_grid, 
                                                        init_grid,
                                                        sampler_param_bounds))
            
        else:
            posterior_samples = p.map(mlp_posterior, zip(data_grid, 
                                                         init_grid, 
                                                         sampler_param_bounds))
    else:
        posterior_samples = ()
        for i in range(n_by_arrayjob):
            start_time = time.time()
            print('Starting job: ', i)
            print('Ground truth parameters for job: ', param_grid[i, :])
            
            if analytic and 'ddm' in method:
                posterior_samples += ((nf_posterior((data_grid[i],
                                                     init_grid[i],
                                                     sampler_param_bounds[i]))), )
            else:
                posterior_samples += ((mlp_posterior((data_grid[i],
                                                      init_grid[i],
                                                      sampler_param_bounds[i]))), )
    
            end_time = time.time()
            exec_time = end_time - start_time
            timings.append(exec_time)
            print('Execution Time: ', exec_time)
    
    # Store files
    print('saving to file')
    if analytic:
        pickle.dump((param_grid, data_grid, posterior_samples, np.array(timings)),
                    open(output_folder + 'analytic/' + outfilesignature + '_' + outfileid + '.pickle', 'wb'))
        print(output_folder +  outfilesignature + '_' + outfileid + ".pickle")
    else:
        print(output_folder + network_id + outfilesignature + '_' + outfileid + ".pickle")
        pickle.dump((param_grid, data_grid, posterior_samples, np.array(timings)), 
                    open(output_folder + network_id + outfilesignature + '_' + outfileid + ".pickle", "wb"))

# ----------------------------------------------------------
# # LBA ANALYTIC 
# def lba_target(params, data): # TODO add active and frozen dim vals
#     return clba.batch_dlba2(rt = data[:, 0],
#                             choice = data[:, 1],
#                             v = params[:2],
#                             A = params[2],
#                             b = params[3], 
#                             s = params[4],
#                             ndt = params[5])
# def lba_posterior(args):
#     scp.random.seed()
#     model = SliceSampler(bounds = args[2],
#                             target = lba_target,
#                             w = .4 / 1024,
#                             p = 8)
    
#     model.sample(args[0], max_samples = nmcmcsamples, init = args[1])
#     return model.samples
# ----------------------------------------------------------