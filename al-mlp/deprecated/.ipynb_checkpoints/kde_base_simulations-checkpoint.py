# RUN FORMAT: python python_file machine method n_choices n_samples binned(bool) file_id type_of_data n_simulators(optional) eps_correction (optional)

# type_of_data --> 'parameter_recovery' , 'base_simulations'

# Basic python utilities
import numpy as np
import scipy as scp
from scipy.stats import gamma

# Parallelization
import multiprocessing as mp
from  multiprocessing import Process
from  multiprocessing import Pool
import psutil

# System utilities
from datetime import datetime
import time
import os
import pickle
import uuid
import sys

# My own code
import kde_class as kde
import cddm_data_simulation as ds
import boundary_functions as bf

def bin_simulator_output(out = [0, 0],
                         bin_dt = 0.04,
                         n_bins = 0,
                         eps_correction = 1e-7, # min p for a bin
                         params = ['v', 'a', 'w', 'ndt']
                        ): # ['v', 'a', 'w', 'ndt', 'angle']

    # Generate bins
    if n_bins == 0:
        n_bins = int(out[2]['max_t'] / bin_dt)
        bins = np.linspace(0, out[2]['max_t'], n_bins + 1)
    else:    
        bins = np.linspace(0, out[2]['max_t'], n_bins + 1)
    
    counts = []
    cnt = 0
    counts = np.zeros( (n_bins, len(out[2]['possible_choices']) ) )
    counts_size = counts.shape[0] * counts.shape[1]
    
    for choice in out[2]['possible_choices']:
        counts[:, cnt] = np.histogram(out[0][out[1] == choice], bins = bins)[0] / out[2]['n_samples']
        cnt += 1
    
    # Apply correction for empty bins
    n_small = 0
    n_big = 0
    n_small = np.sum(counts < eps_correction)
    n_big = counts_size - n_small 
    
    if eps_correction > 0:
        counts[counts <= eps_correction] = eps_correction
        counts[counts > eps_correction] -= (eps_correction * (n_small / n_big))

    return ([out[2][param] for param in params], # features
            counts, # labels
            {'max_t': out[2]['max_t'], 
             'bin_dt': bin_dt, 
             'n_samples': out[2]['n_samples']} # meta data
           )

def data_generator_ddm(*args):
    simulator_data = dgp(*args)
    return simulator_data

# DEFINE FUNCTIONS THAAT NEED INITIALIZATION DEPENDEND ON CONTEXT ----------------------------------------
def data_generator_ddm_binned(*args):
    simulator_data = dgp(*args)
    features, labels, meta = bin_simulator_output(out = simulator_data,
                                                  bin_dt = bin_dt, 
                                                  n_bins = n_bins,
                                                  eps_correction = 0,
                                                  params = param_names_full) 
    return (features, labels, meta) 
# --------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    
    # INITIALIZATIONS ----------------------------------------------------------------------------------------
    # Get cpu cnt
    n_cpus = psutil.cpu_count(logical = False)

    # Choose simulator and datatype
    machine = sys.argv[1]
    method = sys.argv[2]
    n_choices = int(sys.argv[3])
    n_samples = int(sys.argv[4])
    binned = bool(sys.argv[5])
    file_id = sys.argv[6]
    data_type = sys.argv[7]
    analytic = ('analytic' in method)
    file_signature =  method + '_' + data_type + '_'
    
    n_simulators = 10000 
    if len(sys.argv) > 8:
        n_simulators = int(sys.argv[8])
    
    # Extra params for binned version
    bin_dt = 0.04
    n_bins = 256
    eps_correction = 0
    if binned and (len(sys.argv) > 9):
        eps_correction = float(sys.argv[9]) 
        
    # out file name components
    if machine == 'x7':
        stats = pickle.load(open("/media/data_cifs/afengler/git_repos/nn_likelihoods/kde_stats.pickle", "rb"))
        method_params = stats[method]
        method_folder = method_params['method_folder_x7']

    if machine == 'ccv':
        stats = pickle.load(open("/users/afengler/git_repos/nn_likelihoods/kde_stats.pickle", "rb"))
        method_params = stats[method]
        method_folder = method_params['method_folder']
    
    # Simulator parameters
    dgp = method_params["dgp"]
    
    if method != 'race_model':
        s = 1 # Choose
    else:
        s = np.array([1 for i in range(n_choices)], dtype = np.float32)
        
    delta_t = 0.01 # Choose
    if binned:
        max_t = 10
    else:
        max_t = 20

    print_info = False # Choose
    bound = method_params['boundary']
    boundary_multiplicative = method_params['boundary_multiplicative']
    
    if method != 'lba' and method != 'race_model':
        if data_type == 'base_simulations':
            if binned:
                out_folder = method_folder + data_type + '_' + str(n_samples) + '_binned/'
            else:
                out_folder = method_folder + data_type + '_' + str(n_samples) + '/'
        if data_type == 'parameter_recovery':
            if binned:
                out_folder = method_folder + data_type + '_binned/'
            else:
                out_folder = method_folder + data_type + '/'
    else:
        if binned:
            out_folder = method_folder + data_type + '_' + str(n_samples) + '_' + str(n_choices) + '_binned/'
        else:
            out_folder = method_folder + data_type + '_' + str(n_samples) + '_' + str(n_choices) + '/'

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    # --------------------------------------------------------------------------------------------------------
    
    # GENERATE A SET OF PARAMETERS ---------------------------------------------------------------------------
    process_param_names = method_params['param_names']
    boundary_param_names = method_params['boundary_param_names']
    param_names_full = process_param_names + boundary_param_names
    process_param_upper_bnd = []
    process_param_lower_bnd = []
    
    if n_choices > 2 or method == 'lba':
        process_param_depends_n = method_params['param_depends_on_n_choice']
    
    for i in range(len(process_param_names)):
        process_param_upper_bnd.append(method_params['param_bounds_network'][i][0]) 
        process_param_lower_bnd.append(method_params['param_bounds_network'][i][1])
    
    # Process params:
    if n_choices <= 2 and method != 'lba':
        param_samples = tuple(map(tuple, np.random.uniform(low = process_param_lower_bnd,
                                                           high = process_param_upper_bnd,
                                                           size = (n_simulators, len(process_param_names)))))
        
        #print('passing')
        
    else:
        param_samples = tuple()
        for n in range(n_simulators):
            param_samples_tmp = tuple()
            for i in range(len(process_param_names)):
                if process_param_depends_n[i]:
                    param_samples_tmp += (np.float32(np.random.uniform(low = process_param_lower_bnd[i],
                                                                       high = process_param_upper_bnd[i],
                                                                       size = (n_choices))), )
                else:
                    param_samples_tmp += (np.float32(np.random.uniform(low = process_param_lower_bnd[i],
                                          high = process_param_upper_bnd[i])), )
            param_samples += (param_samples_tmp, )
            #print(param_samples_tmp)
            #if n % 100 == 0:
                #print(n, ' parameter sets sampled')
            
        # Update process param names to account for n_choices
        process_param_tmp = []
        for i in range(len(process_param_names)):
            param_cnt = 0
            if process_param_depends_n[i]:
                for j in range(n_choices):
                    process_param_tmp.append(process_param_names[i] + '_' + str(j))
            else:
                process_param_tmp.append(process_param_names[i])
        param_names_full = process_param_tmp + boundary_param_names
        print('param_names_full ', param_names_full)           

    if len(boundary_param_names) > 0:
        boundary_param_lower_bnd = []
        boundary_param_upper_bnd = []
        
        for i in range(len(boundary_param_names)):
            boundary_param_lower_bnd.append(method_params['boundary_param_bounds_network'][i][0])
            boundary_param_upper_bnd.append(method_params['boundary_param_bounds_network'][i][1])
                                      
        boundary_param_samples = np.random.uniform(low = boundary_param_lower_bnd,
                                                   high = boundary_param_upper_bnd,
                                                   size = (n_simulators, len(boundary_param_names)))
    # --------------------------------------------------------------------------------------------------------
    
    # MAKE SUITABLE FOR PARALLEL SIMULATION ------------------------------------------------------------------
    args_list = []
    for i in range(n_simulators):
        # Get current set of parameters
        process_params = param_samples[i]
        print(param_samples[i])
        sampler_params = (s, delta_t, max_t, n_samples, print_info, bound, boundary_multiplicative)
        
        if len(boundary_param_names) > 0:
            boundary_params = (dict(zip(boundary_param_names, boundary_param_samples[i])) ,)
            #print('passed thorugh')
        else:
            boundary_params = ({},)
        
        # Append argument list with current parameters
        args_tmp = process_params + sampler_params + boundary_params
        #sprint(args_tmp)
        args_list.append(args_tmp)
    # --------------------------------------------------------------------------------------------------------
          
    # RUN SIMULATIONS AND STORE DATA -------------------------------------------------------------------------
    
    # BINNED VERSION
    if binned:
        # Parallel Loop
        with Pool(processes = n_cpus) as pool:
            res = pool.starmap(data_generator_ddm_binned, args_list)

        features = []
        labels = []
        for i in range(len(res)):
            features.append(res[i][1])
            labels.append(res[i][0])

        features = np.array(features)
        labels = np.array(labels)
        meta = res[0][2]
        
        # Storing files
        pickle.dump((features, labels), open(out_folder + file_signature + str(n_choices) + '_n_' + str(n_samples) + '_' + file_id + '.pickle', 'wb'))
        pickle.dump(meta, open(out_folder + 'meta_' + file_signature + str(n_choices) + '_n_' + str(n_samples) + '.pickle', 'wb'))
    
    # STANDARD OUTPUT
    else:
        # Parallel Loop
        with Pool(processes = n_cpus) as pool:
            res = pool.starmap(data_generator_ddm, args_list)
            pickle.dump(res, open(out_folder + file_signature + 'n_' + str(n_samples) + '_' + file_id + '.pickle', 'wb'))
    # --------------------------------------------------------------------------------------------------------
    print('finished')
       