# MAIN SCRIPT FOR DATASET GENERATION

# IMPORTS ------------------------------------------------------------------------
import numpy as np
import yaml
import pandas as pd
from itertools import product
import pickle
import uuid
import os
import sys
from datetime import datetime
from scipy.stats import truncnorm
#from tqdm import tqdm

import boundary_functions as bf
import multiprocessing as mp
import cddm_data_simulation as cd
#from cdwiener import batch_fptd
#import clba

# Parallelization
import multiprocessing as mp
from multiprocessing import Process
from multiprocessing import Pool
import psutil
import argparse

# --------------------------------------------------------------------------------

# Data generator class that generates datasets for us ----------------------------
class data_generator():
    def __init__(self,
                 machine = 'x7',
                 file_id = 'id',
                 max_t = 20.0,
                 delta_t = 0.01,
                 config = None):
    # INIT -----------------------------------------
        self.machine = machine
        self.file_id = file_id
        self.config = config
        self.method = self.config['method']
        
        if self.machine == 'x7':  
            self.method_params = pickle.load(open("/media/data_cifs/afengler/" + \
                                                  "git_repos/nn_likelihoods/kde_stats.pickle", "rb"))[self.method]
            self.method_comparison_folder = self.method_params['output_folder_x7']
            self.method_folder = self.method_params['method_folder_x7']

        if self.machine == 'ccv':
            self.method_params = pickle.load(open("/users/afengler/git_repos/" + \
                                                  "nn_likelihoods/kde_stats.pickle", "rb"))[self.method]
            self.method_comparison_folder = self.method_params['output_folder']
            self.method_folder = self.method_params['method_folder']

        if self.machine == 'home':
            self.method_params = pickle.load(open("/Users/afengler/OneDrive/git_repos/" + \
                                                  "nn_likelihoods/kde_stats.pickle", "rb"))[self.method]
            self.method_comparison_folder = self.method_params['output_folder_home']
            self.method_folder = self.method_params['method_folder_home']
            
        if self.machine == 'other': # This doesn't use any extra 
            self.method_params = pickle.load(open('kde_stats.pickle', 'rb'))[self.method]
            if not os.path.exists('data_storage'):
                os.makedirs('data_storage')
                
            print('generated new folder: data_storage. Please update git_ignore if this is not supposed to be committed to repo')
                
            self.method_comparison_folder = 'data_storage/'
            self.method_folder = 'data_storage/' + self.method + '_'
        
        self.dgp_hyperparameters = dict(self.method_params['dgp_hyperparameters'])
        self.dgp_hyperparameters['max_t'] = max_t
        self.dgp_hyperparameters['n_samples'] = self.config['nsamples']
        self.dgp_hyperparameters['delta_t'] = delta_t
   # ----------------------------------------------------
   
    def data_generator(self, *args):
        # Get simulations             
        simulator_output = self.method_params['dgp'](*args)
        
        # Bin simulations if so specified in config
        if self.config['binned']:
            labels = self.bin_simulator_output(out = simulator_output,
                                               bin_dt = self.config['bin_dt'],
                                               nbins = self.config['nbins'])
            return labels
        # Return simulator output as [rts, choices] instead if we don't specify binned
        else:
            return np.concatenate([simulator_output[0], simulator_output[1]], axis = 1)
  
    def bin_simulator_output(self, 
                             out = [0, 0],
                             bin_dt = 0.04,
                             nbins = 0): # ['v', 'a', 'w', 'ndt', 'angle']
        
        # Generate bins
        if nbins == 0:
            nbins = int(out[2]['max_t'] / bin_dt)
            bins = np.zeros(nbins + 1)
            bins[:nbins] = np.linspace(0, out[2]['max_t'], nbins)
            bins[nbins] = np.inf
        else:  
            bins = np.zeros(nbins + 1)
            bins[:nbins] = np.linspace(0, out[2]['max_t'], nbins)
            bins[nbins] = np.inf

        cnt = 0
        counts = np.zeros( (nbins, len(out[2]['possible_choices']) ) )

        for choice in out[2]['possible_choices']:
            counts[:, cnt] = np.histogram(out[0][out[1] == choice], bins = bins)[0] / out[2]['n_samples']
            cnt += 1
        return counts
    
    def zip_dict(self,
                 x = [], 
                 key_vec = ['a', 'b', 'c']):
        return dict(zip(key_vec, x))
    
    def make_args_starmap_ready(self,
                                param_grid = []):
            
        nparamsets = param_grid.shape[0]
        n_params = param_grid.shape[1]
        n_boundary_params = len(self.method_params['boundary_param_names'])
        n_process_params = len(self.method_params["param_names"])
        
        # Boundary parameter touples
        if n_boundary_params > 0:
            boundary_param_tuples = np.apply_along_axis(self.zip_dict, 1,  
                                                        param_grid[:, n_process_params:], 
                                                        self.method_params['boundary_param_names'])
        
        # Process parameters to make suitable for parallel processing
        if self.config['nchoices'] <= 2 and self.method != 'lba' and self.method != 'lca' and self.method != 'race_model':
            process_param_tuples = tuple(map(tuple, param_grid[:, :n_process_params]))
            print('passed through false')
        
        elif self.config['nchoices'] >= 2 and (self.method == 'lba' or self.method == 'lca_3' or self.method == 'lca_4' or self.method == 'race_model_3' or self.method == 'race_model_4'):
            print('passed through correct')
            process_param_tuples = tuple()
            for i in range(param_grid.shape[0]):
                tuple_tmp = tuple()
                cnt = 0
                
                for j in range(len(self.method_params['param_names'])):
                    if self.method_params['param_depends_on_n_choice'][j]:
                        tuple_tmp += (param_grid[i, cnt: (cnt + self.config['nchoices'])], )
                        cnt += self.config['nchoices']
                    else:
                        tuple_tmp += (param_grid[i, cnt], )
                        cnt += 1
                process_param_tuples += (tuple_tmp, )     
                
        # If models is race we want pass noise standarad deviation as an array instead of a single value
        if self.method == 'race_model':
            self.dgp_hyperparameters['s'] = np.float32(np.repeat(self.dgp_hyperparameters['s'], self.config['nchoices']))
        
        # Make final list of tuples of parameters
        args_list = []
        for i in range(nparamsets):
            process_params = process_param_tuples[i]
            
            if n_boundary_params > 0:
                boundary_params = (boundary_param_tuples[i], )
            else:
                boundary_params = ({},)
            
            # N samples
            self.dgp_hyperparameters['n_samples'] = self.nsamples[i]
            
            sampler_hyperparameters = tuple(self.dgp_hyperparameters.values())
            if self.method == 'lba': # TODO change lba sampler to accept boundary params?
                args_list.append(process_params + sampler_hyperparameters)
            else:
                args_list.append(process_params + sampler_hyperparameters + boundary_params)
                #print(self.dgp_hyperparameters)
                #print(process_params + sampler_hyperparameters + boundary_params)
        # print(args_list)
        return args_list
    
    def clean_up_parameters(self):
        
        if self.config['mode'] == 'test':
            param_bounds = self.method_params['param_bounds_sampler'] + self.method_params['boundary_param_bounds_sampler']
        if self.config['mode'] == 'mlp':
            param_bounds = self.method_params['param_bounds_network'] + self.method_params['boundary_param_bounds_network']
        if self.config['mode'] == 'cnn':
            param_bounds = self.method_params['param_bounds_cnn'] + self.method_params['boundary_param_bounds_cnn']
        
        # Epsilon correction of boundaries (to make sure for parameter recovery we don't generate straight at the bound)
        
        eps = 0
        if self.config['datatype'] == 'parameter_recovery' and self.config['mode'] != 'test':
            # TD make eps parameter
            eps = 0.05
            
        print('epsilon correction', eps)

        # If model is lba, lca, race we need to expand parameter boundaries to account for
        # parameters that depend on the number of choices
        if self.method == 'lba' or self.method == 'lca' or self.method == 'race_model':
            param_depends_on_n = self.method_params['param_depends_on_n_choice']
            param_bounds_tmp = []
            
            n_process_params = len(self.method_params['param_names'])
            
            p_cnt = 0
            for i in range(n_process_params):
                if self.method_params['param_depends_on_n_choice'][i]:
                    for c in range(self.config['nchoices']):
                        param_bounds_tmp.append(param_bounds[i])
                        p_cnt += 1
                else:
                    param_bounds_tmp.append(param_bounds[i])
                    p_cnt += 1
            
            self.method_params['n_process_parameters'] = p_cnt
            
            param_bounds_tmp += param_bounds[n_process_params:]
            params_upper_bnd = [bnd[1] - eps for bnd in param_bounds_tmp]
            params_lower_bnd = [bnd[0] + eps for bnd in param_bounds_tmp]
                
            #print(params_lower_bnd)
            
            
        # If our model is not lba, race, lca we use simple procedure 
        else:
            params_upper_bnd = [bnd[1] - eps for bnd in param_bounds]
            params_lower_bnd = [bnd[0] + eps for bnd in param_bounds]
            
        return params_upper_bnd, params_lower_bnd
             
    def make_param_grid_perturbation_experiment(self):

        n_perturbation_levels = len(self.config['perturbation_sizes'][0])
        n_params = len(self.method_params['param_names']) + len(self.method_params['boundary_param_names'])
        
        # Get parameter bounds
        params_upper_bnd, params_lower_bnd = self.clean_up_parameters()
        n_params = len(params_upper_bnd)
        experiment_row_cnt = (n_params * n_perturbation_levels) + 1
                       
        meta_dat = pd.DataFrame(np.zeros((experiment_row_cnt, 2)), 
                                columns = ['param', 'perturbation_level']) 
                       
        param_grid = np.zeros((self.config['n_experiments'], experiment_row_cnt, n_params))
        
        # Make the parameter grids
        for i in range(self.config['n_experiments']):
                       
            # Reinitialize row cnt 
            cnt = 0
            
            # Get base parameters for perturbation experiment i
            param_grid_tmp = np.float32(np.random.uniform(low = params_lower_bnd, 
                                                          high = params_upper_bnd))
                       
            # Store as first row in experiment i
            param_grid[i, cnt, :] = param_grid_tmp

            # Store meta data for experiment (only need to do once --> i == 0)
            if i == 0:
                meta_dat.loc[cnt, :] = [-1, -1]

            cnt += 1
                       
            # Fill in perturbation experiment data i
            for p in range(n_params):
                for l in range(n_perturbation_levels):
                    param_grid_perturbed = param_grid_tmp.copy()
                    if param_grid_tmp[p] > ((params_upper_bnd[p] - params_lower_bnd[p]) / 2):
                        param_grid_perturbed[p] -= self.config['perturbation_sizes'][p][l]
                    else:
                        param_grid_perturbed[p] += self.config['perturbation_sizes'][p][l]

                    param_grid[i, cnt, :] = param_grid_perturbed

                    if i  == 0:
                        meta_dat.loc[cnt, :] = [int(p), int(l)]

                    cnt += 1 
                       
        return (param_grid, meta_dat)
                       
    def make_param_grid_uniform(self,
                                nparamsets = None):
        
        if nparamsets == None:
            nparamsets = self.config['nparamsets']
            
        # Initializations
        params_upper_bnd, params_lower_bnd = self.clean_up_parameters()
        n_params = len(params_upper_bnd)
        param_grid = np.zeros((nparamsets, n_params), dtype = np.float32)
        
        # Generate parameters
        param_grid[:, :] = np.float32(np.random.uniform(low = params_lower_bnd,
                                                        high = params_upper_bnd,
                                                        size = (nparamsets, n_params)))
        return param_grid 
    
    def generate_data_grid_parallel(self,
                                    param_grid = []):
        
        args_list = self.make_args_starmap_ready(param_grid = param_grid)
        
        if self.config['n_cpus'] == 'all':
            n_cpus = psutil.cpu_count(logical = False)
        else:
            n_cpus = self.config['n_cpus']

        # Run Data generation
        with Pool(processes = n_cpus) as pool:
            data_grid = np.array(pool.starmap(self.data_generator, args_list))

        return data_grid   
  
        
    def make_dataset_perturbation_experiment(self,
                                             save = True):
        
        param_grid, meta_dat = self.make_param_grid_perturbation_experiment()
             
        if self.config['binned']:           
            data_grid = np.zeros((self.config['nreps'],
                                  self.config['n_experiments'],
                                  param_grid.shape[1], 
                                  self.config['nbins'],
                                  self.config['nchoices']))         
        
        else:
            data_grid = np.zeros((self.config['nreps'],
                                  self.config['n_experiments'],
                                  param_grid.shape[1], 
                                  self.config['nsamples'],
                                  2)) 

        for experiment in range(self.config['n_experiments']):
            for rep in range(self.config['nreps']):
                data_grid[rep, experiment] = self.generate_data_grid_parallel(param_grid = param_grid[experiment])
                print(experiment, ' experiment data finished')
        
        if save == True:
            print('saving dataset')
            pickle.dump((param_grid, data_grid, meta_dat), open(self.method_comparison_folder + \
                                                                'base_data_perturbation_experiment_nexp_' + \
                                                                str(self.config['n_experiments']) + \
                                                                '_nreps_' + str(self.config['nreps']) + \
                                                                '_n_' + str(self.config['nsamples']) + \
                                                                '_' + self.file_id + '.pickle', 'wb'))
                        
            return 'Dataset completed'
        else:
            return param_grid, data_grid, meta_dat
    
    def make_dataset_parameter_recovery(self,
                                        save = True):
        
        param_grid = self.make_param_grid_uniform()
        self.nsamples = [self.config['nsamples'] for i in range(self.config['nparamsets'])]
        
        if self.config['binned']:
            data_grid = np.zeros((self.config['nreps'],
                                  self.config['nparamsets'],
                                  self.config['nbins'],
                                  self.config['nchoices']))
        else:
            data_grid = np.zeros((self.config['nreps'],
                                  self.config['nparamsets'],
                                  self.config['nsamples'],
                                  2))
        
        for rep in range(self.config['nreps']):
            data_grid[rep] = np.array(self.generate_data_grid_parallel(param_grid = param_grid))
            print(rep, ' repetition finished') 
        
        
        if save:
            training_data_folder = self.method_folder + 'parameter_recovery_data_binned_' + str(int(self.config['binned'])) + \
                                   '_nbins_' + str(self.config['nbins']) + \
                                   '_n_' + str(self.config['nsamples'])
            if not os.path.exists(training_data_folder):
                os.makedirs(training_data_folder)
                
            full_file_name = training_data_folder + '/' + \
                            self.method + \
                            '_nchoices_' + str(self.config['nchoices']) + \
                            '_parameter_recovery_' + \
                            'binned_' + str(int(self.config['binned'])) + \
                            '_nbins_' + str(self.config['nbins']) + \
                            '_nreps_' + str(self.config['nreps']) + \
                            '_n_' + str(self.config['nsamples']) + \
                            '.pickle'
            
            print(full_file_name)
            
            meta = self.dgp_hyperparameters.copy()
            if 'boundary' in meta.keys():
                del meta['boundary']

            pickle.dump((param_grid, data_grid, meta), 
                        open(full_file_name, 'wb'), 
                        protocol = self.config['pickleprotocol'])
            
            return 'Dataset completed'
        else:
            return param_grid, data_grid
    
    def make_dataset_train_network_unif(self,
                                        save = True):

        param_grid = self.make_param_grid_uniform()
        self.nsamples = [self.config['nsamples'] for i in range(self.config['nparamsets'])]

        if self.config['binned']:
            data_grid = np.zeros((self.config['nparamsets'],
                                  self.config['nbins'],
                                  self.config['nchoices']))
        else:
            data_grid = np.zeros((self.config['nparamsets'],
                                  self.config['nsamples'],
                                  2))
        data_grid = np.array(self.generate_data_grid_parallel(param_grid = param_grid))

        if save:
            training_data_folder = self.method_folder + 'training_data_binned_' + str(int(self.config['binned'])) + \
                                   '_nbins_' + str(self.config['nbins']) + \
                                   '_n_' + str(self.config['nsamples'])
            if not os.path.exists(training_data_folder):
                os.makedirs(training_data_folder)
                
            full_file_name = training_data_folder + '/' + \
                            self.method + \
                            '_nchoices_' + str(self.config['nchoices']) + \
                            '_train_data_' + \
                            'binned_' + str(int(self.config['binned'])) + \
                            '_nbins_' + str(self.config['nbins']) + \
                            '_n_' + str(self.config['nsamples']) + \
                            '_' + self.file_id + '.pickle'
            
            print(full_file_name)
            
            meta = self.dgp_hyperparameters.copy()
            if 'boundary' in meta.keys():
                del meta['boundary']

            pickle.dump((param_grid, data_grid, meta), 
                        open(full_file_name, 'wb'), 
                        protocol = self.config['pickleprotocol'])
            return 'Dataset completed'
        else:
            return param_grid, data_grid

    def make_param_grid_hierarchical(self,
                                     ):

        # Initializations
        params_upper_bnd, params_lower_bnd = self.clean_up_parameters()
        nparams = len(params_upper_bnd)
        
        # Initialize global parameters
        params_ranges_half = (np.array(params_upper_bnd) - np.array(params_lower_bnd)) / 2
        
        global_stds = np.random.uniform(low = 0.001,
                                        high = params_ranges_half / 10,
                                        size = (self.config['nparamsets'], nparams))
        global_means = np.random.uniform(low = np.array(params_lower_bnd) + (params_ranges_half / 5),
                                         high = np.array(params_upper_bnd) - (params_ranges_half / 5),
                                         size = (self.config['nparamsets'], nparams))

        # Initialize local parameters (by condition)
        subject_param_grid = np.zeros((self.config['nparamsets'], self.config['nsubjects'], nparams))
        
        for n in range(self.config['nparamsets']):
            for i in range(self.config['nsubjects']):
                a, b = (np.array(params_lower_bnd) - global_means[n]) / global_stds[n], (np.array(params_upper_bnd) - global_means[n]) / global_stds[n]
                subject_param_grid[n, i, :] = np.float32(global_means[n] + truncnorm.rvs(a, b, size = global_stds.shape[1]) * global_stds[n])
                
        return subject_param_grid, global_stds, global_means

    def generate_data_grid_hierarchical_parallel(self, 
                                                 param_grid = []):
        
        nparams = param_grid.shape[2]
        args_list = self.make_args_starmap_ready(param_grid = np.reshape(param_grid, (-1, nparams)))

        if self.config['n_cpus'] == 'all':
            n_cpus = psutil.cpu_count(logical = False)
        else:
            n_cpus = self.config['n_cpus']

        # Run Data generation
        with Pool(processes = n_cpus) as pool:
            data_grid = np.array(pool.starmap(self.data_generator, args_list))

        data_grid = np.reshape(data_grid, (self.config['nparamsets'], self.config['nsubjects'], self.config['nsamples'], self.config['nchoices']))
    
        return data_grid

    def make_dataset_parameter_recovery_hierarchical(self,
                                                     save = True):

        param_grid, global_stds, global_means = self.make_param_grid_hierarchical()
        
        self.nsamples = [self.config['nsamples'] for i in range(self.config['nparamsets'] * self.config['nsubjects'])]
        
        if self.config['binned']:
            data_grid = np.zeros((self.config['nreps'],
                                  self.config['nparamsets'],
                                  self.config['nsubjects'], # add to config
                                  self.config['nbins'],
                                  self.config['nchoices']))
        else:
            data_grid = np.zeros((self.config['nreps'],
                                  self.config['nparamsets'],
                                  self.config['nsubjects'],
                                  self.config['nsamples'],
                                  2))
        
        for rep in range(self.config['nreps']):
            # TD: ADD generate_data_grid_parallel_multisubject
            data_grid[rep] = np.array(self.generate_data_grid_hierarchical_parallel(param_grid = param_grid))
            print(rep, ' repetition finished') 
        
        if save:
            training_data_folder = self.method_folder + 'parameter_recovery_hierarchical_data_binned_' + str(int(self.config['binned'])) + \
                                   '_nbins_' + str(self.config['nbins']) + \
                                   '_n_' + str(self.config['nsamples'])
            if not os.path.exists(training_data_folder):
                os.makedirs(training_data_folder)
            
            print('saving dataset as ', training_data_folder + '/' + \
                                        self.method + \
                                        '_nchoices_' + str(self.config['nchoices']) + \
                                        '_parameter_recovery_hierarchical_' + \
                                        'binned_' + str(int(self.config['binned'])) + \
                                        '_nbins_' + str(self.config['nbins']) + \
                                        '_nreps_' + str(self.config['nreps']) + \
                                        '_n_' + str(self.config['nsamples']) + \
                                        '_nsubj_' + str(self.config['nsubjects']) + \
                                        '.pickle')
            
            meta = self.dgp_hyperparameters.copy()
            if 'boundary' in meta.keys():
                del meta['boundary']

            pickle.dump(([param_grid, global_stds, global_means], data_grid, meta), 
                        open(training_data_folder + '/' + \
                            self.method + \
                            '_nchoices_' + str(self.config['nchoices']) + \
                            '_parameter_recovery_hierarchical_' + \
                            'binned_' + str(int(self.config['binned'])) + \
                            '_nbins_' + str(self.config['nbins']) + \
                            '_nreps_' + str(self.config['nreps']) + \
                            '_n_' + str(self.config['nsamples']) + \
                            '_nsubj_' + str(self.config['nsubjects']) + \
                            '.pickle', 'wb'), 
                        protocol = self.config['pickleprotocol'])
            
            return 'Dataset completed'
        else:
            return ([param_grid, global_stds, global_means], data_grid, meta)

    def make_dataset_r_sim(self,
                           n_sim_bnds = [10000, 100000],
                           save = True):
        
        param_grid = self.make_param_grid_uniform()
        self.nsamples = np.random.uniform(size = self.config['nparamsets'], 
                                          high = n_sim_bnds[1],
                                          low = n_sim_bnds[0])
        
        if self.config['binned']:
            data_grid = np.zeros((self.config['nreps'],
                                  self.config['nparamsets'],
                                  self.config['nbins'],
                                  self.config['nchoices']))
        else:
            return 'random number of simulations is supported under BINNED DATA only for now'
        
        for rep in range(self.config['nreps']):
            data_grid[rep] = np.array(self.generate_data_grid_parallel(param_grid = param_grid))
            print(rep, ' repetition finished')
            
            
        if save == True:
            print('saving dataset')
            pickle.dump((param_grid, data_grid), open(self.method_comparison_folder + \
                                                      'base_data_uniform_r_sim_' + \
                                                      str(n_sim_bnds[0]) + '_' + str(n_sim_bnds[1]) + \
                                                      '_nreps_' + str(self.config['nreps']) + \
                                                      '_' + self.file_id + '.pickle', 'wb'))
            return 'Dataset completed'
        else:
            return param_grid, data_grid, self.nsamples
# ---------------------------------------------------------------------------------------

# Functions outside the data generator class that use it --------------------------------
def make_dataset_r_dgp(dgp_list = ['ddm', 'ornstein', 'angle', 'weibull', 'full_ddm'],
                       machine = 'x7',
                       r_nsamples = True,
                       n_sim_bnds = [100, 100000],
                       file_id = 'TEST',
                       max_t = 10.0,
                       delta_t = 0.001,
                       config = None,
                       save = False):
    """Generates datasets across kinds of simulators
    
    Parameter 
    ---------
    dgp_list : list
        List of simulators that you would like to include (names match the simulators stored in kde_info.py)
    machine : str
        The machine the code is run on (basically changes folder directory for imports, meant to be temporary)
    file_id : str
        Attach an identifier to file name that is stored if 'save' is True (some file name formatting is already applied)
    save : bool
        If true saves outputs to pickle file
    
    Returns
    -------
    list 
        [0] list of arrays of parameters by dgp (n_dgp, nparamsets, n_parameters_given_dgp)
        [1] list of arrays storing sampler outputs as histograms (n_repitions, n_parameters_sets, nbins, nchoices)
        [2] array of model ids (dgp ids)
        [3] dgp_list
        
    """
  
    if config['binned']:
        model_ids = np.random.choice(len(dgp_list),
                                     size = (config['nparamsets'], 1))
        model_ids.sort(axis = 0)
        data_grid_out = []
        param_grid_out = []
        nsamples_out = []
    else:
        return 'Config should specify binned = True for simulations to start'

    for i in range(len(dgp_list)):
        nparamsets = np.sum(model_ids == i)
        
        # Change config to update the sampler
        config['method'] = dgp_list[i]
        # Change config to update the number of parameter sets we want to run for the current sampler
        config['nparamsets'] = nparamsets
        
        # Initialize data_generator class with new properties
        dg_tmp = data_generator(machine = machine,
                                config = config,
                                max_t = max_t,
                                delta_t = delta_t)
        
        # Run the simulator
        if r_nsamples:
            param_grid, data_grid, nsamples = dg_tmp.make_dataset_r_sim(n_sim_bnds = n_sim_bnds,
                                                                        save = False)
        else:
            param_grid, data_grid = dg_tmp.make_dataset_train_network_unif(save = False)
            
        print(data_grid.shape)
        print(param_grid.shape)
        
        # Append results
        data_grid_out.append(data_grid)
        param_grid_out.append(param_grid)
        if r_nsamples:
            nsamples_out.append(nsamples_out)

    if save:
        print('saving dataset')
        if machine == 'x7':
            out_folder = '/media/data_cifs/afengler/data/kde/rdgp/'
        if machine == 'ccv':
            out_folder = '/users/afengler/data/kde/rdgp/'
        if machine == 'home':
            out_folder = '/Users/afengler/OneDrive/project_nn_likelihoods/data/kde/rdgp/'
        if machine == 'other':
            if not os.path.exists('data_storage'):
                os.makedirs('data_storage')
                os.makedirs('data_storage/rdgp')
            
            out_folder = 'data_storage/rdgp/'
            
        out_folder = out_folder + 'training_data_' + str(int(config['binned'])) + \
                     '_nbins_' + str(config['nbins']) + \
                     '_nsimbnds_' + str(config['nsimbnds'][0]) + '_' + str(config['nsimbnds'][1]) +'/'
            
        if not os.path.exists(out_folder):
                os.makedirs(out_folder)
                
        pickle.dump((np.concatenate(data_grid_out, axis = 1), model_ids, param_grid_out, dgp_list),
                                                   open(out_folder + \
                                                   'rdgp_nchoices_' + str(config['nchoices']) + \
                                                   '_nreps_' + str(config['nreps']) + \
                                                   '_nsimbnds_' + str(config['nsimbnds'][0]) + '_' + \
                                                   str(config['nsimbnds'][1]) + \
                                                   '_' + str(config['file_id']) + '.pickle', 'wb'))
        return 'Dataset completed'
    else:
        return (np.concatenate(data_grid_out, axis = 1), model_ids, param_grid_out, dgp_list)  

def bin_arbitrary_fptd(out = [0, 0],
                       bin_dt = 0.04,
                       nbins = 256,
                       nchoices = 2,
                       choice_codes = [-1.0, 1.0],
                       max_t = 10.0): # ['v', 'a', 'w', 'ndt', 'angle']

    # Generate bins
    if nbins == 0:
        nbins = int(max_t / bin_dt)
        bins = np.zeros(nbins + 1)
        bins[:nbins] = np.linspace(0, max_t, nbins)
        bins[nbins] = np.inf
    else:    
        bins = np.zeros(nbins + 1)
        bins[:nbins] = np.linspace(0, max_t, nbins)
        bins[nbins] = np.inf

    cnt = 0
    counts = np.zeros( (nbins, nchoices) ) 

    for choice in choice_codes:
        counts[:, cnt] = np.histogram(out[:, 0][out[:, 1] == choice], bins = bins)[0] 
        print(np.histogram(out[:, 0][out[:, 1] == choice], bins = bins)[1])
        cnt += 1
    return counts

def bin_simulator_output(self, 
                             out = [0, 0],
                             bin_dt = 0.04,
                             nbins = 0): # ['v', 'a', 'w', 'ndt', 'angle']
        
    # Generate bins
    if nbins == 0:
        nbins = int(out[2]['max_t'] / bin_dt)
        bins = np.zeros(nbins + 1)
        bins[:nbins] = np.linspace(0, out[2]['max_t'], nbins)
        bins[nbins] = np.inf
    else:  
        bins = np.zeros(nbins + 1)
        bins[:nbins] = np.linspace(0, out[2]['max_t'], nbins)
        bins[nbins] = np.inf

    cnt = 0
    counts = np.zeros( (nbins, len(out[2]['possible_choices']) ) )

    for choice in out[2]['possible_choices']:
        counts[:, cnt] = np.histogram(out[0][out[1] == choice], bins = bins)[0] / out[2]['n_samples']
        cnt += 1
    return counts
    
# -------------------------------------------------------------------------------------

# RUN 
if __name__ == "__main__":
    # Make command line interface
    CLI = argparse.ArgumentParser()
    
    CLI.add_argument("--machine",
                     type = str,
                     default = 'x7')
    CLI.add_argument("--dgplist", 
                     nargs = "*",
                     type = str,
                     default = ['ddm', 'ornstein', 'angle', 'weibull', 'full_ddm'])
    CLI.add_argument("--datatype",
                     type = str,
                     default = 'uniform') # 'parameter_recovery, 'perturbation_experiment', 'r_sim', 'r_dgp', 'cnn_train', 'parameter_recovery_hierarchical'
    CLI.add_argument("--nsubjects",
                    type = int,
                    default = 5)
    CLI.add_argument("--nreps",
                     type = int,
                     default = 1)
    CLI.add_argument("--binned",
                     type = int,
                     default = 1)
    CLI.add_argument("--nbins",
                     type = int,
                     default = 256)
    CLI.add_argument("--nsamples",
                     type = int,
                     default = 20000)
    CLI.add_argument("--nchoices",
                     type = int,
                     default = 2)
    CLI.add_argument("--mode",
                     type = str,
                     default = 'mlp') # train, test, cnn
    CLI.add_argument("--nsimbnds",
                     nargs = '*',
                     type = int,
                     default = [100, 100000])
    CLI.add_argument("--nparamsets", 
                     type = int,
                     default = 10000)
    CLI.add_argument("--fileid",
                     type = str,
                     default = 'TEST')
    CLI.add_argument("--save",
                     type = bool,
                     default = 0)
    CLI.add_argument("--maxt",
                     type = float,
                     default = 10.0)
    CLI.add_argument("--deltat",
                     type = float,
                     default = 0.001)
    CLI.add_argument("--pickleprotocol",
                     type = int,
                     default = 4)
    
    args = CLI.parse_args()
    print('Arguments passed: ')
    print(args)
    
    machine = args.machine
    
    
    # YAML DATA basically only use for perturbation experiment data
    if machine == 'x7':
        config = yaml.load(open("/media/data_cifs/afengler/git_repos/nn_likelihoods/config_files/config_data_generator.yaml"),
                           Loader = yaml.SafeLoader)
        
    if machine == 'ccv':
        config = yaml.load(open("/users/afengler/git_repos/nn_likelihoods/config_files/config_data_generator.yaml"),
                          Loader = yaml.SafeLoader)
     
    if machine == 'home':
        config = yaml.load(open("/Users/afengler/OneDrive/git_repos/nn_likelihoods/config_files/config_data_generator.yaml"))
    if machine == 'other':
        config = yaml.load(open("config_files/config_data_generator.yaml"))
    
    config = {}
    config['n_cpus'] = 'all'
    
    # Update config with specifics of run
    if args.datatype == 'r_dgp':
        config['method'] = args.dgplist
    else:
        config['method'] = args.dgplist[0]
        
    config['mode'] = args.mode
    config['file_id'] = args.fileid
    config['nsamples'] = args.nsamples
    config['binned'] = args.binned
    config['nbins'] = args.nbins
    config['datatype'] = args.datatype
    config['nchoices'] = args.nchoices
    config['nparamsets'] = args.nparamsets
    config['nreps'] = args.nreps
    config['pickleprotocol'] = args.pickleprotocol
    config['nsimbnds'] = args.nsimbnds
    config['nsubjects'] = args.nsubjects
    
    # Get data for the type of dataset we want
    start_t = datetime.now()

    if args.datatype == 'cnn_train':
        dg = data_generator(machine = args.machine,
                            file_id = args.fileid,
                            max_t = args.maxt,
                            delta_t = args.deltat,
                            config = config)
        out = dg.make_dataset_train_network_unif(save = args.save)
        
    if args.datatype == 'parameter_recovery':
        dg = data_generator(machine = args.machine,
                            file_id = args.fileid,
                            max_t = args.maxt,
                            delta_t = args.deltat,
                            config = config)
        out = dg.make_dataset_parameter_recovery(save = args.save)
    
    if args.datatype == 'perturbation_experiment':      
        dg = data_generator(machine = args.machine,
                            file_id = args.fileid,
                            max_t = args.maxt,
                            delta_t = args.deltat,
                            config = config)
        out = dg.make_dataset_perturbation_experiment(save = args.save)

    if args.datatype == 'r_sim':
        dg = data_generator(machine = args.machine,
                            file_id = args.fileid,
                            max_t = args.maxt,
                            delta_t = args.deltat,
                            config = config)
        out = dg.make_dataset_r_sim(n_sim_bnds = [100, 200000],
                                        save = args.save)
        
    if args.datatype == 'r_dgp':
        out = make_dataset_r_dgp(dgp_list = args.dgplist,
                                 machine = args.machine,
                                 file_id = args.fileid,
                                 r_nsamples = True,
                                 n_sim_bnds = args.nsimbnds,
                                 max_t = args.maxt,
                                 delta_t = args.deltat,
                                 config = config,
                                 save = args.save)

    if args.datatype == 'parameter_recovery_hierarchical':
        dg = data_generator(machine = args.machine,
                            file_id = args.fileid,
                            max_t = args.maxt,
                            delta_t = args.deltat,
                            config = config)

        out = dg.make_dataset_parameter_recovery_hierarchical()

    finish_t = datetime.now()
    print('Time elapsed: ', finish_t - start_t)
    print('Finished')