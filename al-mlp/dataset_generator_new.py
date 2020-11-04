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

import boundary_functions as bf
import multiprocessing as mp
import cddm_data_simulation as cd
import basic_simulator as bs
import kde_info

#from tqdm as tqdm
#from cdwiener import batch_fptd
#import clba

# Parallelization
import multiprocessing as mp
from multiprocessing import Process
from multiprocessing import Pool
import psutil
import argparse
from functools import partial

# --------------------------------------------------------------------------------

# Data generator class that generates datasets for us ----------------------------
class data_generator():
    def __init__(self,
                 config = None):
    # INIT -----------------------------------------
        if config == None:
            print()
            return
        else:
            self.config = config
            self._build_simulator()
            self._get_ncpus()
        
    def _get_ncpus(self):
        
        # Sepfic
        if self.config['n_cpus'] == 'all':
            n_cpus = psutil.cpu_count(logical = False)
        else:
            n_cpus = self.config['n_cpus']
        
        self.config['n_cpus'] = n_cpus
        
    def _build_simulator(self):
        self.simulator = partial(bs.simulator, 
                                 n_samples = config['n_samples'],
                                 max_t = config['max_t'],
                                 bin_dim = config['nbins'],
                                 delta_t = config['delta_t'])
                                 
    def get_simulations(self, theta = None):
        out = self.simulator(theta, 
                             self.config['method'])
        # TODO: Add 
        if self.config['nbins'] is not None:
            return np.concatenate([out[0], out[1]], axis = 1)
        else:
            return out
                                                             
    def generate_data_uniform(self, save = False):
        
        # Make parameters
        theta_list = [np.float32((np.random.uniform(low = self.config['param_bounds'][0], 
                                                    high = self.config['param_bounds'][1])) for i in range(self.config['nparamsets']))]
        
        # Get simulations
        with Pool(processes = self.config['n_cpus']) as pool:
            data_grid = np.array(pool.map(self.get_simulations, theta_list))
         
        
        # Save to correct destination
        if save:
            
            # -----
            if self.config['mode'] == 'test':
                training_data_folder = self.config['method_folder'] + \
                                       'parameter_recovery_data_binned_' + \
                                       str(int(self.config['binned'])) + \
                                       '_nbins_' + str(self.config['nbins']) + \
                                       '_n_' + str(self.config['nsamples'])
                
                if not os.path.exists(training_data_folder):
                    os.makedirs(training_data_folder)

                full_file_name = training_data_folder + '/' + \
                                self.config['method'] + \
                                '_nchoices_' + str(self.config['nchoices']) + \
                                '_parameter_recovery_binned_' + \
                                str(int(self.config['binned'])) + \
                                '_nbins_' + str(self.config['nbins']) + \
                                '_nreps_' + str(self.config['nreps']) + \
                                '_n_' + str(self.config['nsamples']) + \
                                '.pickle'
            
            else:
                training_data_folder = self.config['method_folder'] + \
                                      'training_data_binned_' + \
                                      str(int(self.config['binned'])) + \
                                      '_nbins_' + str(self.config['nbins']) + \
                                      '_n_' + str(self.config['nsamples'])
                
                if not os.path.exists(training_data_folder):
                    os.makedirs(training_data_folder)

                full_file_name = training_data_folder + '/' + \
                                self.config['method'] + \
                                '_nchoices_' + str(self.config['nchoices']) + \
                                '_train_data_binned_' + \
                                str(int(self.config['binned'])) + \
                                '_nbins_' + str(self.config['nbins']) + \
                                '_n_' + str(self.config['nsamples']) + \
                                '_' + self.file_id + '.pickle'
            
            print('Writing to file: ', full_file_name)
            
            pickle.dump((np.float32(np.stack(theta_list)), 
                         np.float32(np.expand_dims(data_grid, axis = 0)), 
                         self.config['meta']), 
                        open(full_file_name, 'wb'), 
                        protocol = self.config['pickleprotocol'])
            
            return 'Dataset completed'
        
        # Or else return the data
        else:
            return np.float32(np.stack(theta_list)), np.float32(np.expand_dims(data_grid, axis = 0)) 
            
    def generate_data_hierarchical(self, save = False):
        
        subject_param_grid, global_stds, global_means = self._make_param_grid_hierarchical()
        subject_param_grid_adj_sim = np.reshape(subject_param_grid, (-1, self.config['nparams'])).tolist()
        subject_param_grid_adj_sim = tuple([(np.array(i),) for i in subject_param_grid_adj_sim])
        
        with Pool(processes = self.config['n_cpus']) as pool:
            data_grid = np.array(pool.starmap(self.get_simulations, subject_param_grid_adj_sim))
            
        if save:
            training_data_folder = self.config['method_folder'] + 'parameter_recovery_hierarchical_data_binned_' + str(int(self.config['binned'])) + \
                                   '_nbins_' + str(self.config['nbins']) + \
                                   '_n_' + str(self.config['nsamples'])
            
            full_file_name = training_data_folder + '/' + \
                             self.config['method'] + \
                             '_nchoices_' + str(self.config['nchoices']) + \
                             '_parameter_recovery_hierarchical_' + \
                             'binned_' + str(int(self.config['binned'])) + \
                             '_nbins_' + str(self.config['nbins']) + \
                             '_nreps_' + str(self.config['nreps']) + \
                             '_n_' + str(self.config['nsamples']) + \
                             '_nsubj_' + str(self.config['nsubjects']) + \
                             '.pickle'
            
            if not os.path.exists(training_data_folder):
                os.makedirs(training_data_folder)
            
            print('saving dataset as ', full_file_name)
            
            pickle.dump(([subject_param_grid, global_stds, global_means], 
                          np.expand_dims(data_grid, axis = 0),
                          self.config['meta']), 
                        open(full_file_name, 'wb'), 
                        protocol = self.config['pickleprotocol'])
            
            return 'Dataset completed'
        else:
            return ([subject_param_grid, global_stds, global_means], data_grid, meta)
  
    def _make_param_grid_hierarchical(self):
        # Initialize global parameters

        params_ranges_half = (np.array(self.config['param_bounds'][1]) - np.array(self.config['param_bounds'][0])) / 2
        
        # Sample global parameters from cushioned parameter space
        global_stds = np.random.uniform(low = 0.001,
                                        high = params_ranges_half / 10,
                                        size = (self.config['nparamsets'], self.config['nparams']))
        global_means = np.random.uniform(low = self.config['param_bounds'][0] + (params_ranges_half / 5),
                                         high = self.config['param_bounds'][1] - (params_ranges_half / 5),
                                         size = (self.config['nparamsets'], self.config['nparams']))

        # Initialize local parameters (by condition)
        subject_param_grid = np.float32(np.zeros((self.config['nparamsets'], self.config['nsubjects'], self.config['nparams'])))
        
        # Sample by subject parameters from global setup (truncate to never go out of allowed parameter space)
        for n in range(self.config['nparamsets']):
            for i in range(self.config['nsubjects']):
                a, b = (self.config['param_bounds'][0] - global_means[n]) / global_stds[n], (self.config['param_bounds'][1] - global_means[n]) / global_stds[n]
                subject_param_grid[n, i, :] = np.float32(global_means[n] + truncnorm.rvs(a, b, size = global_stds.shape[1]) * global_stds[n])

        return subject_param_grid, global_stds, global_means
   # ----------------------------------------------------
 
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
                     default = 'uniform')
    CLI.add_argument("--mode",
                     type = str,
                     default = 'test') # 'parameter_recovery, 'perturbation_experiment', 'r_sim', 'r_dgp', 'cnn_train', 'parameter_recovery_hierarchical'
    CLI.add_argument("--nsubjects",
                    type = int,
                    default = 5)
    CLI.add_argument("--nreps",
                     type = int,
                     default = 1)
    CLI.add_argument("--nbins",
                     type = int,
                     default = None)
    CLI.add_argument("--nsamples",
                     type = int,
                     default = 20000)
    CLI.add_argument("--nchoices",
                     type = int,
                     default = 2)
#     CLI.add_argument("--mode",
#                      type = str,
#                      default = 'mlp') # train, test, cnn
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
    
    # SETTING UP CONFIG --------------------------------------------------------------------------------
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
    if args.nbins is not None:
        config['binned'] = 1
    else:
        config['binned'] = 0
        
    if args.nbins == 0:
        config['nbins'] = None
    else:
        config['nbins'] = args.nbins
        
    config['datatype'] = args.datatype
    config['nchoices'] = args.nchoices
    config['nparamsets'] = args.nparamsets
    config['nreps'] = args.nreps
    config['pickleprotocol'] = args.pickleprotocol
    config['nsimbnds'] = args.nsimbnds
    config['nsubjects'] = args.nsubjects
    config['n_samples'] = args.nsamples
    config['max_t'] = args.maxt
    config['delta_t'] = args.deltat
    
    # Make parameter bounds
    if args.mode == 'train' and config['binned']:
        bounds_tmp = kde_info.temp[config['method']]['param_bounds_cnn'] + kde_info.temp[config['method']]['boundary_param_bounds_cnn']
    elif args.mode == 'train' and not config['binned']:
        bounds_tmp = kde_info.temp[config['method']]['param_bounds_network'] + kde_info.temp[config['method']]['boundary_param_bounds_network']
    elif args.mode == 'test' and config['binned']:
        bounds_tmp = kde_info.temp[config['method']]['param_bounds_sampler'] + kde_info.temp[config['method']]['boundary_param_bounds_sampler']
    elif args.mode == 'test' and not config['binned']:
        bounds_tmp = kde_info.temp[config['method']]['param_bounds_sampler'] + kde_info.temp[config['method']]['boundary_param_bounds_sampler']

    config['param_bounds'] = np.array([[i[0] for i in bounds_tmp], [i[1] for i in bounds_tmp]])
    config['nparams'] = config['param_bounds'][0].shape[0]
    
    config['meta'] = kde_info.temp[config['method']]['dgp_hyperparameters']
    
    # Add some machine dependent folder structure
    if args.machine == 'x7':
        config['method_comparison_folder'] = kde_info.temp[config['method']]['output_folder_x7']
        config['method_folder'] = kde_info.temp[config['method']]['method_folder_x7']

    if args.machine == 'ccv':
        config['method_comparison_folder'] = kde_info.temp[config['method']]['output_folder']
        config['method_folder'] = kde_info.temp[config['method']]['method_folder']

    if args.machine == 'home':
        config['method_comparison_folder'] = kde_info.temp[config['method']]['output_folder_home']
        config['method_folder'] = kde_info.temp[config['method']]['method_folder_home']

    if args.machine == 'other': # This doesn't use any extra 
        if not os.path.exists('data_storage'):
            os.makedirs('data_storage')

        print('generated new folder: data_storage. Please update git_ignore if this is not supposed to be committed to repo')

        config['method_comparison_folder']  = 'data_storage/'
        config['method_folder'] = 'data_storage/' + config['method'] + '_'
    # -------------------------------------------------------------------------------------
    
    # GET DATASETS ------------------------------------------------------------------------
    # Get data for the type of dataset we want
    start_t = datetime.now()
    
    dg = data_generator(config = config)
    
    if args.datatype == 'parameter_recovery' or args.datatype == 'training':
        dg.generate_data_uniform(save = args.save)
        
    if args.datatype == 'parameter_recovery_hierarchical':
        dg.generate_data_hierarchical(save = args.save)
        
    finish_t = datetime.now()
    print('Time elapsed: ', finish_t - start_t)
    print('Finished')
    # -------------------------------------------------------------------------------------
    
# UNUSED ------------------------------    
    
#     def generate_data_grid_parallel(self,
#                                     param_grid = []):
        
#         args_list = self.make_args_starmap_ready(param_grid = param_grid)
        
#         if self.config['n_cpus'] == 'all':
#             n_cpus = psutil.cpu_count(logical = False)
#         else:
#             n_cpus = self.config['n_cpus']

#         # Run Data generation
#         with Pool(processes = n_cpus) as pool:
#             data_grid = np.array(pool.starmap(self.data_generator, args_list))

#         return data_grid   
  
        
  
#     def clean_up_parameters(self):
        
#         if self.config['mode'] == 'test':
#             param_bounds = self.method_params['param_bounds_sampler'] + self.method_params['boundary_param_bounds_sampler']
#         if self.config['mode'] == 'mlp':
#             param_bounds = self.method_params['param_bounds_network'] + self.method_params['boundary_param_bounds_network']
#         if self.config['mode'] == 'cnn':
#             param_bounds = self.method_params['param_bounds_cnn'] + self.method_params['boundary_param_bounds_cnn']
        
#         # Epsilon correction of boundaries (to make sure for parameter recovery we don't generate straight at the bound)
        
#         eps = 0
#         if self.config['datatype'] == 'parameter_recovery' and self.config['mode'] != 'test':
#             # TD make eps parameter
#             eps = 0.05
            
#         print('epsilon correction', eps)

#         # If model is lba, lca, race we need to expand parameter boundaries to account for
#         # parameters that depend on the number of choices
#         if self.method == 'lba' or self.method == 'lca' or self.method == 'race_model':
#             param_depends_on_n = self.method_params['param_depends_on_n_choice']
#             param_bounds_tmp = []
            
#             n_process_params = len(self.method_params['param_names'])
            
#             p_cnt = 0
#             for i in range(n_process_params):
#                 if self.method_params['param_depends_on_n_choice'][i]:
#                     for c in range(self.config['nchoices']):
#                         param_bounds_tmp.append(param_bounds[i])
#                         p_cnt += 1
#                 else:
#                     param_bounds_tmp.append(param_bounds[i])
#                     p_cnt += 1
            
#             self.method_params['n_process_parameters'] = p_cnt
            
#             param_bounds_tmp += param_bounds[n_process_params:]
#             params_upper_bnd = [bnd[1] - eps for bnd in param_bounds_tmp]
#             params_lower_bnd = [bnd[0] + eps for bnd in param_bounds_tmp]
                
#             #print(params_lower_bnd)
            
            
#         # If our model is not lba, race, lca we use simple procedure 
#         else:
#             params_upper_bnd = [bnd[1] - eps for bnd in param_bounds]
#             params_lower_bnd = [bnd[0] + eps for bnd in param_bounds]
            
#         return params_upper_bnd, params_lower_bnd
                       

#     def make_dataset_perturbation_experiment(self,
#                                              save = True):
        
#         param_grid, meta_dat = self.make_param_grid_perturbation_experiment()
             
#         if self.config['binned']:           
#             data_grid = np.zeros((self.config['nreps'],
#                                   self.config['n_experiments'],
#                                   param_grid.shape[1], 
#                                   self.config['nbins'],
#                                   self.config['nchoices']))         
        
#         else:
#             data_grid = np.zeros((self.config['nreps'],
#                                   self.config['n_experiments'],
#                                   param_grid.shape[1], 
#                                   self.config['nsamples'],
#                                   2)) 

#         for experiment in range(self.config['n_experiments']):
#             for rep in range(self.config['nreps']):
#                 data_grid[rep, experiment] = self.generate_data_grid_parallel(param_grid = param_grid[experiment])
#                 print(experiment, ' experiment data finished')
        
#         if save == True:
#             print('saving dataset')
#             pickle.dump((param_grid, data_grid, meta_dat), open(self.method_comparison_folder + \
#                                                                 'base_data_perturbation_experiment_nexp_' + \
#                                                                 str(self.config['n_experiments']) + \
#                                                                 '_nreps_' + str(self.config['nreps']) + \
#                                                                 '_n_' + str(self.config['nsamples']) + \
#                                                                 '_' + self.config['file_id'] + '.pickle', 'wb'))
                        
#             return 'Dataset completed'
#         else:
#             return param_grid, data_grid, meta_dat
    
    
# def make_dataset_parameter_recovery(self,
#                                         save = True):
        
#         param_grid = self.make_param_grid_uniform()
#         self.nsamples = [self.config['nsamples'] for i in range(self.config['nparamsets'])]
        
#         if self.config['binned']:
#             data_grid = np.zeros((self.config['nreps'],
#                                   self.config['nparamsets'],
#                                   self.config['nbins'],
#                                   self.config['nchoices']))
#         else:
#             data_grid = np.zeros((self.config['nreps'],
#                                   self.config['nparamsets'],
#                                   self.config['nsamples'],
#                                   2))
        
#         for rep in range(self.config['nreps']):
#             data_grid[rep] = np.array(self.generate_data_grid_parallel(param_grid = param_grid))
#             print(rep, ' repetition finished') 
        
        
#         if save:
#             training_data_folder = self.method_folder + 'parameter_recovery_data_binned_' + str(int(self.config['binned'])) + \
#                                    '_nbins_' + str(self.config['nbins']) + \
#                                    '_n_' + str(self.config['nsamples'])
#             if not os.path.exists(training_data_folder):
#                 os.makedirs(training_data_folder)
                
#             full_file_name = training_data_folder + '/' + \
#                             self.method + \
#                             '_nchoices_' + str(self.config['nchoices']) + \
#                             '_parameter_recovery_' + \
#                             'binned_' + str(int(self.config['binned'])) + \
#                             '_nbins_' + str(self.config['nbins']) + \
#                             '_nreps_' + str(self.config['nreps']) + \
#                             '_n_' + str(self.config['nsamples']) + \
#                             '.pickle'
            
#             print(full_file_name)
            
#             meta = self.dgp_hyperparameters.copy()
#             if 'boundary' in meta.keys():
#                 del meta['boundary']

#             pickle.dump((param_grid, data_grid, meta), 
#                         open(full_file_name, 'wb'), 
#                         protocol = self.config['pickleprotocol'])
            
#             return 'Dataset completed'
#         else:
#             return param_grid, data_grid

#     def make_dataset_train_network_unif(self,
#                                         save = True):

#         param_grid = self.make_param_grid_uniform()
#         self.nsamples = [self.config['nsamples'] for i in range(self.config['nparamsets'])]

#         if self.config['binned']:
#             data_grid = np.zeros((self.config['nparamsets'],
#                                   self.config['nbins'],
#                                   self.config['nchoices']))
#         else:
#             data_grid = np.zeros((self.config['nparamsets'],
#                                   self.config['nsamples'],
#                                   2))
#         data_grid = np.array(self.generate_data_grid_parallel(param_grid = param_grid))

#         if save:
#             training_data_folder = self.method_folder + 'training_data_binned_' + str(int(self.config['binned'])) + \
#                                    '_nbins_' + str(self.config['nbins']) + \
#                                    '_n_' + str(self.config['nsamples'])
#             if not os.path.exists(training_data_folder):
#                 os.makedirs(training_data_folder)
                
#             full_file_name = training_data_folder + '/' + \
#                             self.method + \
#                             '_nchoices_' + str(self.config['nchoices']) + \
#                             '_train_data_' + \
#                             'binned_' + str(int(self.config['binned'])) + \
#                             '_nbins_' + str(self.config['nbins']) + \
#                             '_n_' + str(self.config['nsamples']) + \
#                             '_' + self.file_id + '.pickle'
            
#             print(full_file_name)
            
#             meta = self.dgp_hyperparameters.copy()
#             if 'boundary' in meta.keys():
#                 del meta['boundary']

#             pickle.dump((param_grid, data_grid, meta), 
#                         open(full_file_name, 'wb'), 
#                         protocol = self.config['pickleprotocol'])
#             return 'Dataset completed'
#         else:
#             return param_grid, data_grid

#     def make_param_grid_hierarchical(self,
#                                      ):

#         # Initializations
#         params_upper_bnd, params_lower_bnd = self.clean_up_parameters()
#         nparams = len(params_upper_bnd)
        
#         # Initialize global parameters
#         params_ranges_half = (np.array(params_upper_bnd) - np.array(params_lower_bnd)) / 2
        
#         global_stds = np.random.uniform(low = 0.001,
#                                         high = params_ranges_half / 10,
#                                         size = (self.config['nparamsets'], nparams))
#         global_means = np.random.uniform(low = np.array(params_lower_bnd) + (params_ranges_half / 5),
#                                          high = np.array(params_upper_bnd) - (params_ranges_half / 5),
#                                          size = (self.config['nparamsets'], nparams))

#         # Initialize local parameters (by condition)
#         subject_param_grid = np.zeros((self.config['nparamsets'], self.config['nsubjects'], nparams))
        
#         for n in range(self.config['nparamsets']):
#             for i in range(self.config['nsubjects']):
#                 a, b = (np.array(params_lower_bnd) - global_means[n]) / global_stds[n], (np.array(params_upper_bnd) - global_means[n]) / global_stds[n]
#                 subject_param_grid[n, i, :] = np.float32(global_means[n] + truncnorm.rvs(a, b, size = global_stds.shape[1]) * global_stds[n])
                
# #                 Print statements to test if sampling from truncated distribution works properly
# #                 print('random variates')
# #                 print(truncnorm.rvs(a, b, size = global_stds.shape[1]))
# #                 print('samples')
# #                 print(subject_param_grid[n, i, :])

#         return subject_param_grid, global_stds, global_means


   
#     def generate_data_grid_hierarchical_parallel(self, 
#                                                  param_grid = []):
        
#         nparams = param_grid.shape[2]
#         args_list = self.make_args_starmap_ready(param_grid = np.reshape(param_grid, (-1, nparams)))

#         if self.config['n_cpus'] == 'all':
#             n_cpus = psutil.cpu_count(logical = False)
#         else:
#             n_cpus = self.config['n_cpus']

#         # Run Data generation
#         with Pool(processes = n_cpus) as pool:
#             data_grid = np.array(pool.starmap(self.data_generator, args_list))

#         data_grid = np.reshape(data_grid, (self.config['nparamsets'], self.config['nsubjects'], self.config['nsamples'], self.config['nchoices']))
    
#         return data_grid

#     def make_dataset_parameter_recovery_hierarchical(self,
#                                                      save = True):

#         param_grid, global_stds, global_means = self.make_param_grid_hierarchical()
        
#         self.nsamples = [self.config['nsamples'] for i in range(self.config['nparamsets'] * self.config['nsubjects'])]
        
#         if self.config['binned']:
#             data_grid = np.zeros((self.config['nreps'],
#                                   self.config['nparamsets'],
#                                   self.config['nsubjects'], # add to config
#                                   self.config['nbins'],
#                                   self.config['nchoices']))
#         else:
#             data_grid = np.zeros((self.config['nreps'],
#                                   self.config['nparamsets'],
#                                   self.config['nsubjects'],
#                                   self.config['nsamples'],
#                                   2))
        
#         for rep in range(self.config['nreps']):
#             # TD: ADD generate_data_grid_parallel_multisubject
#             data_grid[rep] = np.array(self.generate_data_grid_hierarchical_parallel(param_grid = param_grid))
#             print(rep, ' repetition finished') 
        
#         if save:
#             training_data_folder = self.method_folder + 'parameter_recovery_hierarchical_data_binned_' + str(int(self.config['binned'])) + \
#                                    '_nbins_' + str(self.config['nbins']) + \
#                                    '_n_' + str(self.config['nsamples'])
#             if not os.path.exists(training_data_folder):
#                 os.makedirs(training_data_folder)
            
#             print('saving dataset as ', training_data_folder + '/' + \
#                                         self.method + \
#                                         '_nchoices_' + str(self.config['nchoices']) + \
#                                         '_parameter_recovery_hierarchical_' + \
#                                         'binned_' + str(int(self.config['binned'])) + \
#                                         '_nbins_' + str(self.config['nbins']) + \
#                                         '_nreps_' + str(self.config['nreps']) + \
#                                         '_n_' + str(self.config['nsamples']) + \
#                                         '_nsubj_' + str(self.config['nsubjects']) + \
#                                         '.pickle')
            
#             meta = self.dgp_hyperparameters.copy()
#             if 'boundary' in meta.keys():
#                 del meta['boundary']

#             pickle.dump(([param_grid, global_stds, global_means], data_grid, meta), 
#                         open(training_data_folder + '/' + \
#                             self.method + \
#                             '_nchoices_' + str(self.config['nchoices']) + \
#                             '_parameter_recovery_hierarchical_' + \
#                             'binned_' + str(int(self.config['binned'])) + \
#                             '_nbins_' + str(self.config['nbins']) + \
#                             '_nreps_' + str(self.config['nreps']) + \
#                             '_n_' + str(self.config['nsamples']) + \
#                             '_nsubj_' + str(self.config['nsubjects']) + \
#                             '.pickle', 'wb'), 
#                         protocol = self.config['pickleprotocol'])
            
#             return 'Dataset completed'
#         else:
#             return ([param_grid, global_stds, global_means], data_grid, meta)

#     def make_dataset_r_sim(self,
#                            n_sim_bnds = [10000, 100000],
#                            save = True):
        
#         param_grid = self.make_param_grid_uniform()
#         self.nsamples = np.random.uniform(size = self.config['nparamsets'], 
#                                           high = n_sim_bnds[1],
#                                           low = n_sim_bnds[0])
        
#         if self.config['binned']:
#             data_grid = np.zeros((self.config['nreps'],
#                                   self.config['nparamsets'],
#                                   self.config['nbins'],
#                                   self.config['nchoices']))
#         else:
#             return 'random number of simulations is supported under BINNED DATA only for now'
        
#         for rep in range(self.config['nreps']):
#             data_grid[rep] = np.array(self.generate_data_grid_parallel(param_grid = param_grid))
#             print(rep, ' repetition finished')
            
            
#         if save == True:
#             print('saving dataset')
#             pickle.dump((param_grid, data_grid), open(self.method_comparison_folder + \
#                                                       'base_data_uniform_r_sim_' + \
#                                                       str(n_sim_bnds[0]) + '_' + str(n_sim_bnds[1]) + \
#                                                       '_nreps_' + str(self.config['nreps']) + \
#                                                       '_' + self.file_id + '.pickle', 'wb'))
#             return 'Dataset completed'
#         else:
#             return param_grid, data_grid, self.nsamples
# ---------------------------------------------------------------------------------------

# # Functions outside the data generator class that use it --------------------------------
# def make_dataset_r_dgp(dgp_list = ['ddm', 'ornstein', 'angle', 'weibull', 'full_ddm'],
#                        machine = 'x7',
#                        r_nsamples = True,
#                        n_sim_bnds = [100, 100000],
#                        file_id = 'TEST',
#                        max_t = 10.0,
#                        delta_t = 0.001,
#                        config = None,
#                        save = False):
#     """Generates datasets across kinds of simulators
    
#     Parameter 
#     ---------
#     dgp_list : list
#         List of simulators that you would like to include (names match the simulators stored in kde_info.py)
#     machine : str
#         The machine the code is run on (basically changes folder directory for imports, meant to be temporary)
#     file_id : str
#         Attach an identifier to file name that is stored if 'save' is True (some file name formatting is already applied)
#     save : bool
#         If true saves outputs to pickle file
    
#     Returns
#     -------
#     list 
#         [0] list of arrays of parameters by dgp (n_dgp, nparamsets, n_parameters_given_dgp)
#         [1] list of arrays storing sampler outputs as histograms (n_repitions, n_parameters_sets, nbins, nchoices)
#         [2] array of model ids (dgp ids)
#         [3] dgp_list
        
#     """
  
#     if config['binned']:
#         model_ids = np.random.choice(len(dgp_list),
#                                      size = (config['nparamsets'], 1))
#         model_ids.sort(axis = 0)
#         data_grid_out = []
#         param_grid_out = []
#         nsamples_out = []
#     else:
#         return 'Config should specify binned = True for simulations to start'

#     for i in range(len(dgp_list)):
#         nparamsets = np.sum(model_ids == i)
        
#         # Change config to update the sampler
#         config['method'] = dgp_list[i]
#         # Change config to update the number of parameter sets we want to run for the current sampler
#         config['nparamsets'] = nparamsets
        
#         # Initialize data_generator class with new properties
#         dg_tmp = data_generator(machine = machine,
#                                 config = config,
#                                 max_t = max_t,
#                                 delta_t = delta_t)
        
#         # Run the simulator
#         if r_nsamples:
#             param_grid, data_grid, nsamples = dg_tmp.make_dataset_r_sim(n_sim_bnds = n_sim_bnds,
#                                                                         save = False)
#         else:
#             param_grid, data_grid = dg_tmp.make_dataset_train_network_unif(save = False)
            
#         print(data_grid.shape)
#         print(param_grid.shape)
        
#         # Append results
#         data_grid_out.append(data_grid)
#         param_grid_out.append(param_grid)
#         if r_nsamples:
#             nsamples_out.append(nsamples_out)

#     if save:
#         print('saving dataset')
#         if machine == 'x7':
#             out_folder = '/media/data_cifs/afengler/data/kde/rdgp/'
#         if machine == 'ccv':
#             out_folder = '/users/afengler/data/kde/rdgp/'
#         if machine == 'home':
#             out_folder = '/Users/afengler/OneDrive/project_nn_likelihoods/data/kde/rdgp/'
#         if machine == 'other':
#             if not os.path.exists('data_storage'):
#                 os.makedirs('data_storage')
#                 os.makedirs('data_storage/rdgp')
            
#             out_folder = 'data_storage/rdgp/'
            
#         out_folder = out_folder + 'training_data_' + str(int(config['binned'])) + \
#                      '_nbins_' + str(config['nbins']) + \
#                      '_nsimbnds_' + str(config['nsimbnds'][0]) + '_' + str(config['nsimbnds'][1]) +'/'
            
#         if not os.path.exists(out_folder):
#                 os.makedirs(out_folder)
                
#         pickle.dump((np.concatenate(data_grid_out, axis = 1), model_ids, param_grid_out, dgp_list),
#                                                    open(out_folder + \
#                                                    'rdgp_nchoices_' + str(config['nchoices']) + \
#                                                    '_nreps_' + str(config['nreps']) + \
#                                                    '_nsimbnds_' + str(config['nsimbnds'][0]) + '_' + \
#                                                    str(config['nsimbnds'][1]) + \
#                                                    '_' + str(config['file_id']) + '.pickle', 'wb'))
#         return 'Dataset completed'
#     else:
#         return (np.concatenate(data_grid_out, axis = 1), model_ids, param_grid_out, dgp_list)  

#      if args.datatype == 'cnn_train':
#         simulator = bs.simulator()
#         dg = data_generator(machine = args.machine,
#                             file_id = args.fileid,
#                             max_t = args.maxt,
#                             delta_t = args.deltat,
#                             config = config)
#         out = dg.make_dataset_train_network_unif(save = args.save)
        
#     if args.datatype == 'parameter_recovery':
#         dg = data_generator(machine = args.machine,
#                             file_id = args.fileid,
#                             max_t = args.maxt,
#                             delta_t = args.deltat,
#                             config = config)
#         out = dg.make_dataset_parameter_recovery(save = args.save)
    
#     if args.datatype == 'perturbation_experiment':      
#         dg = data_generator(machine = args.machine,
#                             file_id = args.fileid,
#                             max_t = args.maxt,
#                             delta_t = args.deltat,
#                             config = config)
#         out = dg.make_dataset_perturbation_experiment(save = args.save)

#     if args.datatype == 'r_sim':
#         dg = data_generator(machine = args.machine,
#                             file_id = args.fileid,
#                             max_t = args.maxt,
#                             delta_t = args.deltat,
#                             config = config)
#         out = dg.make_dataset_r_sim(n_sim_bnds = [100, 200000],
#                                         save = args.save)
        
#     if args.datatype == 'r_dgp':
#         out = make_dataset_r_dgp(dgp_list = args.dgplist,
#                                  machine = args.machine,
#                                  file_id = args.fileid,
#                                  r_nsamples = True,
#                                  n_sim_bnds = args.nsimbnds,
#                                  max_t = args.maxt,
#                                  delta_t = args.deltat,
#                                  config = config,
#                                  save = args.save)

#     if args.datatype == 'parameter_recovery_hierarchical':
#         dg = data_generator(machine = args.machine,
#                             file_id = args.fileid,
#                             max_t = args.maxt,
#                             delta_t = args.deltat,
#                             config = config)

#         out = dg.make_dataset_parameter_recovery_hierarchical()
