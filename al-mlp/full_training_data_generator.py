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
from scipy.stats import mode

import boundary_functions as bf
import multiprocessing as mp
import cddm_data_simulation as cd
import basic_simulator as bs
import kde_info
import kde_class

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
            print('n_cpus: ', n_cpus)
        else:
            n_cpus = self.config['n_cpus']

        self.config['n_cpus'] = n_cpus
             
    def _build_simulator(self):
        self.simulator = partial(bs.simulator, 
                                 n_samples = self.config['n_samples'],
                                 max_t = self.config['max_t'],
                                 bin_dim = self.config['nbins'],
                                 delta_t = self.config['delta_t'])
                                 
    def get_simulations(self, theta = None):
        out = self.simulator(theta  = theta, 
                             model = self.config['method'])
        
        #print(self.config['nbins'])
        if self.config['nbins'] > 0:
            return self._bin_simulator_output(simulations = out)
        else:
            return out
    
    def _get_training_data_theta(self, theta):
        out = self.get_simulations(theta)
        
    def _bin_simulator_output(self, 
                              simulations = None): # ['v', 'a', 'w', 'ndt', 'angle']
        
        # Generate bins 
        bins = np.zeros(self.config['nbins'] + 1)
        bins[:self.config['nbins']] = np.linspace(0, simulations[2]['max_t'], self.config['nbins'])
        bins[self.config['nbins']] = np.inf

        # Make Choice Histogram
        cnt = 0
        counts = np.zeros( (self.config['nbins'], len(simulations[2]['possible_choices']) ) )
        for choice in simulations[2]['possible_choices']:
            counts[:, cnt] = np.histogram(simulations[0][simulations[1] == choice], bins = bins)[0] / simulations[2]['n_samples']
            cnt += 1
        return counts
    
    def _filter_simulations_fast(self,
                                 simulations = None,
                                 ):
        #print(simulations[2])

        max_t = simulations[2]['max_t']
        #tmp_max_rt_ = simulations[0].max().round(2)

        keep = 1
        for choice_tmp in simulations[2]['possible_choices']:
            tmp_rts = simulations[0][simulations[1] == choice_tmp]
            tmp_n_c = len(tmp_rts)

            if tmp_n_c > 0:
                mode_, mode_cnt_ = mode(tmp_rts)
                std_ = np.std(tmp_rts)
                mean_ = np.mean(tmp_rts)
                mode_cnt_rel_ = mode_cnt_ / tmp_n_c
                choice_prop_  = tmp_n_c / simulations[2]['n_samples']
            else:
                mode_ = -1
                mode_cnt_ = 0
                mean_ = -1
                std_ = -1
                mode_cnt_rel_ = 1
                choice_prop_  = 0

            keep = keep & \
                   (mode_ != self.config['filter']['mode']) & \
                   (mean_ < self.config['filter']['mean_rt']) & \
                   (std_ > self.config['filter']['std']) & \
                   (mode_cnt_rel_ < self.config['filter']['mode_cnt_rel']) & \
                   (tmp_n_c > self.config['filter']['choice_cnt'])
        
        return keep, np.array([mode_, mean_, std_, mode_cnt_rel_, tmp_n_c])
             
    def _make_kde_data(self,
                       simulations = None, 
                       theta = None):
        
        
        n = self.config['n_by_param']
        p = self.config['mixture_probabilities']
        n_kde = int(n * p[0])
        n_unif_up = int(n * p[1])
        n_unif_down = int(n * p[2])
                    
        out = np.zeros((n_kde + n_unif_up + n_unif_down, 
                        3 + len(theta)))
        out[:, :len(theta)] = np.tile(theta, (n_kde + n_unif_up + n_unif_down, 1) )
        
        tmp_kde = kde_class.logkde((simulations[0],
                                    simulations[1], 
                                    simulations[2]))

        # Get kde part
        samples_kde = tmp_kde.kde_sample(n_samples = n_kde)
        likelihoods_kde = tmp_kde.kde_eval(data = samples_kde).ravel()

        out[:n_kde, -3] = samples_kde[0].ravel()
        out[:n_kde, -2] = samples_kde[1].ravel()
        out[:n_kde, -1] = likelihoods_kde

        # Get positive uniform part:
        choice_tmp = np.random.choice(simulations[2]['possible_choices'],
                                      size = n_unif_up)

        if simulations[2]['max_t'] < 100:
            rt_tmp = np.random.uniform(low = 0.0001,
                                       high = simulations[2]['max_t'],
                                       size = n_unif_up)
        else: 
            rt_tmp = np.random.uniform(low = 0.0001, 
                                       high = 100,
                                       size = n_unif_up)

        likelihoods_unif = tmp_kde.kde_eval(data = (rt_tmp, choice_tmp)).ravel()

        out[n_kde:(n_kde + n_unif_up), -3] = rt_tmp
        out[n_kde:(n_kde + n_unif_up), -2] = choice_tmp
        out[n_kde:(n_kde + n_unif_up), -1] = likelihoods_unif

        # Get negative uniform part:
        choice_tmp = np.random.choice(simulations[2]['possible_choices'],
                                      size = n_unif_down)

        rt_tmp = np.random.uniform(low = - 1.0,
                                   high = 0.0001,
                                   size = n_unif_down)

        out[(n_kde + n_unif_up):, -3] = rt_tmp
        out[(n_kde + n_unif_up):, -2] = choice_tmp
        out[(n_kde + n_unif_up):, -1] = -66.77497

        return out.astype(np.float)
    
    def _mlp_get_processed_data_for_theta(self,
                                      random_seed):
        np.random.seed(random_seed)
        keep = 0
        while not keep:
            theta = np.float32(np.random.uniform(low = self.config['param_bounds'][0], 
                                                 high = self.config['param_bounds'][1]))
            
            simulations = self.get_simulations(theta = theta)
            #print(theta)
            #print(simulations)
            keep, stats = self._filter_simulations_fast(simulations)

        data = self._make_kde_data(simulations = simulations,
                                   theta = theta)
        return data
                     
    def _cnn_get_processed_data_for_theta(self,
                                          random_seed):
        np.random.seed(random_seed)
        theta = np.float32(np.random.uniform(low = self.config['param_bounds'][0], 
                                             high = self.config['param_bounds'][1]))
        return self.get_simulations(theta = theta)
             
    def _get_rejected_parameter_setups(self,
                                       random_seed):
        np.random.seed(random_seed)
        rejected_thetas = []
        keep = 1
        rej_cnt = 0
        while keep and rej_cnt < 100:
            theta = np.float32(np.random.uniform(low = self.config['param_bounds'][0], 
                                                     high = self.config['param_bounds'][1]))
            
            simulations = self.get_simulations(theta = theta)
            
            keep, stats = self._filter_simulations_fast(simulations)
            
            if keep == 0:
                print('simulation rejected')
                print('stats: ', stats)
                print('theta', theta)
                rejected_thetas.append(theta)
                break

            rej_cnt += 1
        
        return rejected_thetas
  
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
                 
    def generate_data_training_uniform(self, 
                                       save = False):
        
        seeds = np.random.choice(400000000, size = self.config['nparamsets'])
        
        # Inits
        subrun_n = self.config['nparamsets'] // self.config['printsplit']
        
        if self.config['nbins'] == 0:
            data_grid = np.zeros((int(self.config['nparamsets'] * 1000), 
                              len(self.config['param_bounds'][0]) + 3))

        # Get Simulations 
            for i in range(self.config['printsplit']):
                print('simulation round:', i + 1 , ' of', self.config['printsplit'])
                with Pool(processes = self.config['n_cpus']) as pool:
                    data_grid[(i * subrun_n * 1000):((i + 1) * subrun_n * 1000), :] = np.concatenate(pool.map(self._mlp_get_processed_data_for_theta, 
                                                                                                              [j for j in seeds[(i * subrun_n):((i + 1) * subrun_n)]]))
        else:
            data_grid = np.zeros((int(self.config['nparamsets'], self.config['nbins'], self.config['nchoices'])))
            
            for i in range(self.config['printsplit']):
                print('simulation round: ', i + 1, ' of', self.config['printsplit'])
                with Pool(processes = self.config['n_cpus']) as pool:
                    data_grid[(i * subrun_n): ((i + 1) * subrun_n), :, :] = np.concatenate(pool.map(self._cnn_get_processed_data_for_theta,
                                                                                                    [j for j in seeds[(i * subrun_n):((i + 1) * subrun_n)]]))
        
        if save:
            training_data_folder = self.config['method_folder'] + \
                                  'training_data_binned_' + \
                                  str(int(self.config['binned'])) + \
                                  '_nbins_' + str(self.config['nbins']) + \
                                  '_n_' + str(self.config['nsamples'])

            if not os.path.exists(training_data_folder):
                os.makedirs(training_data_folder)

            full_file_name = training_data_folder + '/' + \
                             'data_' + \
                             self.config['file_id'] + '.pickle'

            print('Writing to file: ', full_file_name)

            pickle.dump(np.float32(data_grid),
                        open(full_file_name, 'wb'), 
                        protocol = self.config['pickleprotocol'])
            

            return 'Dataset completed'
        
        else:
            return data_grid
                 
    def generate_data_parameter_recovery(self, save = False):
        
        # Make parameters
        theta_list = [np.float32((np.random.uniform(low = self.config['param_bounds'][0], 
                                                    high = self.config['param_bounds'][1])) for i in range(self.config['nparamsets']))]
        
        # Get simulations
        with Pool(processes = self.config['n_cpus']) as pool:
            data_grid = np.array(pool.map(self.get_simulations, theta_list))
         
        
        # Save to correct destination
        if save:
            
            # -----
            #if self.config['mode'] == 'test':
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

#             else:
#                 training_data_folder = self.config['method_folder'] + \
#                                       'training_data_binned_' + \
#                                       str(int(self.config['binned'])) + \
#                                       '_nbins_' + str(self.config['nbins']) + \
#                                       '_n_' + str(self.config['nsamples'])
                
#                 if not os.path.exists(training_data_folder):
#                     os.makedirs(training_data_folder)

#                 full_file_name = training_data_folder + '/' + \
#                                 self.config['method'] + \
#                                 '_nchoices_' + str(self.config['nchoices']) + \
#                                 '_train_data_binned_' + \
#                                 str(int(self.config['binned'])) + \
#                                 '_nbins_' + str(self.config['nbins']) + \
#                                 '_n_' + str(self.config['nsamples']) + \
#                                 '_' + self.config['file_id'] + '.pickle'
            
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

    def generate_rejected_parameterizations(self, 
                                            save = False):

        seeds = np.random.choice(400000000, size = self.config['nparamsetsrej'])

        # Get Simulations 
        with Pool(processes = self.config['n_cpus']) as pool:
            rejected_parameterization_list = pool.map(self._get_rejected_parameter_setups, 
                                                      seeds)
        rejected_parameterization_list = np.concatenate([l for l in rejected_parameterization_list if len(l) > 0])

        if save:
            training_data_folder = self.config['method_folder'] + \
                                  'training_data_binned_' + \
                                  str(int(self.config['binned'])) + \
                                  '_nbins_' + str(self.config['nbins']) + \
                                  '_n_' + str(self.config['nsamples'])

            if not os.path.exists(training_data_folder):
                os.makedirs(training_data_folder)

            full_file_name = training_data_folder + '/' + \
                             'rejected_parameterizations_' + \
                             self.config['file_id'] + '.pickle'

            print('Writing to file: ', full_file_name)

            pickle.dump(np.float32(rejected_parameterization_list),
                        open(full_file_name, 'wb'), 
                        protocol = self.config['pickleprotocol'])

            return 'Dataset completed'

        else:
            return data_grid
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
#     CLI.add_argument("--mode",
#                      type = str,
#                      default = 'test') # 'parameter_recovery, 'perturbation_experiment', 'r_sim', 'r_dgp', 'cnn_train', 'parameter_recovery_hierarchical'
    CLI.add_argument("--nsubjects",
                    type = int,
                    default = 5)
    CLI.add_argument("--nreps",
                     type = int,
                     default = 1)
    CLI.add_argument("--nbins",
                     type = int,
                     default = 0)
    CLI.add_argument("--nsamples",
                     type = int,
                     default = 20000)
    CLI.add_argument("--nchoices",
                     type = int,
                     default = 2)
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
    CLI.add_argument("--printsplit",
                     type = int,
                     default = 10)
    CLI.add_argument("--nbyparam",
                     type = int,
                     default = 1000)
    CLI.add_argument("--nparamsetsrej",
                     type = int,
                     default = 100)
    
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
        
    # config['mode'] = args.mode
    config['file_id'] = args.fileid
    config['nsamples'] = args.nsamples

    config['nbins'] = args.nbins
        
    if args.nbins > 0:
        config['binned'] = 1
    else:
        config['binned'] = 0
    
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
    config['printsplit'] = args.printsplit
    config['n_by_param'] = args.nbyparam
    config['nparamsetsrej'] = args.nparamsetsrej
    config['mixture_probabilities'] = [0.8, 0.1, 0.1]
    config['filter'] =  {'mode': 20, # != (if mode is max_rt)
                         'choice_cnt': 10, # > (each choice receive at least 10 samples in simulator)
                         'mean_rt': 15, # < (mean_rt is smaller than specified value
                         'std': 0, # > (std is positive for each choice)
                         'mode_cnt_rel': 0.5  # < (mode does not receive more than a proportion of samples for each choice)
                        }
    
    # Make parameter bounds
    if args.datatype == 'training' and config['binned']:
        bounds_tmp = kde_info.temp[config['method']]['param_bounds_cnn'] + kde_info.temp[config['method']]['boundary_param_bounds_cnn']
    elif args.datatype == 'training' and not config['binned']:
        bounds_tmp = kde_info.temp[config['method']]['param_bounds_network'] + kde_info.temp[config['method']]['boundary_param_bounds_network']
    else:
        bounds_tmp = kde_info.temp[config['method']]['param_bounds_sampler'] + kde_info.temp[config['method']]['boundary_param_bounds_sampler']

    config['param_bounds'] = np.array([[i[0] for i in bounds_tmp], [i[1] for i in bounds_tmp]])
    config['nparams'] = config['param_bounds'][0].shape[0]
    
    config['meta'] = dict(kde_info.temp[config['method']]['dgp_hyperparameters'])
    config['meta']['n_samples'] = config['n_samples']
    config['meta']['delta_t'] = config['delta_t']
    
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
    
    if args.datatype == 'parameter_recovery':
        dg.generate_data_parameter_recovery(save = args.save)
        
    if args.datatype == 'parameter_recovery_hierarchical':
        dg.generate_data_hierarchical(save = args.save)
        
    if args.datatype == 'training':
        dg.generate_data_training_uniform(save = args.save)
    
    if args.datatype == 'rej':
        dg.generate_rejected_parameterizations(save = args.save)

    finish_t = datetime.now()
    print('Time elapsed: ', finish_t - start_t)
    print('Finished')
    # -------------------------------------------------------------------------------------