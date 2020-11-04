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
import argparse

# My own code
import kde_class as kde
#import cddm_data_simulation as ddm_simulator 
import boundary_functions as bf
import kde_training_utilities as kde_utils

if __name__ == "__main__":
    # Interface ------
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--machine",
                     type = str,
                     default = 'ccv')
    CLI.add_argument("--method",
                     type = str,
                     default = 'ddm')
    CLI.add_argument("--simfolder",
                     type = str,
                     default = 'base_simulations')
    CLI.add_argument("--fileprefix",
                     type = str,
                     default = 'ddm_base_simulations')
    CLI.add_argument("--fileid",
                     type = str,
                     default = 'TEST')
    args = CLI.parse_args()
    print(args)
    
    # Specify base simulation folder ------
    if args.machine == 'x7':
        method_params = pickle.load(open("/media/data_cifs/afengler/git_repos/nn_likelihoods/kde_stats.pickle",
                                         "rb"))[args.method]
        base_simulation_folder = method_params['method_folder_x7'] + args.simfolder +'/'
        
    if args.machine == 'ccv':
        method_params = pickle.load(open("/users/afengler/git_repos/nn_likelihoods/kde_stats.pickle", 
                                         "rb"))[args.method]
        base_simulation_folder = method_params['method_folder'] + args.simfolder + '/'
        
    if args.machine == 'home':
        method_params = pickle.load(open("/Users/afengler/OneDrive/git_repos/nn_likelihoods/kde_stats.pickle", "rb"))[args.method]
        base_simulation_folder = method_params['method_folder_home'] + args.simfolder + '/'
        
    # FILTERS: GENERAL
    filters = {'mode': 20, # != 
               'choice_cnt': 10, # > 
               'mean_rt': 18, # < 
               'std': 0, # > 
               'mode_cnt_rel': 0.5  # < 
              }
    
    # Run filter new
    start_time = time.time()
    kde_utils.filter_simulations_fast(base_simulation_folder = base_simulation_folder,
                                      file_name_prefix = args.fileprefix,
                                      file_id = args.fileid,
                                      method_params = method_params,
                                      param_ranges = 'none',
                                      filters = filters)
    
    end_time = time.time()
    exec_time = end_time - start_time
    print('Time elapsed: ', exec_time)