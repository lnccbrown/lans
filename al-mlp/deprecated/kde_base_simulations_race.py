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

# My own code
import kde_class as kde
import ddm_data_simulation as ddm_simulator
import boundary_functions as bf

def data_generator(*args):
    # CHOOSE SIMULATOR HERE
    simulator_data = ddm_simulator.race_model(*args)
    
    # CHOOSE TARGET DIRECTORY HERE
    file_dir =  'users/afengler/data/kde/race_2/base_simulations/'
    
    # STORE
    file_name = file_dir + simulator + '_' + uuid.uuid1().hex
    pickle.dump(simulator_data, open( file_name + '.pickle', "wb" ) )
    print('success')

if __name__ == "__main__":
    # Get cpu cnt
    n_cpus = psutil.cpu_count(logical = False)
    
    # Number of particles 
    n_particles = 2
    
    # Parameter ranges (for the simulator)
    v = [0.0, 2.0]
    w = [0.0, 0.7]
    a = [1, 3]

    # Simulator parameters
    simulator = 'race'
    s = 1
    delta_t = 0.01
    max_t = 20
    n_samples = 10000
    print_info = False
    boundary_multiplicative = True

    # Number of kdes to generate
    n_kdes = 2500

    # Make function input tuples
    v_sample = np.random.uniform(low = v[0], high = v[1], size = (n_kdes, n_particles) )
    w_sample = np.random.uniform(low = w[0], high = w[1], size = (n_kdes, n_particles) )
    a_sample = np.random.uniform(low = a[0], high = a[1], size = n_kdes)

    # Defining main function to iterate over:
    # Folder in which we would like to dump files

    args_list = []
    for i in range(0, n_kdes, 1):
        args_list.append((v_sample[i, :],
                          a_sample[i],
                          w_sample[i, :],
                          s,
                          delta_t,
                          max_t,
                          n_samples,
                          print_info,
                          bf.constant,
                          boundary_multiplicative,
                          {}))

    # Parallel Loop
    with Pool(processes = n_cpus) as pool:
        res = pool.starmap(data_generator, args_list)
