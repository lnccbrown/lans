# Basic python utilities
import numpy as np
import scipy as scp

# Parallelization
import multiprocessing as mp
from  multiprocessing import Process
from  multiprocessing import Pool
import psutil

# System utilities
import os
import pickle
import uuid

# My own code
from clba import rlba

def data_generator(*args):
    # CHOOSE SIMULATOR HERE
    simulator_data = rlba(*args)
    
    # CHOOSE TARGET DIRECTORY HERE
    file_dir =  '/users/afengler/data/kde/lba/base_simulations_ndt_20000/'
    
    # USE FOR x7 MACHINE 
    #file_dir = '/media/data_cifs/afengler/tmp/'

    # STORE
    file_name = file_dir + simulator + '_' + uuid.uuid1().hex
    pickle.dump(simulator_data, open( file_name + '.pickle', "wb" ) )
    print(args)

if __name__ == "__main__":
    # Get cpu cnt
    n_cpus = psutil.cpu_count(logical = False)

    # Parameter ranges (for the simulator)
    A = [0, 1]
    b = [1.5, 3]
    v = [1, 2]
    s = [0.1, 0.2]
    ndt = [0, 0.1]
    
    # Simulator parameters
    simulator = 'lba'
    n_choices = 2
    n_samples = 20000

    # Number of simulators to run
    n_simulators = 500000

    v_sample = []
    for i in range(n_simulators):
        v_tmp = []
        
        for j in range(n_choices):
            v_tmp.append(np.random.uniform(low = v[0], high = v[1]))

        v_sample.append(np.array(v_tmp, dtype = np.float32))

    A_sample = np.random.uniform(low = A[0], high = A[1], size = n_simulators)
    b_sample = np.random.uniform(low = b[0], high = b[1], size = n_simulators)
    s_sample = np.random.uniform(low = s[0], high = s[1], size = n_simulators)
    ndt_sample = np.random.uniform(low = ndt[0], high = ndt[1], size = n_simulators)
    

    args_list = []
    for i in range(n_simulators):
        # Get current set of parameters
        process_params = (v_sample[i], A_sample[i], b_sample[i], s_sample[i], ndt_sample[i])
        sampler_params = (n_samples,)
        
        # Append argument list with current parameters
        args_tmp = process_params + sampler_params
        args_list.append(args_tmp)

    # Parallel Loop
    with Pool(processes = n_cpus) as pool:
        res = pool.starmap(data_generator, args_list)