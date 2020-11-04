# Environ
import scipy as scp
#import tensorflow as tf
#from scipy.stats import gamma
import numpy as np
import random
import sys
#import multiocessing as mp
#import psutil
import pickle
import os

# Own
#import ddm_data_simulation as ds
#import cddm_data_simulation as cds
#import kde_training_utilities as kde_util
#import kde_class as kde
#import boundary_functions as bf


def bin_simulator_output(out = [0, 0],
                         bin_dt = 0.04,
                         eps_correction = 1e-7,
                         params = ['v', 'a', 'w', 'ndt', 'angle']):

    # hardcode 'max_t' to 20sec for now
    n_bins = int(20.0 / bin_dt + 1)
    #n_bins = int(out[2]['max_t'] / bin_dt + 1)
    bins = np.linspace(0, out[2]['max_t'], n_bins)
    counts = []
    counts.append(np.histogram(out[0][out[1] == 1], bins = bins)[0] / out[2]['n_samples'])
    counts.append(np.histogram(out[0][out[1] == -1], bins = bins)[0] / out[2]['n_samples'])

    n_small = 0
    n_big = 0

    for i in range(len(counts)):
        n_small += sum(counts[i] < eps_correction)
        n_big += sum(counts[i] >= eps_correction)

    for i in range(len(counts)):
        counts[i][counts[i] <= eps_correction] = eps_correction
        counts[i][counts[i] > eps_correction] = counts[i][counts[i] > eps_correction] - (eps_correction * (n_small / n_big))    

    for i in range(len(counts)):
        counts[i] =  np.asmatrix(counts[i]).T

    labels = np.concatenate(counts, axis = 1)
    features = [out[2]['v'], out[2]['a'], out[2]['w'], out[2]['ndt']]
    return (features, labels)

files_ = os.listdir('/users/afengler/data/kde/angle/base_simulations_ndt_20000/')
random.shuffle(files_)

labels = np.zeros((len(files_) - 2, 500, 2))
features = np.zeros((len(files_) - 2, 4))
   
cnt = 0
i = 0
file_dim = 1000
for file_ in files_[:10100]:
    if file_[:4] == 'ddm_':
        out = pickle.load(open('/users/afengler/data/kde/angle/base_simulations_ndt_20000/' + file_, 'rb'))
        features[cnt], labels[cnt] = bin_simulator_output(out = out)
        cnt += 1
        if (cnt % file_dim) == 0 and cnt > 0:
            print(cnt)
            pickle.dump((labels[(i * file_dim):((i + 1) * file_dim)], features[(i * file_dim):((i + 1) * file_dim)]), 
                        open('/users/afengler/data/kde/angle/base_simulations_ndt_20000_binned/dataset_parallel_job_' + sys.argv[1] + '_' + str(i), 'wb'))
            i += 1
        




