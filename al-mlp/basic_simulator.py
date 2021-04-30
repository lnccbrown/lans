import pandas as pd
import numpy as np
#import re
import argparse
import sys
import pickle
from cddm_data_simulation import ddm 
from cddm_data_simulation import ddm_flexbound
from cddm_data_simulation import levy_flexbound
from cddm_data_simulation import ornstein_uhlenbeck
from cddm_data_simulation import full_ddm
from cddm_data_simulation import ddm_sdv
from cddm_data_simulation import ddm_flexbound_pre
from cddm_data_simulation import race_model
from cddm_data_simulation import lca
import cddm_data_simulation as cds
import boundary_functions as bf


def bin_simulator_output(out = None,
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

def bin_arbitrary_fptd(out = None,
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



def simulator(theta, 
              model = 'angle', 
              n_samples = 1000, 
              delta_t = 0.001,
              max_t = 20,
              bin_dim = None):
    
    # Useful for sbi
    if type(theta) == list or type(theta) == np.ndarray:
        pass
    else:
        theta = theta.numpy()
    
    if model == 'ddm':
        x = ddm_flexbound(v = theta[0],
                          a = theta[1], 
                          w = theta[2],
                          ndt = theta[3],
                          n_samples = n_samples,
                          delta_t = delta_t,
                          boundary_params = {},
                          boundary_fun = bf.constant,
                          boundary_multiplicative = True,
                          max_t = max_t)
    
    if model == 'angle' or model == 'angle2':
        x = ddm_flexbound(v = theta[0], 
                          a = theta[1],
                          w = theta[2], 
                          ndt = theta[3], 
                          boundary_fun = bf.angle, 
                          boundary_multiplicative = False,
                          boundary_params = {'theta': theta[4]}, 
                          delta_t = delta_t,
                          n_samples = n_samples,
                          max_t = max_t)
    
    if model == 'weibull_cdf' or model == 'weibull_cdf2' or model == 'weibull_cdf_ext' or model == 'weibull_cdf_concave':
        x = ddm_flexbound(v = theta[0], 
                          a = theta[1], 
                          w = theta[2], 
                          ndt = theta[3], 
                          boundary_fun = bf.weibull_cdf, 
                          boundary_multiplicative = True, 
                          boundary_params = {'alpha': theta[4], 'beta': theta[5]}, 
                          delta_t = delta_t,
                          n_samples = n_samples,
                          max_t = max_t)
    
    if model == 'levy':
        x = levy_flexbound(v = theta[0], 
                           a = theta[1], 
                           w = theta[2], 
                           alpha_diff = theta[3], 
                           ndt = theta[4], 
                           boundary_fun = bf.constant, 
                           boundary_multiplicative = True, 
                           boundary_params = {},
                           delta_t = delta_t,
                           n_samples = n_samples,
                           max_t = max_t)
    
    if model == 'full_ddm' or model == 'full_ddm2':
        x = full_ddm(v = theta[0],
                     a = theta[1],
                     w = theta[2], 
                     ndt = theta[3], 
                     dw = theta[4], 
                     sdv = theta[5], 
                     dndt = theta[6], 
                     boundary_fun = bf.constant, 
                     boundary_multiplicative = True, 
                     boundary_params = {}, 
                     delta_t = delta_t,
                     n_samples = n_samples,
                     max_t = max_t)

    if model == 'ddm_sdv':
        x = ddm_sdv(v = theta[0], 
                    a = theta[1], 
                    w = theta[2], 
                    ndt = theta[3],
                    sdv = theta[4],
                    boundary_fun = bf.constant,
                    boundary_multiplicative = True, 
                    boundary_params = {},
                    delta_t = delta_t,
                    n_samples = n_samples,
                    max_t = max_t)
        
    if model == 'ornstein':
        x = ornstein_uhlenbeck(v = theta[0], 
                               a = theta[1], 
                               w = theta[2], 
                               g = theta[3], 
                               ndt = theta[4],
                               boundary_fun = bf.constant,
                               boundary_multiplicative = True,
                               boundary_params = {},
                               delta_t = delta_t,
                               n_samples = n_samples,
                               max_t = max_t)

    if model == 'pre':
        x = ddm_flexbound_pre(v = theta[0],
                              a = theta[1], 
                              w = theta[2], 
                              ndt = theta[3],
                              boundary_fun = bf.angle,
                              boundary_multiplicative = False,
                              boundary_params = {'theta': theta[4]},
                              delta_t = delta_t,
                              n_samples = n_samples,
                              max_t = max_t)
        
    if model == 'race_model_3':
        x = race_model(v = theta[:3],
                       a = theta[3],
                       w = theta[4:7],
                       ndt = theta[7],
                       s = np.array([1, 1, 1], dtype = np.float32),
                       boundary_fun = bf.constant,
                       boundary_multiplicative = True,
                       boundary_params = {},
                       delta_t = delta_t,
                       n_samples = n_samples,
                       max_t = max_t)
        
    if model == 'race_model_4':
        x = race_model(v = theta[:4],
                       a = theta[4],
                       w = theta[5:9],
                       ndt = theta[9],
                       s = np.array([1, 1, 1, 1], dtype = np.float32),
                       boundary_fun = bf.constant,
                       boundary_multiplicative = True,
                       boundary_params = {},
                       delta_t = delta_t,
                       n_samples = n_samples,
                       max_t = max_t)
        
    if model == 'lca_3':
        x = lca(v = theta[:3],
                a = theta[4],
                w = theta[4:7],
                g = theta[7],
                b = theta[8],
                ndt = theta[9],
                s = 1.0,
                boundary_fun = bf.constant,
                boundary_multiplicative = True,
                boundary_params = {},
                delta_t = delta_t,
                n_samples = n_samples,
                max_t = max_t)
        
    if model == 'lca_4':
        x = lca(v = theta[:4],
                a = theta[4],
                w = theta[5:9],
                g = theta[9],
                b = theta[10],
                ndt = theta[11],
                s = 1.0,
                boundary_fun = bf.constant,
                boundary_multiplicative = True,
                boundary_params = {},
                delta_t = delta_t,
                n_samples = n_samples,
                max_t = max_t)
    
    if bin_dim == 0:
        return x
    else:
        return bin_simulator_output(x, nbins = bin_dim)
    