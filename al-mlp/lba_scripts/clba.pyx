# Functions for DDM data simulation
import cython
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport log, sqrt, fmax

import numpy as np
import pandas as pd
from time import time
from scipy.stats import norm
import inspect

DTYPE = np.float32

# Method to draw random samples from a gaussian
cdef float random_uniform():
    cdef float r = rand()
    return r / RAND_MAX

cdef float random_gaussian():
    cdef float x1, x2, w
    w = 2.0

    while(w >= 1.0):
        x1 = 2.0 * random_uniform() - 1.0
        x2 = 2.0 * random_uniform() - 1.0
        w = x1 * x1 + x2 * x2

    w = ((-2.0 * log(w)) / w) ** 0.5
    return x1 * w

# cdef int sign(float x):
#     return (x > 0) - (x < 0)

# cdef float csum(float[:] x):
#     cdef int i
#     cdef int n = x.shape[0]
#     cdef float total = 0
    
#     for i in range(n):
#         total += x[i]
    
#     return total

@cython.boundscheck(False)
cdef void assign_random_gaussian_pair(float[:] out, int assign_ix):
    cdef float x1, x2, w
    w = 2.0

    while(w >= 1.0):
        x1 = 2.0 * random_uniform() - 1.0
        x2 = 2.0 * random_uniform() - 1.0
        w = x1 * x1 + x2 * x2

    w = ((-2.0 * log(w)) / w) ** 0.5
    out[assign_ix] = x1 * w
    out[assign_ix + 1] = x2 * 2

@cython.boundscheck(False)
cdef float[:] draw_gaussian(int n):
    # Draws standard normal variables - need to have the variance rescaled
    cdef int i
    cdef float[:] result = np.zeros(n, dtype = DTYPE)
    for i in range(n // 2):
        assign_random_gaussian_pair(result, i * 2)
    if n % 2 == 1:
        result[n - 1] = random_gaussian()
    return result

# Function generates choice data from lba
@cython.boundscheck(False)
@cython.wraparound(False)

def rlba(v = np.array([1, 1], dtype = DTYPE), 
         float A = 1.0, 
         float b = 1.5, 
         float s = 0.1,
         float ndt = 0.0,
         int n_samples = 1000,
         float max_t = 20.0,
         float d_lower_lim = 0.01):
    
    cdef int n_choices = len(v)
    cdef int c, m, n, i
    
    rts = np.zeros((n_samples, 1), dtype = DTYPE)
    cdef float[:,:] rts_view = rts
    
    choices = np.zeros((n_samples, 1), dtype = np.intc)
    cdef int[:,:] choices_view = choices
    
    cdef float tmp_rt, d, k, tmp
    
    cdef int num_draws = n_samples * n_choices * 2
    cdef float[:] gaussian_values = draw_gaussian(num_draws)
    
    m = 0
    i = 0
    while i < n_samples:
        c = 0
        tmp_rt = max_t + 1
        while c < n_choices:
            d = - 0.1
            
            while d < d_lower_lim:
                d = v[c] + (s * gaussian_values[m])
                
                m += 1
                if m == num_draws:
                    gaussian_values = draw_gaussian(num_draws)
                    m = 0
                    
            k = random_uniform() * A
            
            tmp = (b - k) / d
            
            if tmp < tmp_rt:
                tmp_rt = tmp
                rts_view[i] = tmp + ndt
                choices_view[i] = c
            c += 1
        i += 1
            
    
    # Create some dics
    v_dict = {}
    for i in range(n_choices):
        v_dict['v_' + str(i)] = v[i]

    return (rts, choices, {**v_dict,
                           'A': A,
                           'b': b,
                           's': s,
                           'ndt': ndt,
                           'delta_t': 0,
                           'max_t': max_t,
                           'n_samples': n_samples,
                           'simulator': 'lba',
                           'boundary_fun_type': 'none',
                           'possible_choices': [i for i in range(n_choices)]})


#Function computes probability of choice at rt provided parameters for all options
def batch_dlba2(rt = np.array([1,2,3]),
                choice = np.array([0, 1, 0]),
                v = np.array([1, 1]),
                A = 1,
                b = 1.5,
                s = 0.1,
                ndt = 0.0,
                return_log = True,
                eps = 1e-16):
    log_eps = np.log(eps)
    rt_higher_zero  = rt - ndt > 0
    zeros = len(rt) - np.sum(rt_higher_zero)
    rt = rt[rt_higher_zero] - ndt
    tmp = np.zeros((2, len(rt), 2))
    tmp[0, :, 0] = np.log(np.maximum(flba(rt = rt, A = A, b = b, v = v[0], s = s), eps))
    tmp[0, :, 1] = np.log(np.maximum(flba(rt = rt, A = A, b = b, v = v[1], s = s), eps))
    tmp[1, :, 0] = np.log(np.maximum(1 - Flba(rt = rt, A = A, b = b, v = v[1], s = s), eps))
    tmp[1, :, 1] = np.log(np.maximum(1 - Flba(rt = rt, A = A, b = b, v = v[0], s = s), eps))
    #tmp[tmp < log_eps] = log_eps
    tmp = tmp[0, :, :] + tmp [1, :, :]
    return np.sum(tmp[choice[rt_higher_zero] == 0, 0]) + np.sum(tmp[choice[rt_higher_zero] == 1, 1]) + (log_eps * zeros)


def dlba(rt = 0.5, 
         choice = 0,
         v = np.array([1, 1]),
         A = 1,
         b = 1.5,
         s = 0.1,
         ndt = 0.0,
         return_log = True,
         eps = 1e-16):
    
    rt = rt - ndt # adjusting rt values for non-decision time
    n_choices = len(v)
    l_f_t = 0
    
    if rt > 0:
        # Get probability of choice i at time t = rt
        for i in range(n_choices):
            if i == choice:
                tmp = flba(rt = rt, A = A, b = b, v = v[i], s = s)
                if tmp < eps:
                    tmp = eps
                l_f_t += np.log(tmp)
            else:
                tmp = Flba(rt = rt, v = v[i], A = A, b = b, s = s)

                # numerically robust l_f_t update
                if (1.0 - tmp) <= eps:
                    l_f_t += np.log(eps)
                else:
                    l_f_t += np.log(1.0 - tmp)
    else:
        # l_f_t += (np.log(eps) * n_choices) # n_choices seems wrong here
        l_f_t += np.log(eps)
            
    if return_log: 
        return l_f_t
    else:
        return np.exp(l_f_t)
    
# Function computes cdf of a given lba ray  
def Flba(rt = 0.5, 
         v = 1,
         A = 1,
         b = 1.5,
         s = 0.1):
    return (1 + ((1 / A) * ((b - A - (rt * v)) * norm.cdf((b - A - (rt * v)) / (rt * s)) - \
           (b - (rt * v)) * norm.cdf((b - (rt * v)) / (rt * s)) + \
           (rt * s) * (norm.pdf((b - A - (rt * v)) / (rt * s)) - norm.pdf((b - (rt * v)) / (rt * s))))))


# Function computes pdf of a given lba ray
def flba(rt = 0.5, 
         v = 1,
         A = 1,
         b = 1.5,
         s = 0.1):
    return ((1 / A) * ( (-v) * norm.cdf((b - A - (rt * v)) / (rt * s)) + \
                     s * norm.pdf((b - A - (rt * v)) / (rt * s)) + \
                     v * norm.cdf((b - (rt * v)) / (rt * s)) + \
                     (-s) * norm.pdf((b - (rt * v)) / (rt * s)) ))

# def Flba_batch(rt = [0.5, 1],
#                v = 1,
#                A = 1, 
#                b = 1.5,
#                s = 0.1):
#     return (1 + ((1 / A) * ((b - A - (rt * v)) * norm.cdf((b - A - (rt * v)) / (rt * s)) - \
#         (b - (rt * v)) * norm.cdf((b - (rt * v)) / (rt * s)) + \
#                     (rt * s) * (norm.pdf((b - A - (rt * v)) / (rt * s)) - norm.pdf((b - (rt * v)) / (rt * s))))))