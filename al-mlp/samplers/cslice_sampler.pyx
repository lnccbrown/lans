# Functions for DDM data simulation
import cython
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport log, sqrt, fmax
#from libcpp cimport bool

import numpy as np
import pandas as pd
from time import time
import inspect
import pickle

DTYPE = np.float32

# Method to draw random samples from a gaussian
cdef float random_uniform():
    cdef float r = rand()
    return r / RAND_MAX

# cdef float random_gaussian():
#     cdef float x1, x2, w
#     w = 2.0

#     while(w >= 1.0):
#         x1 = 2.0 * random_uniform() - 1.0
#         x2 = 2.0 * random_uniform() - 1.0
#         w = x1 * x1 + x2 * x2

#     w = ((-2.0 * log(w)) / w) ** 0.5
#     return x1 * w

# cdef int sign(float x):
#     return (x > 0) - (x < 0)

# cdef float csum(float[:] x):
#     cdef int i
#     cdef int n = x.shape[0]
#     cdef float total = 0
    
#     for i in range(n):
#         total += x[i]
    
#     return total

class SliceSampler:
    cdef int dims, p, m, sample_cnt
    cdef float w
    cdef float[:, :] bounds
    cdef float[:] left
    cdef float[:] right
    
    def __init__(self, 
                 bounds,
                 target,
                 float w = 1 / 256,
                 int p = 8,
                 int m = 50,
                 print_interval = 1,
                 data = []):
        
        self.dims = len(bounds)
        self.bounds = np.float32(np.array(bounds))
        self.left = np.zeros(self.dims, dtype = DTYPE)
        self.right = np.zeros(self.dims, dtype = DTYPE)
        # self.target = target
        self.data = data
        self.w = w
        self.p = p
        self.m = m
        self.print_interval = print_interval
        self.sample_cnt = 0
        
    cdef void _find_interval_doubling(self, 
                                      float z, 
                                      float[:] prev, 
                                      int dim):
         
        #
        self.left = prev.copy()
        self.right = prev.copy()
        
        #
        self.left[dim] = prev[dim] - self.w * random_uniform()
        self.right[dim] = self.left[dim] + self.w
        
        cdef int k = self.p
        
        cdef float lp_l = self.target(self.left, self.data)
        cdef float lp_r = self.target(self.right, self.data)
        
        while (k > 0 and (z < lp_l or z < lp_r)):
            #u = random_uniform()
            if random_uniform() < .5:
                self.left[dim] += self.left[dim] - self.right[dim]
                if self.left[dim] <= self.bounds[dim, 0]:
                    self.left[dim] = self.bounds[dim, 0]
                if self.left[dim] >= self.bounds[dim, 1]:
                    self.left[dim] = self.bounds[dim, 1]
                    
                lp_l = self.target(self.left, self.data)
                
            else:
                self.right[dim] += self.right[dim] - self.left[dim]
                if self.right[dim] <= self.bounds[dim, 0]:
                    self.right[dim] = self.bounds[dim, 0]
                if self.right[dim] >= self.bounds[dim, 1]:
                    self.right[dim] = self.bounds[dim, 1]
            
            k -= 1
        return
         
    cdef int _accept(self, float[:] prev, float[:] upd, float z, int dim): #float[:]left, right, dim):
        
        # Initialize
        cdef float [:] tmp_upd_left = prev.copy()
        cdef float [:] tmp_upd_right = prev.copy()
        cdef int D = 0
        cdef float M
        
        
        tmp_upd_left[dim] = self.left[dim]
        tmp_upd_right[dim] = self.right[dim]

        while (tmp_upd_right[dim] - tmp_upd_left[dim]) > (1.1 * self.w):
            M = (tmp_upd_left[dim] + tmp_upd_right[dim]) / 2

            if (prev[dim] < M and upd[dim] >= M) or (prev[dim] >= M and upd[dim] < M):
                D = 1

            if upd[dim] < M:
                tmp_upd_right[dim] = M

            else:
                tmp_upd_left[dim] = M

            if D and z >= self.target(tmp_upd_left, self.data) and z >= self.target(tmp_upd_right, self.data):
                return 0

        return 1
      
    cdef void _slice_sample_doubling(self, prev, prev_lp):
        cdef float[:] out = prev.copy()
        cdef float lp = prev_lp
        cdef float z 
        #cdef float[:] out = left
        
        for dim in range(self.dims):
            z = prev_lp - np.float32(np.random.exponential())
            #left, right = self._find_interval_doubling(z, prev, dim)
            
            # Make initial interval
            self._find_interval_doubling(z, prev, dim)
            
            # Adaptively shrink the interval
            while not np.isclose(self.left, self.right, 1e-5): # This condition might be unnecessary
                #u = random_uniform()
                out[dim] = self.left[dim] + (random_uniform() * (self.right[dim] - self.left[dim]))
                lp = self.target(out, self.data)
                if z < lp and self._accept(prev, out, z, dim): # left, right, dim):
                    break
                else:
                    if out[dim] < prev[dim]:
                        self.left[dim] = out[dim]
                    else:
                        self.right[dim] = out[dim]
        
        # Store samples
        self.samples[self.sample_cnt, :] = out
        self.lp[self.sample_cnt] = lp
        
       # Sampling wrapper
    def sample(self, 
               int num_samples = 1000,
               add = False, 
               str method = 'doubling', 
               init = 'random'):
        
#         # Initialize data
#         self.data = data
        cdef int id_start = 1
        cdef float init_lp
        self.samples = np.zeros((num_samples, self.dims), dtype = DTYPE) # samples
        self.lp = np.zeros(num_samples, dtype = DTYPE) # sample log likelihoods
        tmp = np.zeros(self.dims, dtype = DTYPE)

        # New sampler ? 
        if add == False:
            # Initialize sample storage

            # Random initialization of starting points
            if init[0] == 'r':
                for dim in range(self.dims):
                    tmp[dim] = self.bounds[dim, 0] + random_uniform() * (self.bounds[dim,1] - self.bounds[dim, 0])   
            
            else:
                tmp = init
                
            init_lp = self.target(tmp, self.data)
            
            # Make first sample
            if method == 'doubling':
                self._slice_sample_doubling(tmp, init_lp)
            if method == 'step_out':
                self._slice_sample_step_out(tmp, init_lp)
        
        # Or do we add samples to previous samples ?
#         else: 
#             # choose appropriate starting idx 
#             id_start = self.samples.shape[0]
            
#             # Increase size so sample container
#             float[:] tmp_samples = np.zeros((self.samples.shape[0] + num_samples, self.dims))
#             float[:] tmp_samples[:self.samples.shape[0], :] = self.samples
#             self.samples = tmp_samples
            
#             float[:] tmp_lp = np.zeros(self.lp.shape[0] + num_samples)
#             tmp_lp[:self.lp.shape[0]] = self.lp
#             float[:]self.lp = tmp_lp
            
#             print(self.samples[10])

#             print('Adding to previous samples...')
        
        print('Beginning sampling...')
        final_n_samples = self.samples.shape[0]
        cdef int i = id_start
        
        while i < final_n_samples:
            self.sample_cnt = i
            if method == 'doubling':
                self._slice_sample_doubling(self.samples[i - 1, :], self.lp[i - 1])
#             if method == 'step_out':
#                 self._slice_sample_step_out(self.samples[i - 1], self.lp[i - 1])
#             if i % self.print_interval == 0:
#                 print("Iteration {}".format(i))         
            i += 1