import numpy as np
import scipy as scp
#import ddm_data_simulation as ddm_data_simulator
import scipy.integrate as integrate

from libc.math cimport sin, exp, sqrt, M_PI, fmax, fmin, log

# WFPT NAVARROS FUSS -------------
# Large-time approximation to fpt distribution
cdef double fptd_large(double t, double w, int k):
    cdef double fptd_sum = 0
    cdef int i
    for i in range(1, k + 1):
        fptd_sum += i * exp( - ((i**2) * (M_PI ** 2) * t) / 2) * sin(i * M_PI * w)
    return fptd_sum * M_PI

# Small-time approximation to fpt distribution
cdef double fptd_small(double t, double w, int k):
    cdef double temp = ((k - 1) / 2)
    
    # This is what I think makes more sense... second equation
    cdef int upper_k = np.ceil(temp)
    cdef int lower_k = - np.floor(temp)
    
    cdef double fptd_sum = 0
    cdef int i

    for i in range(lower_k, upper_k + 1, 1):
        fptd_sum += (w + (2 * i)) * exp(- ((w + (2 * i))**2) / (2 * t))
    return fptd_sum * (1 / sqrt(2 * M_PI * (t**3)))

# Leading term (shows up for both large and small time)
cdef double calculate_leading_term(double t, double v, double a, double w):
    return 1 / (a**2) * exp( - (v * a * w) - (((v**2) * t) / 2))

cdef double calculate_leading_term_with_drift_noise(double t, double v, double a, double w, double sdv):
    return (1 / ((a**2) * sqrt((sdv**2) * t + 1))) * exp((((a * w * sdv)**2) - ( 2 * a * v * w ) - ( (v**2) * t )) / ((2 * (sdv**2) * t) + 2))

# Choice function to determine which approximation is appropriate (small or large time)
cdef choice_function(double t, double eps):
    eps_l = fmin(eps, 1 / (t * M_PI))
    eps_s = fmin(eps, 1 / (2 * sqrt(2 * M_PI * t)))
    k_l = np.ceil(fmax(sqrt( - (2 * log(M_PI * t * eps_l)) / (M_PI**2 * t)), 1 / (M_PI * sqrt(t))))
    k_s = np.ceil(fmax(2 + sqrt( - 2 * t * log(2 * eps_s * sqrt(2 * M_PI * t))), 1 + sqrt(t)))
    return k_s - k_l, k_l, k_s

# Actual fptd (first-passage-time-distribution) algorithm
def fptd(t = 0, v = 0, a = 1, w = 0.5, eps=1e-10):
    # negative reaction times signify upper boundary crossing
    # we have to change the parameters as suggested by navarro & fuss (2009)
    if t < 0:
        v = - v
        w = 1 - w
        t = np.abs(t)

    #print('lambda: ' + str(sgn_lambda))
    #print('k_s: ' + str(k_s))
    if t != 0:
        leading_term = calculate_leading_term(t, v, a, w)
        t_adj = t / (a**2)
        sgn_lambda, k_l, k_s = choice_function(t_adj, eps)
        if sgn_lambda >= 0:
            return max(eps, leading_term * fptd_large(t_adj, w, k_l))
        else:
            return max(eps, leading_term * fptd_small(t_adj, w, k_s))
    else:
        return 1e-29
# --------------------------------

def batch_fptd(t, double v = 1, double a = 1, double w = 0.5, double ndt = 1.0, double sdv = 0, double eps = 1e-48):
    # Use when rts and choices vary, but parameters are held constant
    cdef int i
    cdef double[:] t_view = t
    cdef int n = t.shape[0]
    
    # NEW
    leading_terms = np.zeros(n)
    cdef double[:] leading_terms_view = leading_terms

    likelihoods = np.zeros(n)
    cdef double[:] likelihoods_view = likelihoods

    for i in range(n):
        if t_view[i] == 0:
            likelihoods_view[i] = 1e-48
        elif t_view[i] < 0:
            t_view[i] = (-1) * t_view[i]
            
            # adjust rt for ndt
            t_view[i] = t_view[i] - ndt
            
            if t_view[i] <= 0:
                likelihoods_view[i] = 1e-48
            else:
                sgn_lambda, k_l, k_s = choice_function(t_view[i] / (a**2), eps)
                
                if sdv == 0:
                    leading_terms_view[i] = calculate_leading_term(t_view[i], (-1) * v, a, 1 - w)
                else:
                    leading_terms_view[i] = calculate_leading_term_with_drift_noise(t_view[i], (-1) * v, a, 1 - w, sdv)
            
                if sgn_lambda >= 0:
                    likelihoods_view[i] = fmax(1e-48, leading_terms_view[i] * fptd_large(t_view[i] / (a**2),
                        1 - w, k_l))
                else:
                    likelihoods_view[i] = fmax(1e-48, leading_terms_view[i] * fptd_small(t_view[i] / (a**2),
                        1 - w, k_s))
        elif t_view[i] > 0:
            # adjust rt for ndt
            t_view[i] = t_view[i] - ndt
            
            if t_view[i] <= 0:
                likelihoods_view[i] = 1e-48
            else:
                sgn_lambda, k_l, k_s = choice_function(t_view[i] / (a**2), eps)
                
                if sdv == 0:
                    leading_terms_view[i] = calculate_leading_term(t_view[i], v, a, w)
                else:
                    leading_terms_view[i] = calculate_leading_term_with_drift_noise(t_view[i], v, a, w, sdv)
                 
                if sgn_lambda >= 0:
                    likelihoods_view[i] = fmax(1e-48, leading_terms_view[i] * fptd_large(t_view[i] / (a**2),
                        w, k_l))
                else:
                    likelihoods_view[i] = fmax(1e-48, leading_terms_view[i] * fptd_small(t_view[i] / (a**2),
                        w, k_s))
    
    return likelihoods

def array_fptd(t, v, a, w, double eps=1e-9):
    # Use when all inputs vary, but we want to feed in an array of them
    cdef int i
    cdef int n = t.shape[0]
    cdef double[:] t_view = t
    cdef double[:] v_view = v
    cdef double[:] a_view = a
    cdef double[:] w_view = w

    likelihoods = np.zeros(n)
    cdef double[:] likelihoods_view = likelihoods

    for i in range(n):
        if t_view[i] == 0:
            likelihoods_view[i] = 1e-48
        elif t_view[i] < 0:
            t_view[i] = (-1) * t_view[i]

            sgn_lambda, k_l, k_s = choice_function(t_view[i], eps)
            leading_term = calculate_leading_term(t_view[i], (-1) * v_view[i], a_view[i],
                    1 - w_view[i])
            if sgn_lambda >= 0:
                likelihoods_view[i] = fmax(1e-48, leading_term * fptd_large(
                    t_view[i] / (a_view[i]**2), 1 - w_view[i], k_l))
            else:
                likelihoods_view[i] = fmax(1e-48, leading_term * fptd_small(
                    t_view[i] / (a_view[i]**2), 1 - w_view[i], k_s))
        elif t_view[i] > 0:
            sgn_lambda, k_l, k_s = choice_function(t_view[i], eps)
            leading_term = calculate_leading_term(t_view[i], v_view[i], 
                    a_view[i], w_view[i])
            if sgn_lambda >= 0:
                likelihoods_view[i] = fmax(1e-48, leading_term * fptd_large(
                    t_view[i] / (a_view[i]**2), w_view[i], k_l))
            else:
                likelihoods_view[i] = fmax(1e-48, leading_term * fptd_small(
                    t_view[i] / (a_view[i]**2), w_view[i], k_s))

    return likelihoods