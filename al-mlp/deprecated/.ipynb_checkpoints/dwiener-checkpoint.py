import math
import numpy as np
import scipy as scp
import pandas as pd
from datetime import datetime
import glob
import ddm_data_simulation as ddm_data_simulator
import scipy.integrate as integrate

# WFPT NAVARROS FUSS -------------
# Large-time approximation to fpt distribution
def fptd_large(t, w, k):
    terms = np.arange(1, k+1, 1)
    fptd_sum = 0

    for i in terms:
        fptd_sum += i * np.exp( - (math.pow(i,2) * math.pow(np.pi, 2) * t) / 2) * np.sin(i * np.pi * w)
    return fptd_sum * np.pi

# Small-time approximation to fpt distribution
def fptd_small(t, w, k):
    temp = (k-1) / 2
    flr = np.floor(temp).astype(int)
    cei = - np.ceil(temp).astype(int)
    terms = np.arange(cei, flr + 1, 1)
    #print(terms)
    fptd_sum = 0

    for i in terms:
        fptd_sum += (w + 2 * i) * np.exp( - math.pow(w + 2 * i, 2) / (2 * t))
    return fptd_sum * (1 / np.sqrt(2 * np.pi * math.pow(t, 3)))

# Leading term (shows up for both large and small time)
def calculate_leading_term(t, v, a ,w):
    return 1 / math.pow(a, 2) * np.exp(- (v * a * w) - ((math.pow(v, 2) * t) / 2))

# Choice function to determine which approximation is appropriate (small or large time)
def choice_function(t, eps):
    eps_l = min(eps, 1 / (t * np.pi))
    eps_s = min(eps, 1 / (2 * np.sqrt(2 * np.pi * t)))

    k_l = np.ceil(max(np.sqrt(- (2 * np.log(np.pi * t * eps_l)) / (np.pi**2 * t)), 1 / (np.pi * np.sqrt(t))))
    k_s = np.ceil(max(2 + np.sqrt(- 2 * t * np.log(2 * eps_s * np.sqrt(2 * np.pi * t))), 1 + np.sqrt(t)))
    return k_s - k_l, k_l, k_s

# Actual fptd (first-passage-time-distribution) algorithm
def fptd(t,
         v = 0.0,
         a = 1.0,
         w = 0.5,
         eps = 1e-10 # potentially have min_l != eps as a minimum likelihood
         ):

    # negative reaction times signify upper boundary crossing
    # we have to change the parameters as suggested by navarro & fuss (2009)
    if t < 0:
       v = - v
       w = 1 - w
       t = np.abs(t)

    if t != 0:
        t_adj =  t / math.pow(a, 2)
        leading_term = calculate_leading_term(t, v, a, w)
        sgn_lambda, k_l, k_s = choice_function(t_adj, eps)

        if sgn_lambda >= 0:
            return max(eps, leading_term * fptd_large(t_adj, w, k_l))
        else:
            return max(eps, leading_term * fptd_small(t_adj, w, k_s))
    else:
        return 1e-29
# ---------------------------------------

# Calculation of choice probabilities
# ---------------------------------------

def choice_probabilities(v = 0.0,
                         a = 1.0,
                         w = 0.5,
                         eps = 1e-10,
                         allow_analytic = True
                         ):
    if w == 0.5 and allow_analytic:
        return choice_probabilities_analytic(v, a)
    return integrate.quad(dwiener.fptd, 0, 100, args = (v, a, w, eps))[0]

def choice_probabilities_analytic(v = 0.0, a = 1.0):
    return (1 / (1 + np.exp(v * a)))

# ----------------------------------------
