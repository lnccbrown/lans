import numpy as np
from scipy.stats import norm

# Function generates choice data from lba
def rlba(v = np.array([1, 1]), 
         A = 1, 
         b = 1.5, 
         s = 0.1,
         ndt = 0.0,
         n_samples = 1000,
         max_t = 20,
         d_lower_lim = 0.01):
    
    rts = np.zeros((n_samples, 1))
    choices = np.zeros((n_samples, 1))
    
    n_choices = len(v)
    for i in range(n_samples):
        d = np.array([- 0.1] * n_choices)
        while np.max(d) < d_lower_lim:
            k = np.random.uniform(low = 0, high = A, size = n_choices)
            d = np.random.normal(loc = v, scale = s)
            tmp_rt = (b - k) / d
        
        rts[i] = np.min(tmp_rt) + ndt
        choices[i] = np.argmin(tmp_rt)
    
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


# Function computes probability of choice at rt provided parameters for all options
def dlba(rt = 0.5, 
         choice = 0,
         v = np.array([1, 1]),
         A = 1,
         b = 1.5,
         s = 0.1,
         return_log = True,
         eps = 1e-16):
    
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
        l_f_t += (np.log(eps) * n_choices)
            
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