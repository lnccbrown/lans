# Functions for DDM data simulation
import numpy as np
import pandas as pd
import time
import inspect

# Simulate (rt, choice) tuples from: SIMPLE DDM -----------------------------------------------

# Simplest algorithm
def  ddm(v = 0, # drift by timestep 'delta_t'
         a = 1, # boundary separation
         w = 0.5,  # between 0 and 1
         s = 1, # noise sigma
         delta_t = 0.001, # timesteps fraction of seconds
         max_t = 20, # maximum rt allowed
         n_samples = 20000, # number of samples considered
         print_info = True # timesteps fraction of seconds
         ):

    rts = np.zeros((n_samples, 1))
    choices = np.zeros((n_samples, 1))

    delta_t_sqrt = np.sqrt(delta_t)

    for n in range(0, n_samples, 1):
        y = w * a
        t = 0

        while y <= a and y >= 0 and t <= max_t:
            y += v * delta_t + delta_t_sqrt * np.random.normal(loc = 0,
                                                               scale = s,
                                                               size = 1)
            t += delta_t

        # Store choice and reaction time
        rts[n] = t
        # Note that for purposes of consistency with Navarro and Fuss, the choice corresponding the lower barrier is +1, higher barrier is -1
        choices[n] = (-1) * np.sign(y)

        if print_info == True:
            if n % 1000 == 0:
                print(n, ' datapoints sampled')

    return (rts, choices, {'v': v,
                           'a': a,
                           'w': w,
                           's': s,
                           'delta_t': delta_t,
                           'max_t': max_t,
                           'n_samples': n_samples,
                           'simulator': 'ddm',
                           'boundary_fun_type': 'constant',
                           'possible_choices': [-1, 1]})
# -----------------------------------------------------------------------------------------------

# Simulate (rt, choice) tuples from: DDM WITH ARBITRARY BOUNDARY --------------------------------

# For the flexbound (and in fact for all other dd)
# We expect the boundary function to have the following general shape:
# 1. An initial separation (using the a parameter) with lower bound 0
# (this allows interpretation of the w parameter as usual)
# 2. No touching of boundaries
# 3. It return upper and lower bounds for every t as a list [upper, lower]

def ddm_flexbound(v = 0,
                  a = 1,
                  w = 0.5,
                  s = 1,
                  delta_t = 0.001,
                  max_t = 20,
                  n_samples = 20000,
                  print_info = True,
                  boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
                  boundary_multiplicative = True,
                  boundary_params = {'p1': 0, 'p2':0}
                  ):

    # Initializations
    rts = np.zeros((n_samples,1)) #rt storage
    choices = np.zeros((n_samples,1)) # choice storage
    delta_t_sqrt = np.sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion

    # Boundary storage for the upper bound
    num_steps = int((max_t / delta_t) + 1)
    boundary = np.zeros(num_steps)

    # Precompute boundary evaluations
    if boundary_multiplicative:
        for i in range(num_steps):
            tmp = a * boundary_fun(t = i * delta_t, **boundary_params)
            if tmp > 0:
                boundary[i] = tmp
    else:
        for i in range(num_steps):
            tmp = a + boundary_fun(t = i * delta_t, **boundary_params)
            if tmp > 0:
                boundary[i] = tmp

    # Outer loop over n - number of samples
    for n in range(0, n_samples, 1):
        # initialize y, t, and time_counter
        y = (- 1) * boundary[0] + (w * 2 * boundary[0])
        t = 0
        cnt = 0

        # Inner loop (trajection simulation)
        while y <= boundary[cnt] and y >= ((- 1) * boundary[cnt]) and t < max_t:
            # Increment y position (particle position)
            y += v * delta_t + delta_t_sqrt * np.random.normal(loc = 0,
                                                               scale = s,
                                                               size = 1)
            # Increment time
            t += delta_t
            # increment count
            cnt += 1

        # Store choice and reaction time
        rts[n] = t
        choices[n] = np.sign(y)

        if print_info == True:
            if n % 1000 == 0:
                print(n, ' datapoints sampled')

    return (rts, choices,  {'v': v,
                           'a': a,
                           'w': w,
                           's': s,
                           **boundary_params,
                           'delta_t': delta_t,
                           'max_t': max_t,
                           'n_samples': n_samples,
                           'simulator': 'ddm_flexbound',
                           'boundary_fun_type': boundary_fun.__name__,
                           'possible_choices': [-1, 1]})
# -----------------------------------------------------------------------------------------------

# FULL DDM --------------------------------------------------------------------------------------
def full_ddm(v = 0,
             a = 1,
             w = 0.5,
             dw = 0.05, # starting point perturbation from unif(-dw, dw)
             sdv = 0.1, # drift perturbation from normal(0, sd = sdv)
             s = 1,
             delta_t = 0.001,
             max_t = 20,
             n_samples = 20000,
             print_info = True,
             boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
             boundary_multiplicative = True,
             boundary_params = {'p1': 0, 'p2':0}
             ):

    # Initializations
    rts = np.zeros((n_samples,1)) #rt storage
    choices = np.zeros((n_samples,1)) # choice storage
    delta_t_sqrt = np.sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion

    # Boundary storage for the upper bound
    num_steps = int((max_t / delta_t) + 1)
    boundary = np.zeros(num_steps)

    # Precompute boundary evaluations
    if boundary_multiplicative:
        for i in range(num_steps):
            tmp = a * boundary_fun(t = i * delta_t, **boundary_params)
            if tmp > 0:
                boundary[i] = tmp
    else:
        for i in range(num_steps):
            tmp = a + boundary_fun(t = i * delta_t, **boundary_params)
            if tmp > 0:
                boundary[i] = tmp

    # Outer loop over n - number of samples
    for n in range(n_samples):
        y = (- 1) * boundary[0] + (w * 2 * boundary[0])
        # Apply perturbation to starting point
        y += np.random.uniform(low = (- 1) * dw, high = dw)
        # Define drift increment
        drift_increment = np.random.normal(loc = v, scale = sdv, size = 1) * delta_t

        t = 0.0
        cnt = 0

        # Inner loop (trajectory simulation)
        while y <= boundary[cnt] and y >= ((- 1) * boundary[cnt]) and t < max_t:
            # Increment y position (particle position)
            y += drift_increment + delta_t_sqrt * np.random.normal(loc = 0,
                                                                   scale = s,
                                                                   size = 1)
            # Increment time
            t += delta_t
            # increment count
            cnt += 1

        # Store choice and reaction time
        rts[n] = t
        choices[n] = np.sign(y)

        if print_info == True:
             if n % 1000 == 0:
                 print(n, ' datapoints sampled')

    return (rts, choices,  {'v': v,
                            'a': a,
                            'w': w,
                            'dw': dw,
                            'sdv': sdv,
                            's': s,
                            **boundary_params,
                            'delta_t': delta_t,
                            'max_t': max_t,
                            'n_samples': n_samples,
                            'simulator': 'full_ddm',
                            'boundary_fun_type': boundary_fun.__name__,
                            'possible_choices': [-1, 1]})
# -----------------------------------------------------------------------------------------------

# return ({'rts': rts,
#          'choices': choices,
#          'process_params': {'v': v, 'a': a, 'w': w, 's': s},
#          'boundary_params': {**boundary_params},
#          'simulator_params': {'delta_t': delta_t,
#                               'max_t': max_t,
#                               'n_samples': n_samples,
#                               'simulator': 'ddm_flexbound',
#                               'possible_choices': [-1, 1],
#                               'boundary_fun_type': boundary_fun.__name__},
#          'simulation_stats': {'choice_proportions': [len(choices[choices == -1]) / len(choices), len(choices[choices == 1]) / len(choices)]})

# Simulate (rt, choice) tuples from: RACE MODEL WITH N SAMPLES ----------------------------------
def race_model(v = [0, 0, 0], # np.array expected in fact, one column of floats
               a = 1, # Initial boundary height
               w = [0, 0, 0], # np.array expected in fact, one column of floats
               s = 1, # np.array expected in fact, one column of floats
               delta_t = 0.001,
               max_t = 20,
               n_samples = 2000,
               print_info = True,
               boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
               boundary_multiplicative = True,
               boundary_params = {'p1': 0, 'p2':0}
               ):

    # Initializations
    n_particles = len(v)
    rts = np.zeros((n_samples, 1))
    choices = np.zeros((n_samples, 1))
    delta_t_sqrt = np.sqrt(delta_t)
    particles = np.zeros((n_particles, 1))

    # We just care about an upper boundary here: (more complicated things possible)

    # Boundary Storage 
    num_steps = int(max_t / delta_t) + 1
    boundary = np.zeros(num_steps)

    # Precompute boundary evaluations
    if boundary_multiplicative:
        for i in range(num_steps):
            tmp = a * boundary_fun(t = i * delta_t, **boundary_params)
            if tmp > 0:
                boundary[i] = tmp
    else:
        for i in range(num_steps):
            tmp = a + boundary_fun(t = i * delta_t, **boundary_params)
            if tmp > 0:
                boundary[i] = tmp


    # Loop over samples
    for n in range(n_samples):
        # initialize y, t and time_counter
        particles = w * boundary[0]
        t = 0
        cnt = 0

        # Random walker
        while np.less_equal(particles, boundary[cnt]).all() and t <= max_t:
            particles += (v * delta_t) + (delta_t_sqrt * np.random.normal(loc = 0, scale = s, size = n_particles))
            t += delta_t
            cnt += 1

        rts[n] = t
        choices[n] = particles.argmax()

        if print_info == True:
            if n % 1000 == 0:
                print(n, ' datapoints sampled')

    # Create some dics
    v_dict = {}
    w_dict = {}
    for i in range(n_particles):
        v_dict['v_' + str(i)] = v[i, 0]
        w_dict['w_' + str(i)] = w[i, 0]


    return (rts, choices, {**v_dict,
                           'a': a,
                           **w_dict,
                           's': s,
                           **boundary_params,
                           'delta_t': delta_t,
                           'max_t': max_t,
                           'n_samples': n_samples,
                           'simulator': 'race_model',
                           'boundary_fun_type': boundary_fun.__name__,
                           'possible_choices': list(np.arange(0, n_particles, 1))})
# -------------------------------------------------------------------------------------------------

# Simulate (rt, choice) tuples from: Onstein-Uhlenbeck with flexible bounds -----------------------
def ornstein_uhlenbeck(v = 0, # drift parameter
                       a = 1, # initial boundary separation
                       w = 0.5, # starting point bias
                       g = 0.1, # decay parameter
                       s = 1, # standard deviation
                       delta_t = 0.001, # size of timestep
                       max_t = 20, # maximal time in trial
                       n_samples = 20000, # number of samples from process
                       print_info = True, # whether or not to print periodic update on number of samples generated
                       boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
                       boundary_multiplicative = True,
                       boundary_params = {'p1': 0, 'p2':0}):

    # Initializations
    rts = np.zeros((n_samples,1)) # rt storage
    choices = np.zeros((n_samples,1)) # choice storage
    delta_t_sqrt = np.sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion

    # Boundary storage for the upper bound
    num_steps = int((max_t / delta_t) + 1)
    boundary = np.zeros(num_steps)

    # Precompute boundary evaluations
    if boundary_multiplicative:
        for i in range(num_steps):
            tmp = a * boundary_fun(t = i * delta_t, **boundary_params)
            if tmp > 0:
                boundary[i] = tmp
    else:
        for i in range(num_steps):
            tmp = a + boundary_fun(t = i * delta_t, **boundary_params)
            if tmp > 0:
                boundary[i] = tmp

    # Outer loop over n - number of samples
    for n in range(0, n_samples, 1):
        # initialize y, t, and time_counter
        y = (- 1) * boundary[0] + (w * 2 * boundary[0])
        t = 0
        cnt = 0

        # Inner loop (trajection simulation)
        while y <= boundary[cnt] and y >= (- 1) * boundary[cnt] and t <= max_t:

            # Increment y position (particle position)
            y += ((v * delta_t) - (delta_t * g * y)) + delta_t_sqrt * np.random.normal(loc = 0,
                                                                                       scale = s,
                                                                                       size = 1)
            # Increment time
            t += delta_t
            # increment count
            cnt += 1

        # Store choice and reaction time
        rts[n] = t
        choices[n] = np.sign(y)

        if print_info == True:
            if n % 1000 == 0:
                print(n, ' datapoints sampled')

    return (rts, choices, {'v': v,
                           'a': a,
                           'w': w,
                           'g': g,
                           's': s,
                           **boundary_params,
                           'delta_t': delta_t,
                           'max_t': max_t,
                           'n_samples': n_samples,
                           'simulator': 'ornstein_uhlenbeck',
                           'boundary_fun_type': boundary_fun.__name__,
                           'possible_choices': [-1, 1]})
# --------------------------------------------------------------------------------------------------

# Simulate (rt, choice) tuples from: Leaky Competing Accumulator Model -----------------------------
def lca(v = np.zeros((2, 1)), # drift parameters (np.array expect: one column of floats)
        w = np.zeros((2, 1)), # initial bias parameters (np.array expect: one column of floats)
        a = 1, # criterion height
        g = 0, # decay parameter
        b = 1, # inhibition parameter
        s = 1, # variance (can be one value or np.array of size as v and w)
        delta_t = 0.001, # time-step size in simulator
        max_t = 20, # maximal time
        n_samples = 2000, # number of samples to produce
        print_info = True, # whether or not to periodically report the number of samples generated thus far
        boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
        boundary_multiplicative = True,
        boundary_params = {'p1': 0, 'p2':0}
        ):

    # Initializations
    n_particles = len(v)
    rts = np.zeros((n_samples, 1))
    choices = np.zeros((n_samples, 1))
    delta_t_sqrt = np.sqrt(delta_t)
    particles = np.zeros((n_particles, 1))

    # Boundary storage for the upper bound
    num_steps = int((max_t / delta_t) + 1)
    boundary = np.zeros(num_steps)

    # Precompute boundary evaluations
    if boundary_multiplicative:
        for i in range(num_steps):
            tmp = a * boundary_fun(t = i * delta_t, **boundary_params)
            if tmp > 0:
                boundary[i] = tmp
    else:
        for i in range(num_steps):
            tmp = a + boundary_fun(t = i * delta_t, **boundary_params)
            if tmp > 0:
                boundary[i] = tmp

    for n in range(0, n_samples, 1):

        # initialize y, t and time_counter
        particles_reduced_sum = particles
        particles = w * a
        t = 0
        cnt = 0

        while np.less_equal(particles, boundary[cnt]).all() and t <= max_t:
            particles_reduced_sum[:, ] = ((-1) * particles) + np.sum(particles)
            particles += ((v - (g * particles) - (b * particles_reduced_sum)) * delta_t) + \
                         (delta_t_sqrt * np.random.normal(loc = 0, scale = s, size = (n_particles, 1)))
            particles = np.maximum(particles, 0.0)
            t += delta_t
            cnt += 1

        rts[n] = t
        choices[n] = particles.argmax()

        if print_info == True:
            if n % 1000 == 0:
                print(n, ' datapoints sampled')

    # Create some dics
    v_dict = {}
    w_dict = {}
    for i in range(n_particles):
        v_dict['v_' + str(i)] = v[i, 0]
        w_dict['w_' + str(i)] = w[i, 0]


    return (rts, choices, {**v_dict,
                           'a': a,
                           **w_dict,
                           'g': g,
                           'b': b,
                           's': s,
                           **boundary_params,
                           'delta_t': delta_t,
                           'max_t': max_t,
                           'n_samples': n_samples,
                           'simulator': 'ornstein_uhlenbeck',
                           'boundary_fun_type': boundary_fun.__name__,
                           'possible_choices': list(np.arange(0, n_particles, 1))})
# --------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    start_time = time.time()
    #ddm_simulate(n_samples = 1000)
    #print("--- %s seconds ---" % (time.time() - start_time))
    #start_time = time.time()
    #ddm_simulate_fast(n_samples = 1000)
    #print("--- %s seconds ---" % (time.time() - start_time))
