import numpy as np
import scipy
import scipy.optimize as scp_opt
import scipy.stats as scp_stat
import samplers.diagnostics as mcmcdiag
import time

class DifferentialEvolutionSequential():
    
    def __init__(self, 
                 bounds, 
                 target, 
                 NP_multiplier = 5, 
                 gamma = 'auto', # 'auto' or float 
                 mode_switch_p = 0.1,
                 proposal_std = 0.01,
                 crp = 0.3,
                 n_burn_in = 2000):
        
        """
        Params
        -----
        dims: int
            dimension of the parameter space
        bounds: list of np.ndarrays
            The first element is a numpy array of lower bounds, the
            second element is the numpy array of upper bounds for the params
        NP: int
            number of particles to use
        target: function(ndarray params, ndarray data) -> float
            function that takes in two arguments: params, and data, and returns
            the log likelihood of the data conditioned on the parameters.
        gamma: float
            gamma parameter to mediate the magnitude of the update
        """
        
        self.optimizer = scp_opt.differential_evolution # TD: USE FOR INITIALIZATION
        self.dims = bounds.shape[0] #np.array([i for i in range(len(bounds))])
        self.bounds = bounds
        self.NP = int(np.floor(NP_multiplier * self.dims))
        self.NP_multiplier = NP_multiplier
        self.target = target
        
        # for optimization ( --> minimization) need to return - target (loglikelihood)
        def opt_target(params, data):
            return - self.target(params, data)
        
        self.opt_target = opt_target
        
        if gamma == 'auto':
            self.gamma = 2.38 / np.sqrt(2 * self.dims)
        else: 
            self.gamma = gamma

        #self.gamma = gamma 
        self.mode_switch_p = mode_switch_p
        self.proposal_std = proposal_std
        self.crp = crp
        self.n_burn_in = n_burn_in
        self.accept_cnt = 0
        self.total_cnt = 0
        self.gelman_rubin = 10
        self.gelman_rubin_r_hat = []
        np.random.seed()
        self.random_seed = np.random.get_state()
        
        self.optim_time = -1
        self.sample_time = -1
        
        # variables to carry around
        #self.tmp_prop = np.zeros(self.sample)

        #print(self.random_seed)
        #print(np.random.normal(size = 10))
    
    def attach_sample(self, samples):
        assert samples.shape[0] == self.NP, 'Population size of previous sample does not match NP parameter value'
        self.samples = samples

    def anneal_logistic(self, x = 1, k = 1/100, L = 10):
        return 1 + (2 * L - (2 * L / (1 + np.exp(- k * (x)))))
    
    def propose(self, idx, anneal_k, anneal_L, crossover = True):
        """
        Takes in a chain, and updates the chain parameters and log-likelihood
        """
        
        proposals = self.samples[:, idx - 1, :].copy()
        proposals_lps = self.lps[:, idx - 1].copy()
        
        self.samples[:, idx, :] = self.samples[:, idx - 1, :].copy()
        self.lps[:, idx] = self.lps[:, idx - 1].copy()
        
        pop_seq = np.arange(self.NP)
        np.random.shuffle(pop_seq)
        
        # LOOP OVER POPULATIONS (CHAINS)
        for pop in pop_seq:
            # TD: MODE SWITCH UPON GAMMA = 1 CAN BE BE IMPLEMENTED BETTER (Actually attempt exchange of modes?)
            # Get candidates that affect current vectors update:
            R1 = pop
            while R1 == pop:
                R1 = np.random.choice(pop_seq)
            
            R2 = pop
            while R2 == pop or R2 == R1:
                R2 = np.random.choice(pop_seq)
                
            # Assign gamma == 1 according to parameter mode_switch_p
            gamma_cur = np.random.choice([self.gamma, 1],  p = [1 - self.mode_switch_p, self.mode_switch_p]) 
            
            proposals[pop, :] += gamma_cur * (proposals[R1, :] - proposals[R2, :]) +  \
                                                        np.random.normal(loc = 0, scale = self.proposal_std, size = self.dims)
                                                        #self.proposal_std * np.random.standard_t(df = 2, size = self.dims)
                                                        
            # Crossover:
            if gamma_cur == 1: # If we allow mode switch --> we do not subsample dimensions
                pass  
            else: # If we are not mode switching we use dimension subsampling
                if crossover == True:
                    n_keep = np.random.binomial(self.dims - 1, p = 1 - self.crp)
                    id_keep = np.random.choice(self.dims, n_keep, replace = False)
                    proposals[pop, id_keep] = self.samples[pop, idx - 1, id_keep]

            # Clip proposal and reflect (should help with not getting stuck on the bounds all the time) at bounds: 
            # If in a certain dimension all chains reach the border we have no way of returning from the border
            # This is in part due to the minimal actual 'noise' perturbation that the sampler uses...
            # Note how reflecting works: clipped + (clipped - prev)
            # IF clipped > prev (low end) we add something 
            # IF clipped < prev (upper end) we deduct something
            # IF clipped = prev no effect
            
            self.tmp_prop[:] = proposals[pop, :]
            proposals[pop, :] = np.clip(proposals[pop, :], self.bounds[:, 0], self.bounds[:, 1])
            proposals[pop, :] = proposals[pop, :] + (proposals[pop, :] - self.tmp_prop[:])

            # If we didn't clip anything away we run rejection step ( DEPRECATED )
            # if np.array_equal(self.tmp_prop, proposals[pop, :]):
                
            proposals_lps[pop] = self.target(proposals[pop, :], self.data)
            acceptance_prob = proposals_lps[pop] - self.lps[pop, idx - 1]
            self.total_cnt += 1

            if (np.log(np.random.uniform()) / self.anneal_logistic(x = idx, k = anneal_k, L = anneal_L)) < acceptance_prob:
                self.samples[pop, idx, :] = proposals[pop, :]
                self.lps[pop, idx] = proposals_lps[pop]
                self.accept_cnt += 1

            # If we needed to clip we reject immediately ( DEPRECATED )
#             else:
#                 self.total_cnt += 1
#                 self.samples[pop, idx, :] = self.samples[pop, idx - 1, :]
   
    def sample(self, 
               data, 
               max_samples = 5000, 
               add = False, 
               crossover = True, 
               anneal_k = 1 / 80, 
               anneal_L = 10,
               init = 'random', # TD: Initialization only random at this point
               active_dims = None, # ADD ACTIVE DIMS PROPERLY HERE
               frozen_dim_vals = None,
               n_burn_in = 2000,
               min_samples = 3000,
               gelman_rubin_force_stop = False,
               mle_popsize = 100,
               mle_polish = False,
               mle_disp = True,
               mle_maxiter = 30): 
        
        if add == False:
            self.n_burn_in = n_burn_in
            self.min_samples = min_samples
            self.max_samples = max_samples
            
            self.data = data
            self.lps = np.zeros((self.NP, max_samples))
            self.samples = np.zeros((self.NP, max_samples, self.dims))
            
            # Accept and total counts reset
            self.accept_cnt = 0
            self.total_cnt = 0
            
            # variables to carry around
            self.tmp_prop = self.samples[0, 0, :]
            
            # Initialize parameters
            temp = np.zeros((self.NP, self.dims))
            
            if init == 'random':
                for pop in range(self.NP):
                    
                    for dim in range(self.dims):
                        # Initialize at random but give leave some buffer on each side of parameter boundaries
                        dim_range = self.bounds[dim][1] - self.bounds[dim][0]
                        temp[pop, dim] = np.random.uniform(low = self.bounds[dim, 0] + (0.2 * dim_range), 
                                                           high = self.bounds[dim, 1] - (0.2 * dim_range))

                    self.samples[pop, 0, :] = temp[pop, :]
                    self.lps[pop, 0] = self.target(temp[pop, :], self.data)
            
            elif init == 'mle':
                optim_time_start = time.time()
                # Make bounds for mle optimizer
                bounds_tmp = [tuple(b) for b in self.bounds]
                
                # Run mle 
                pop = 0
                while pop < (int(self.NP)):
                    # Run one mle for each parameter in the model
                    # Then create starting points by slight perturbation around this mle estimate
                    # Running one mle for each parameter is somewhat arbitrary --> point is to create some initial spread and
                    # detect potentially local modes
                    if pop % self.NP_multiplier == 0:
                        out = self.optimizer(self.opt_target, 
                                             bounds = bounds_tmp, 
                                             args = (self.data,), 
                                             popsize = mle_popsize,
                                             polish = mle_polish,
                                             disp = mle_disp,
                                             maxiter = mle_maxiter,
                                             workers = 1)
                        
                        print('MLE vector: ', out.x)
                        temp[pop, :] = np.clip(out.x, 
                                              self.bounds[:, 0] + 0.01, 
                                              self.bounds[:, 1] - 0.01)
                    else:
                        temp[pop, :] = np.clip(out.x + np.random.normal(loc = 0, 
                                                                        scale = 0.05, 
                                                                        size = self.bounds.shape[0]),
                                               self.bounds[:, 0] + 0.01, 
                                               self.bounds[:, 1] - 0.01)
                        
                    self.samples[pop, 0, :] = temp[pop, :]
                    self.lps[pop, 0] = self.target(temp[pop, :], self.data)
            
                    
                    pop += 1
                
                optim_time_end = time.time()
                self.optim_time = optim_time_end - optim_time_start
                
            elif init == 'groundtruth':
                for pop in range(self.NP):
                    temp[pop, :] = groundtruth + np.random.normal(loc = 0, scale = 0.05, size = len(init))
                    temp[pop, :] = np.clip(temp[pop, :], self.bounds[:, 0] + 0.01, self.bounds[:, 1] - 0.01)
                    
                    self.samples[pop, 0, :] = temp[pop, :]
                    self.lps[pop, 0] = self.target(temp[pop, :], self.data)
                    
            id_start = 1
            
        if add == True:
            # Make extended data structure
            shape_prev = self.samples.shape
            samples_tmp = np.zeros((shape_prev[0], shape_prev[1] + max_samples, shape_prev[2]))
            samples_tmp[:shape_prev[0], :shape_prev[1], :shape_prev[2]] = self.samples
            self.samples = samples_tmp

            lps_tmp = np.zeros((self.NP, shape_prev[1] + max_samples))
            lps_tmp[:, :shape_prev[1]] = self.lps
            self.lps = lps_tmp

            id_start = shape_prev[1]
            
            # Accept and total counts reset
            self.accept_cnt = 0
            self.total_cnt = 0
            
            
        print("Beginning sampling: ")
        n_samples_final = self.samples.shape[1] # If we allow adding samples to a set of previous samples we need to access samples.shape instead of simply picking max_samples --> if add == False then max_samples = n_samples_final
        adaptation_start = int(self.n_burn_in / 2)
        i = id_start
        continue_ = 1
        sample_time_start = time.time()
        while i < n_samples_final:
            
            # Print iteration number periodically
            if (i % 200 == 0):
                print("Iteration {}".format(i))
            
            # Apply adaptations during burn in
            if ((i > adaptation_start) and (i % 200 == 0) and (i < self.n_burn_in)):
                
                acc_rat_tmp = self.accept_cnt / self.total_cnt
                print('Acceptance ratio: ', acc_rat_tmp)
                if (acc_rat_tmp) < 0.05:
                    self.proposal_std = self.proposal_std / 2
                    print('New proposal std: ', self.proposal_std)

                if (acc_rat_tmp) > 0.5:
                    self.proposal_std = self.proposal_std * 1.5
                    print('New proposal std: ', self.proposal_std)

                self.accept_cnt = 0
                self.total_cnt = 0

                # Pull outlier chains in: TD make adaptive periods proper !

                # Get log posterior means
                lp_means = np.mean(self.lps[:, int(i / 2):(i - 1)], axis = 1)
                print('LP means: ' , lp_means)
                print('LP means dim: ', lp_means.shape)
                
                # Get corresponding parameter values 
                print(self.samples[:, (i-1), :])
                
                # Get first quantile
                q1 = np.quantile(lp_means, .25)
                print('quantile: ', q1)
                
                # Get interquartile range
                iqr = scp_stat.iqr(lp_means)
                print('iqr: ', iqr)
                # Get the ids of outliers and proper chains
                okids = (lp_means > (q1 - 2 * iqr)).nonzero()[0]
                outlierids = (lp_means < (q1 - 2 * iqr)).nonzero()[0]
                print(okids)
                print(outlierids)
                # Exchange last sample of outlier chains with proper chains
                for outlierid in outlierids:
                    okid = np.random.choice(okids)
                    self.samples[outlierid, i - 1, :] = self.samples[okid, i - 1, :]
                    self.lps[outlierid, i - 1] = self.lps[okid, i - 1]

                print('Number of Outliers: ', len(outlierids))

                # If outliers were pulled in we extend burn in period
                if len(outlierids) >= 1:
                    self.n_burn_in += 200
                    # We also extend min sample accordingly
                    self.min_samples += 200
                    print('Burn in extended to: ', self.n_burn_in)
                
            # Periodically compute gelman rubin and potentially use it as stopping rule if desired 
            if ((i > self.n_burn_in) and (i % 1000 == 0)):
                # Compute gelman rubin
                continue_, r_hat = mcmcdiag.get_gelman_rubin_mv(chains = self.samples,
                                                                burn_in = self.n_burn_in,
                                                                thresh = 1.01)
                self.gelman_rubin_r_hat.append(r_hat)

                print('Gelman Rubin: ', r_hat)
                print('Continue: ', continue_)

                if not continue_:
                    if gelman_rubin_force_stop and (i > self.min_samples):
                        print('Sampler stopped...')
                        break
                    
            self.propose(i, anneal_k, anneal_L, crossover)
            i += 1
        
        if continue_:
            # Here I need to adjust samples so that the final datastructure doesn't have 0 elements
            print( 'Stopped due to max samplers reach, NOT converged' )
        sample_time_end = time.time()
        self.sample_time = sample_time_end - sample_time_start
        
        return (self.samples[:, self.n_burn_in:i, :], 
                self.lps[:, self.n_burn_in:i], 
                self.gelman_rubin_r_hat,
                self.sample_time,
                self.optim_time) #, self.random_seed)