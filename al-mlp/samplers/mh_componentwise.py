import numpy as np
import multiprocessing as mp
import ctypes

class MetropolisHastingsComponentwise:

    def __init__(self, num_chains, bounds, target, proposal_var = .01):
        """
        Params
        -----
        dims: int
            dimension of the parameter space
        bounds: list of np.ndarrays
            The first element is a numpy array of lower bounds, the
            second element is the numpy array of upper bounds for the params
        target: function(ndarray params, ndarray data) -> float
            function that takes in two arguments: params, and data, and returns
            the log likelihood of the data conditioned on the parameters.
        """
        
        self.dims = self.bounds.shape[0]
        self.target = target
        self.num_chains = num_chains
        self.bounds = bounds
        
        # ----
        self.proposal_var = proposal_var
    
    def propose(self, chain):
        proposal = chain[0].copy()
        dim_seq = np.arange(self.dims)
        np.random.shuffle(dim_seq)
        tmp_lp = chain[1]
        
        for dim in dim_seq:
            
            proposal[dim] += np.random.normal(loc = 0, scale = self.proposal_var)
            proposal[dim] = np.clip(proposal[dim], self.bounds[dim][0], self.bounds[dim][1])
            proposal_lp = self.target(proposal, data = self.data)
            acceptance_prob = proposal_lp - tmp_lp
            
            # Rejection step
            if np.log(np.random.uniform()) < acceptance_prob:
                # If accept keep proposal as is and update target log probability to proposal one
                tmp_lp = proposal_lp
            else:
                # If reject revert back to former state
                proposal[dim] = chain[0][dim]
                
        return (proposal, tmp_lp)
    
    def sample(self, data, num_samples = 800, add = False, n_cores = 4, init = 'random'):
        
        if add == False:
            self.data = data
            self.chains = []

            # Define array of chain value to be shared in memory
            if init == 'random':
                
                for i in range(self.num_chains):
                    tmp = []
                    for dim in range(self.dims):
                        tmp.append(np.random.uniform(self.bounds[dim][0], self.bounds[dim][1]))
                    tmp = np.array(tmp)           
                    self.chains.append((tmp, self.target(tmp, self.data)))
                    
            else:
                for i in range(self.num_chains):
                    for dim in range(self.dims):
                        self.chains.append((init[dim], self.target(init[dim], self.data)))

            self.samples = np.zeros((num_samples, self.num_chains, self.dims)) # more natural to have (chain, iteration, parameters)
            self.lp = np.zeros((num_samples, self.num_chains))
            id_start = 0
            
        if add == True:
            # Make extended data structure
            self.data = data
            shape_prev = self.samples.shape
            
            samples_tmp = np.zeros((shape_prev[0] + num_samples, shape_prev[1], shape_prev[2]))
            samples_tmp[:shape_prev[0], :shape_prev[1], :shape_prev[2]] = self.samples
            self.samples = samples_tmp
            
            lp_tmp = np.zeros((shape_prev[0] + num_samples, self.num_chains))
            lp_tmp[:shape_prev[0], :] = self.lp
            self.lp = lp_tmp
            
            self.chains = []
            for i in range(self.num_chains):
                tmp = self.samples[shape_prev[0] - 1, i, :]
                self.chains.append((tmp, self.target(tmp, self.data)))
                
            id_start = shape_prev[0]
            
        print("Beginning sampling")
        n_samples_final = self.samples.shape[0]
        i = id_start
        p = mp.Pool(n_cores)
        
        while i < n_samples_final:
            if i % 100 == 0:
                print("Iteration: {}".format(i))   
            
            out = p.map(self.propose, self.chains)
            out_unzip = [list(k) for k in zip(*out)]
            self.samples[i, :, :] = np.array(out_unzip[0])
            self.lp[i, :] = out_unzip[1]
            
            self.chains = out
            i += 1

        p.close()
        p.terminate()
              
# ----------- UNUSED --------------------
# # Update should go over all dimensions !
# def propose_old(self, chain):
#     proposal = chain[0].copy()
#     dim = np.random.choice(self.dims)
#     proposal[dim] += np.random.normal(loc = 0, scale = self.proposal_var)
#     proposal = np.clip(proposal, self.bounds[dim][0], self.bounds[dim][1])
#     proposal_lp = self.target(proposal, data = self.data)
#     acceptance_prob = proposal_lp - chain[1]
#     if np.log(np.random.uniform()) < acceptance_prob:
#         return (proposal, proposal_lp)
#     else:
#         return (chain[0], chain[1])
