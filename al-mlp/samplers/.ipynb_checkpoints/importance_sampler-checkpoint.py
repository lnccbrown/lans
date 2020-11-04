import numpy as np
import multiprocessing as mp
import scipy.stats as scps
from scipy.special import logsumexp

class ImportanceSampler:
    def __init__(self, bounds, target):
        self.dims = bounds.shape[0]
        self.bounds = bounds
        self.target = target
        self.proposal = scps.multivariate_normal
        self.eff = 0

    def calculate_weights(self):
        out = self.p.starmap(self.target, zip(self.particles, self.data_tile)) # NOTE: Mem-leak ?
        return np.fromiter(out, np.float)

    def sample(self, data, num_particles = 10000, max_iter = 20):
        self.particles = np.zeros((num_particles, self.dims))

        self.data = data
        self.data_tile = np.tile(self.data, (self.particles.shape[0], 1, 1)) # NOTE: Mem-leak ?
        self.p = mp.Pool(mp.cpu_count())

        # Start off with uniform proposal
        samples = []
        while len(samples) < 5:
            self.particles = np.random.uniform(self.bounds[0], self.bounds[1],
                    size=(num_particles, self.bounds[0].shape[0]))
            self.weights = self.calculate_weights()
            self.weights = np.exp(self.weights - logsumexp(self.weights))
            mask = np.isclose(self.weights, 0)
            samples = self.particles[~mask]

        # Proposal 0
        self.eff = self.weights.var() / np.power(self.weights.mean(), 2) + 1
        ratio = self.eff

        iteration = 1
        while ratio > 1e-4 and iteration < max_iter:
            print("iteration {}".format(iteration))
            mask = np.isclose(self.weights, 0) 
            samples = self.particles[~mask]
            print(samples.shape[0])
            sample_mu = samples.mean(axis=0)
            sample_cov = np.cov(samples.T)

            self.proposal = scps.multivariate_normal(mean = sample_mu, cov = sample_cov)
            try:
                self.particles = self.proposal.rvs(num_particles)
            except:
                print(samples)
                return
            self.particles = np.clip(self.particles, self.bounds[0], self.bounds[1])
            self.weights = self.calculate_weights()
            self.weights -= self.proposal.logpdf(self.particles) # Step not necessary above since we use uniform 
            self.weights = np.exp(self.weights - logsumexp(self.weights))

            eff = self.weights.var() / np.power(self.weights.mean(), 2) + 1
            ratio = eff / self.eff
            self.eff = eff
            iteration += 1
