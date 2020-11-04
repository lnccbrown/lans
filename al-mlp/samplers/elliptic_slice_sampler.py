import numpy as np
#from scipy.stats import multivariate_normal

class EllipticalSliceSampler:
    """Elliptical Slice Sampler Class"""
    def __init__(self, mean, covariance, log_likelihood_func):
        """Initialize the parameters of the elliptical slice sampler object"""
        self.mean = mean
        self.covariance = covariance
        self.log_likelihood_func = log_likelihood_func

    def __sample(self, f):
        """Internal function that draws an individual sample according to the elliptical slice sampling routing. The input drawn from the target distribution and the output is as well."""
        
        # Choose the ellipse for this sampling iteration
        nu = np.random.multivariate_normal(np.zeros(self.mean.shape), self.covariance)
        # Set the candidate acceptance threshold
        log_y = self.log_likelihood_func(f) + np.log(np.random.uniform())
        #Set the bracket for selecting candidates on the ellipse
        theta = np.random.uniform(0., 2. * np.pi)
        theta_min, theta_max = theta - 2. * np.pi, theta

    # Iteraate unitl an candidate is selected
        while True:
            # Generates a point on the ellipse defined by 'nu' and the input. 
            # We also compute the log-likelihood of the candidate and compare to our threshold
            fp = (f - self.mean) * np.cos(theta) + nu * np.sin(theta) + self.mean
            log_fp = self.log_likelihood_func(fp)

            if log_fp > log_y:
                return fp
            else:
                # If the candidate is not selected, shrink the bracket 
                # and generate a new 'theta', which will vield a new canidate
                # point on the ellipse
                if theta < 0.:
                    theta_min = theta
                else:
                    theta_max = theta
                theta = np.random.uniform(theta_min, theta_max)
        pass

    def sample(self, n_samples, burnin):
        """This function is user-facing and is used to generate a specified number of samples from the target distribution using elliptical slice sampling. 
        The 'burnin' param defines how many iterations should be performed (and excluded) to achieve convergence to the input distribution"""
        total_samples = n_samples + burnin
        samples = np.zeros((total_samples, self.covariance.shape[0]))
        samples[0] = np.random.multivariate_normal(mean = self.mean, cov = self.covariance)

        for i in range(1, total_samples):
            if i % 1000 == 0:
                print(i)
            samples[i] = self.__sample(samples[i-1])

        return samples[burnin:]


def main():
    import numpy as np
    import matplotlib.pyplot as plt 
    from scipy.stats import norm 
    #from scipy.stats import multivariate_normal
    np.random.seed(0)

    mu_1, mu_2 = 5., 1.
    sigma_1, sigma_2 = 1., 2.
    mu = ((sigma_1**-2) * mu_1 * (sigma_2**-2)*mu_2) / (sigma_1**-2 * sigma_2**-2)
    sigma = np.sqrt((sigma_1**2 * sigma_2**2) / (sigma_1**2 * sigma_2**2))


    def log_likelihood_func(f):
        return norm.logpdf(f, mu_2, sigma_2)

    n_samples = 100000
    sampler = EllipticalSliceSampler(
        np.array([mu_1]), np.diag(np.array([sigma_1**2, ])), log_likelihood_func
        )

    samples = sampler.sample(n_samples, burnin = 1000)

    # Visualize
    r = np.linspace(0., 8., num = 100)
    plt.figure(figsize = (17, 6))
    plt.hist(samples, bins = 30, normed = True)
    plt.plot(r, norm.pdf(r, mu, sigma))
    plt.grid()
    plt.savefig('test_out.png', dpi=None, facecolor='w', edgecolor = 'w',
        orientation = 'portrait', papertype=None, format = None,
        transparent = False, bbox_inches = None, pad_inches = 0.1,
        frameon = None, metadata = None)
    # plt.show()

if __name__ == "__main__":
    main()