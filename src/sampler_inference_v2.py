import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from reverse_model import cnn_reverse_model
from train_detector import cnn_model_struct
import config
import tensorflow as tf
import math
from math import *
from scipy import linalg
from scipy.optimize import differential_evolution
import numdifftools as nd
import tqdm, gzip, cProfile, time, argparse, pickle, os
from multiprocessing import Pool
from functools import partial
# just to prevent tensorflow from printing logs
os.environ['TF_CPP_MIN_LOG_LEVEL']="2"
tf.logging.set_verbosity(tf.logging.ERROR)


class Infer:
    def __init__(self, config):
	self.cfg = config
	self.target = []
	self.inp = tf.placeholder(tf.float32, self.cfg.test_param_dims)
	self.initialized = False
	with tf.device('/gpu:0'):
	    with tf.variable_scope("model", reuse=tf.AUTO_REUSE) as scope:
		self.model = cnn_model_struct()
		self.model.build(self.inp, self.cfg.test_param_dims[1:], self.cfg.output_hist_dims[1:], train_mode=False, verbose=False)
	    self.gpuconfig = tf.ConfigProto()
	    self.gpuconfig.gpu_options.allow_growth = True
	    self.gpuconfig.allow_soft_placement = True
	    self.saver = tf.train.Saver()

    def __getitem__(self, item):
	return getattr(self, item)

    def __contains__(self, item):
	return hasattr(self, item)

    def klDivergence(self, x, y, eps1=1e-7, eps2=1e-30):
	return np.sum(x * np.log(eps2 + x/(y+eps1)))

    def likelihood(self, x, y, eps=1e-7):
	return np.sum(-np.log(x+eps)*y)

    def objectivefn(self, params):
	if self.initialized == False:
	    self.sess = tf.Session(config=self.gpuconfig)
	    ckpts = tf.train.latest_checkpoint(self.cfg.model_output)
	    self.saver.restore(self.sess, ckpts)
	    self.initialized = True
	pred_hist = self.sess.run(self.model.output, feed_dict={self.inp:params.reshape(self.cfg.test_param_dims)})
	return self.likelihood(pred_hist, self.target)


class ImportanceSampler:
    def __getitem__(self, item):
        return getattr(self, item)

    def __contains__(self, item):
        return hasattr(self, item)

    """
    We need to initialize two models here:
    Model1: dataset -> params (this will give us an initial proposal)
    Model2: params -> dataset (allows us to evaluate likelihood of subsequent IS iterations)
    """
    def __init__(self, config, max_iters=100, tol=1e-7, nsamples=1e5, tdist=False):     

        if tdist:
            self.generateFromProposal = self.generateFromProposalTDist
            self.evalProposalByComponent = self.evalProposalByComponentTDist
        else:
            self.generateFromProposal = self.generateFromProposalNormal
            self.evalProposalByComponent = self.evalProposalByComponentNormal

        self.max_iters = max_iters
        self.tol = tol
        self.N = nsamples

        self.cfg = config
        self.target = []

        self.inf_batch_size = 50000

        # placeholder for forward model
        #self.forward_model_inpdims = [self.inf_batch_size] + self.cfg.param_dims[1:]
        self.forward_model_inpdims = [None] + self.cfg.param_dims[1:]

        self.forward_input = tf.placeholder(tf.float32, self.forward_model_inpdims)
        self.forward_initialized = False

        with tf.device('/gpu:0'):
            with tf.variable_scope("model", reuse=tf.AUTO_REUSE) as scope:
                # build the forward model
                self.forward_model = cnn_model_struct()
                self.forward_model.build(self.forward_input, self.cfg.param_dims[1:], self.cfg.output_hist_dims[1:], train_mode=False, verbose=False)

            self.gpuconfig = tf.ConfigProto()
            self.gpuconfig.gpu_options.allow_growth = True
            self.gpuconfig.allow_soft_placement = True
            self.saver = tf.train.Saver()
        self.forward_sess = tf.Session(config=self.gpuconfig)
        ckpts = tf.train.latest_checkpoint(self.cfg.model_output)
        self.saver.restore(self.forward_sess, ckpts)

        '''
        # placeholder for inverse model
        self.inv_model_inpdims = [1] + self.cfg.output_hist_dims[1:]
        self.inv_input = tf.placeholder(tf.float32, self.inv_model_inpdims)
        
        with tf.device('/gpu:1'):
            with tf.variable_scope("reversemodel", reuse=tf.AUTO_REUSE) as scope:
                # build the inverse model
                self.inv_model = cnn_reverse_model()
                self.inv_model.build(self.inv_input, self.cfg.output_hist_dims[1:], self.cfg.param_dims[1:], train_mode=False, verbose=False)

            self.gpuconfig1 = tf.ConfigProto()
            self.gpuconfig1.gpu_options.allow_growth = True
            self.gpuconfig1.allow_soft_placement = True
            self.saver1 = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='reversemodel'))

        self.inv_sess = tf.Session(config=self.gpuconfig1)
        ckpts = tf.train.latest_checkpoint(os.path.join(self.cfg.base_dir, 'models', 'rev_'+self.cfg.model_name+'_training_data_binned_{}_nbins_{}_n_{}'.format(int(self.cfg.isBinned),self.cfg.nBins,self.cfg.nDatapoints))
)
        self.saver1.restore(self.inv_sess, ckpts)
        '''
    def objectivefn(self, params):
        #import ipdb; ipdb.set_trace()
        pred_data = self.forward_sess.run(self.forward_model.output, feed_dict={self.forward_input:params.reshape(self.cfg.test_param_dims)})
        return self.MLELikelihood(pred_data[0], self.target) 

    def initializeMixtures(self, mu_initial, std_initial, n_components=3, mu_perturbation=(-2., 2.), spread=10., tdist=False):
        padding = 0.2
        n_dims = mu_initial.shape[0]
        # set the inital component weights
        alpha_p = np.random.randint(low=1, high=n_components-1, size=n_components).astype(np.float32)
        alpha_p[0] = n_components
        alpha_p = alpha_p / alpha_p.sum()

        # initialize the mu and sigma
        mu_p = np.zeros((n_components, n_dims))
        std_p = np.zeros((n_components, n_dims, n_dims))

        # set the first component right on the point estimate
        for k in range(n_dims):
            if mu_initial[k] > self.cfg.bounds[k][1]:
                mu_p[0][k] = self.cfg.bounds[k][1] - padding
            elif mu_initial[k] < self.cfg.bounds[k][0]:
                mu_p[0][k] = self.cfg.bounds[k][0] + padding
            else:
                mu_p[0][k] = mu_initial[k]
        #mu_p[0] = mu_initial
        std_p[0] = np.diag(std_initial)

        # initialize the other components by perturbing them a little bit around the point estimate
        # Here we need to make sure that the centers aren't outside the appropriate bounds
        for c in range(1, n_components):
            #mu_p[c] = mu_p[0] + np.random.uniform(low=mu_perturbation[0], high=mu_perturbation[1], size=n_dims)*std_initial
            sample_goodness = False
            sample_again = False
            attempts = 0
            while not(sample_goodness) and (attempts < 100):
                new_mu = mu_p[0] + np.random.uniform(low=mu_perturbation[0], high=mu_perturbation[1], size=n_dims)*std_initial
                for k in range(n_dims):
                    if ((new_mu[k] < self.cfg.bounds[k][0]) | (new_mu[k] > self.cfg.bounds[k][1])):
                        sample_again = True
                        break
                if sample_again:
                    sample_goodness = False
                    attempts = attempts + 1
                else :
                    sample_goodness = True

            mu_p[c] = new_mu
            std_p[c] = spread * std_p[0]
        
        # set the members
        self.n_components = n_components
        self.alpha_p = alpha_p
        self.mu_p = mu_p
        self.std_p = std_p

        # initialize the degrees of freedom of the T proposals
        if tdist:
            self.degrees_p = np.zeros_like(alpha_p) + 1 + np.random.randint(10)

    def reval_components(self, tol=1e-6):
        cidx = np.argsort(self.alpha_p)
        for c in range(self.alpha_p.shape[0]):
            if self.alpha_p[c] <= tol:
                self.alpha_p[c] = self.alpha_p[cidx[-1]]
                self.mu_p[c] = self.mu_p[cidx[-1]]
                self.std_p[c] = self.std_p[cidx[-1]]
        self.alpha_p = self.alpha_p/np.sum(self.alpha_p)

    def reparamSigmoid(self, a, b):
        return lambda v : a + (b-a)/(1 + np.exp(-v))

    def reparamInvSigmoid(self, a, b):
        return lambda v : np.log((v - a)/(b - v))

    def initializeMixturesMLE(self, dataset, n_components=3, tdist=False):
        self.target = dataset.reshape((-1,))
        hessianfn = nd.Hessian(self.objectivefn)

        n_dims = len(self.cfg.bounds)
        # set the inital component weights
        alpha_p = np.random.randint(low=1, high=n_components-1, size=n_components).astype(np.float32)
        alpha_p = alpha_p / alpha_p.sum()

        # initialize the mu and sigma
        mu_p = np.zeros((n_components, n_dims))
        std_p = np.zeros((n_components, n_dims, n_dims))

        print('Initializing proposal distributions...')
        for c in tqdm.tqdm(range(n_components)):
            output = differential_evolution(self.objectivefn, self.cfg.bounds)
            #mu_p[c,:] = [self.reparamInvSigmoid(self.cfg.bounds[idx][0], self.cfg.bounds[idx][1])(y) for idx,y in enumerate(output.x)]
            mu_p[c,:] = output.x
            std_p[c, :, :] = np.eye(std_p[c,:,:].shape[0]) * 0.5

        # set the members
        self.n_components = n_components
        self.alpha_p = alpha_p
        self.mu_p = mu_p
        self.std_p = std_p

        # initialize the degrees of freedom of the T proposals
        if tdist:
            self.degrees_p = np.zeros_like(alpha_p) + 1 + np.random.randint(10)

    def initializeMixturesMLEV2(self, params, n_components=3, tdist=False):
        n_dims = len(self.cfg.bounds)
        # set the inital component weights
        alpha_p = np.random.randint(low=1, high=n_components-1, size=n_components).astype(np.float32)
        alpha_p = alpha_p / alpha_p.sum()

        # initialize the mu and sigma
        mu_p = np.zeros((n_components, n_dims))
        std_p = np.zeros((n_components, n_dims, n_dims))

        print('Initializing proposal distributions...')
        for c in range(n_components):
            mu_p[c,:] = params[c]
            std_p[c, :, :] = np.eye(std_p[c,:,:].shape[0]) * 0.5

        # set the members
        self.n_components = n_components
        self.alpha_p = alpha_p
        self.mu_p = mu_p
        self.std_p = std_p

        # initialize the degrees of freedom of the T proposals
        if tdist:
            self.degrees_p = np.zeros_like(alpha_p) + 1 + np.random.randint(10)


    def klDivergence(self, x, y, eps1=1e-7, eps2=1e-30):
        return np.sum(x * np.log(eps2 + x/(y+eps1)))

    def likelihood(self, x, y, gamma, eps=1e-7):
        #import ipdb; ipdb.set_trace()
        #return np.sum(np.log(x+eps)*y/gamma, axis=1)
        return np.dot(np.log(x+eps),y/gamma)

    def MLELikelihood(self, x, y, eps=1e-7):
        return np.sum(-np.log(x+eps)*y)

    '''
    feed forward through the inverse model to get a point estimate of the parameters that could've generated a given dataset
    '''
    def getPointEstimate(self, dataset):
        params = self.inv_sess.run(self.inv_model.output, feed_dict={self.inv_input: dataset.reshape(self.inv_model_inpdims)})
        nparams = np.prod(self.cfg.param_dims[1:])
        means, stds = params[0][:nparams], params[0][nparams:]
        return means, stds

    def getLikelihoodFromProposals(self, params, target, gamma):
        nbatches = params.shape[0] / self.inf_batch_size
        L = []

        #import ipdb; ipdb.set_trace()
        # transforming back to original parameter space
        #params_dummy = np.zeros_like(params)
        #for k in range(params.shape[1]):
        #    params_dummy[:,k] = self.reparamSigmoid(self.cfg.bounds[k][0], self.cfg.bounds[k][1])(params[:,k])

        for batch in range(nbatches):
            params_cur = np.expand_dims(np.expand_dims(params[batch*self.inf_batch_size : (batch+1)*self.inf_batch_size, :],-1),1)
            pred_data = self.forward_sess.run(self.forward_model.output, feed_dict={self.forward_input:params_cur})
            likelihoods = self.likelihood(pred_data, target.reshape((-1,)), gamma)
            L.extend(likelihoods)
        return np.array(L)

 
    def evalProposalByComponentNormal(self, x, component):
        return stats.multivariate_normal.pdf(x, mean=self.mu_p[component], cov=self.std_p[component])

    def evalProposalByComponentTDist(self, x, component):
        return self.multivariate_t_distribution(x, self.mu_p[component], self.std_p[component], self.degrees_p[component], self.mu_p[component].shape[0])

    def generateFromProposalNormal(self):
        samples = np.array([])
        cur_iter = 0
        while samples.shape[0] < self.N:
            iter_samples = np.array([])
            component_indices = np.random.choice(
                                        self.n_components, # number of components
                                        p = self.alpha_p ,          # component probabilities
                                        size = self.N ,
                                        replace = True ) 
            _, unique_counts = np.unique(component_indices, return_counts=True)

            for c in range(self.n_components):
                if c >= unique_counts.shape[0]:
                    continue
                cur_samps = np.random.multivariate_normal(size = unique_counts[c], mean = self.mu_p[c], cov = self.std_p[c])
                if c == 0:
                    iter_samples = cur_samps
                else:
                    iter_samples = np.concatenate([iter_samples, cur_samps], axis=0)
            
            idx = self.getOOBIndices(iter_samples)
            iter_samples = iter_samples[idx[0],:]
            if cur_iter == 0:
                samples = iter_samples
            else:
                samples = np.concatenate([samples, iter_samples], axis = 0)
            cur_iter = cur_iter + 1

        return samples[:self.N,:]

    def generateFromProposalTDist(self):

        samples = np.array([])
        cur_iter = 0
        while samples.shape[0] < self.N:
            iter_samples = np.array([])
            component_indices = np.random.choice(
                                        self.n_components, # number of components
                                        p = self.alpha_p ,          # component probabilities
                                        size = self.N ,
                                        replace = True ) 
            _, unique_counts = np.unique(component_indices, return_counts=True)

            for c in range(self.n_components):
                if c >= unique_counts.shape[0]:
                    continue
                #import ipdb; ipdb.set_trace()
                cur_samps = self.multivariate_t_rvs(self.mu_p[c], self.std_p[c], df=self.degrees_p[c], n=unique_counts[c])
                if c == 0:
                    iter_samples = cur_samps
                else:
                    iter_samples = np.concatenate([iter_samples, cur_samps], axis=0)
            
            idx = self.getOOBIndices(iter_samples)
            iter_samples = iter_samples[idx[0],:]
            if cur_iter == 0:
                samples = iter_samples
            else:
                samples = np.concatenate([samples, iter_samples], axis = 0)
            cur_iter = cur_iter + 1

        return samples[:self.N,:]


    def getOOBIndices(self, x):
        v = np.zeros((x.shape[0],), dtype=bool)
        for k in range(x.shape[1]):
            v = v | ((x[:,k] < self.cfg.bounds[k][0]) | (x[:,k] > self.cfg.bounds[k][1]))
        return np.where(v == 0)

    def countOOB(self, x):
        v = np.zeros((x.shape[0],), dtype=bool)
        for k in range(x.shape[1]):
            v = v | ((x[:,k] < self.cfg.bounds[k][0]) | (x[:,k] > self.cfg.bounds[k][1]))
        return np.sum(v)

    def multivariate_t_rvs(self, m, S, df=np.inf, n=1):
        '''generate random variables of multivariate t distribution
        Parameters
        ----------
        m : array_like
        mean of random variable, length determines dimension of random variable
        S : array_like
        square array of covariance  matrix
        df : int or float
        degrees of freedom
        n : int
        number of observations, return random array will be (n, len(m))
        Returns
        -------
        rvs : ndarray, (n, len(m))
           each row is an independent draw of a multivariate t distributed
           random variable
        '''
        m = np.asarray(m)
        d = len(m)
        if df == np.inf:
            x = 1.
        else:
            x = np.random.chisquare(df, n)/df
        z = np.random.multivariate_normal(np.zeros(d),S,(n,))
        return m + z/np.sqrt(x)[:,None]   # same output format as random.multivariate_normal

    def multivariate_t_distribution(self, x,mu,Sigma,df,d):
        '''
        Multivariate t-student density:
        output:
            the density of the given element
        input:
            x = parameter (d dimensional numpy array or scalar)
            mu = mean (d dimensional numpy array or scalar)
            Sigma = scale matrix (dxd numpy array)
            df = degrees of freedom
            d: dimension
        '''
        Num = gamma(1. * (d+df)/2)
        inv_sqrt_sigma = np.linalg.inv(linalg.sqrtm(Sigma))
        Denom = ( gamma(1.*df/2) * np.power(df*pi,1.*d/2) * np.power(np.linalg.det(Sigma),1./2) * np.power(1 + (1./df)*np.sum(np.square(np.dot(x-mu, inv_sqrt_sigma)), axis=-1),1.* (d+df)/2))
        d = 1. * Num / Denom 
        return d
 
def plotMarginals(posterior_samples, params, filename):
    plt.figure()
    for k in range(params.shape[0]):
        plt.hist(posterior_samples[:,k], bins=100, alpha=0.2)
        #plt.scatter(params[k],0,s=10)
    plt.savefig(filename)
    plt.show()    

def model_inference(args):
    simdata = args['data']
    inf_cfg = config.Config(model=args['model'], bins=args['nbin'], N=args['N'])
    inference_class = Infer(config=inf_cfg)
    bounds = inf_cfg.bounds
    inference_class.target = simdata.reshape((-1,))
    output = differential_evolution(inference_class.objectivefn,bounds)
    inference_class.sess.close()
    return output.x

def run(datafile='../data/chong/chong_full_cnn_coh.pickle', nsample=6, model=None, nbin=None, N=None, proposal=None, n_components=24):
    
    # set up the parallel pool for MLE initialization
    n_workers = 8
    workers = Pool(n_workers)

    # load in the configurations
    cfg = config.Config(model=model, bins=nbin, N=N)

    tdist = True if proposal == 'tdist' else False

    # initialize the importance sampler
    i_sampler = ImportanceSampler(cfg, max_iters=50, tol=1e-6, nsamples=200000, tdist=tdist)
    
    #for sample in range(1,nsample):
    for sample in [nsample]:
        # let's choose the dataset for which we'll try to get posteriors
        '''
        my_data = pickle.load(open(datafile ,'rb'))
        data = my_data[1][sample]
        data_norm = data / data.sum()
        '''

        # get the parameter recovery data
        my_data = pickle.load(open(cfg.inference_dataset[0],'rb'))
        dataset_idx = sample
        data_norm = my_data[1][0][dataset_idx] # index by 'sample'
        data = data_norm * N

        '''DEPRECATED: Use the reverse model to initialize
        # get an initial point estimate
        #mu_initial, std_initial = i_sampler.getPointEstimate(data_norm)
 
        # Initializing the mixture
        #i_sampler.initializeMixtures(mu_initial, std_initial, n_components=12, mu_perturbation=(-.5, .5), spread=10., tdist=tdist)
        '''

        grand_start_time = time.time()
        #i_sampler.initializeMixturesMLE(data, n_components=24, tdist=tdist)
        rec_params = []
        tuples = [{'data':data, 'model':model, 'nbin':nbin, 'N':N} for x in range(n_components)]
        for _ in tqdm.tqdm(workers.imap(model_inference, tuples), total=n_components):
            rec_params.append(_)
            pass
        workers.close()
        i_sampler.initializeMixturesMLEV2(rec_params, n_components=n_components, tdist=tdist)
        init_end_time = time.time()

        # convergence metric
        norm_perplexity, cur_iter = -1.0, 0.
        # annealing factor
        gamma = 32.
        # nan counter
        nan_counter = 0
        # timing measures
        timing_sampler = []
        timing_nn = []
        timing_updates = []

        current_iter = 0
        perplexity_trace = [norm_perplexity]

        cov_mats = np.zeros((i_sampler.N,i_sampler.mu_p[0].shape[0], i_sampler.mu_p[0].shape[0]))

        while (cur_iter < i_sampler.max_iters):
        
            start_time = time.time()
            i_sampler.reval_components()
            # sample parameters from the proposal distribution
            X = i_sampler.generateFromProposal()
            end_time = time.time()
            timing_sampler.append(end_time-start_time)

            start_time = time.time()
            # evaluate the likelihood of observering these parameters
            log_target = i_sampler.getLikelihoodFromProposals(X, data, gamma)
            end_time = time.time()
            timing_nn.append(end_time-start_time)

            start_time = time.time()
            # numerical stability
            log_target = log_target - log_target.max()
            rho = np.zeros((i_sampler.n_components, i_sampler.N))

            # rho: mixture posterior probabilities
            for c in range(i_sampler.n_components):
                rho[c] = i_sampler.alpha_p[c] * i_sampler.evalProposalByComponent(X, c)

            rho_sum = np.sum(rho, axis = 0)
            rho = rho / rho_sum
            w = np.exp(log_target - np.log(rho_sum))
            w = w / np.sum(w)

            entropy = -1*np.sum(w * np.log(w + 1e-30))
            norm_perplexity_cur = np.exp(entropy)/i_sampler.N
            perplexity_trace.append(norm_perplexity_cur)

            if math.isnan(norm_perplexity) and math.isnan(norm_perplexity_cur):
                nan_counter = nan_counter + 1

            # convergence criterion
            if (gamma == 1.) and (((norm_perplexity - norm_perplexity_cur)**2 < i_sampler.tol) or ( (norm_perplexity >= 0.8) and ( (norm_perplexity - norm_perplexity_cur)**2 < 1e-3) ) or (nan_counter > 8)):
                break
            current_iter += 1
            # update annealing term
            diff = np.sign(norm_perplexity - norm_perplexity_cur)
            if diff < 0:
                gamma = np.maximum(gamma/2. , 1.)

            norm_perplexity = norm_perplexity_cur
            print('Step: {}, Perplexity: {}, Num OOB: {}, gamma: {}, alphas: {}'.format(cur_iter, norm_perplexity, i_sampler.countOOB(X), gamma, i_sampler.alpha_p))

            # update proposal model parameters; in our case it is the alpha (s), mu (s) and std (s)
            for c in range(i_sampler.n_components):
                tmp = w * rho[c]
                i_sampler.alpha_p[c] = np.sum(tmp)
                #import ipdb; ipdb.set_trace()
                # Parameter updates
                if tdist:
                    p = i_sampler.mu_p[0].shape[0]
                    v_d = i_sampler.degrees_p[c]
                    z_d = X - i_sampler.mu_p[c]
                    inv_sqrt_sigma = np.linalg.inv(linalg.sqrtm(i_sampler.std_p[c]))
                    gamma_proposal_c = (v_d + p) / (v_d + np.sum(np.square(np.dot(z_d, inv_sqrt_sigma)), axis=-1))
                    tmp_gamma = tmp * gamma_proposal_c
                    i_sampler.mu_p[c] = np.sum(X.transpose() * tmp_gamma, axis = 1) / np.sum(tmp_gamma)
                    #cov_mats = np.array([np.outer(x,x) for x in (X - i_sampler.mu_p[c])])
                    cov_mats = np.matmul(z_d[:,:,np.newaxis], z_d[:,np.newaxis,:])
                    i_sampler.std_p[c] =  np.sum(cov_mats * tmp_gamma[:,np.newaxis, np.newaxis], axis = 0) / i_sampler.alpha_p[c]

                else:
                    i_sampler.mu_p[c] = np.sum(X.transpose() * tmp, axis = 1) / i_sampler.alpha_p[c]
                    z_d = X - i_sampler.mu_p[c]
                    cov_mats = np.matmul(z_d[:,:,np.newaxis], z_d[:,np.newaxis,:])
                    #cov_mats = np.array([np.outer(x,x) for x in (X - i_sampler.mu_p[c])])
                    i_sampler.std_p[c] =  np.sum(cov_mats * tmp[:,np.newaxis, np.newaxis], axis = 0) / i_sampler.alpha_p[c]

            cur_iter += 1
            end_time = time.time()
            timing_updates.append(end_time-start_time)

        grand_end_time = time.time()
        print('Time elapsed: {}'.format(grand_end_time - grand_start_time))
        #print('Predicted variances from the reverse model: {}'.format(std_initial))
        post_idx = np.random.choice(w.shape[0], p=w, replace=True, size = 100000)
        posterior_samples = X[post_idx, :]
        print ('Covariance matrix: {}'.format(np.around(np.cov(posterior_samples.transpose()),decimals=6)))
        print ('Correlation matrix: {}'.format(np.around(np.corrcoef(posterior_samples.transpose()),decimals=6)))

        results = {'timeToInitialize':init_end_time-grand_start_time, 'final_x':X, 'final_w':w, 'posterior_samples':posterior_samples, 'alpha':i_sampler.alpha_p, 'mu':i_sampler.mu_p, 'cov':i_sampler.std_p, 'gt_params':my_data[0][dataset_idx], 'timeToConvergence':grand_end_time-grand_start_time, 'norm_perplexity':perplexity_trace, 'log_likelihood':log_target, 'timing_nn':timing_nn, 'timing_updates':timing_updates, 'timing_sampler':timing_sampler}
        #results = {'final_x':X, 'final_w':w, 'posterior_samples':posterior_samples, 'alpha':i_sampler.alpha_p, 'mu':i_sampler.mu_p, 'cov':i_sampler.std_p, 'timeToConvergence':end_time-start_time, 'norm_perplexity':norm_perplexity}


        #pickle.dump(results, open(os.path.join(cfg.results_dir, 'results_chong_sample_{}_model_{}.pickle'.format(sample,cfg.refname)),'wb'))
        f =  gzip.open(os.path.join(cfg.results_dir, 'time_benchmark_eLIFE_exps/IS_model_{}_N_{}_idx_{}_{}.pklz'.format(cfg.refname,N,dataset_idx,proposal)),'wb')
        pickle.dump(results,f)
        f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--nbin', type=int)
    parser.add_argument('--N', type=int)
    parser.add_argument('--nsample', type=int)
    parser.add_argument('--proposal', type=str)
    args = parser.parse_args()
    run(nsample=args.nsample, model=args.model, nbin=args.nbin, N=args.N, proposal=args.proposal)
