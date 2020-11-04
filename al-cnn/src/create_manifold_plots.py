import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib as mpl

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
	

	#import time

        for batch in range(nbatches):
            params_cur = np.expand_dims(np.expand_dims(params[batch*self.inf_batch_size : (batch+1)*self.inf_batch_size, :],-1),1)
            #st = time.time()
            pred_data = self.forward_sess.run(self.forward_model.output, feed_dict={self.forward_input:params_cur})
            #et = time.time()
            #print(et-st)
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

def run(datafile='../data/chong/chong_full_cnn_coh.pickle', nsample=6, model=None, nbin=None, N=None, proposal=None, n_components=24, start_params=[], vary_range=[]):
    # load in the configurations
    cfg = config.Config(model=model, bins=nbin, N=N)
    tdist = True if proposal == 'tdist' else False
    # initialize the importance sampler
    i_sampler = ImportanceSampler(cfg, max_iters=50, tol=1e-6, nsamples=200000, tdist=tdist)
    param_bounds = cfg.bounds
    n_params = len(param_bounds)

    for param in range(n_params):
	baseline_params = np.array(start_params).copy()
	pts = np.linspace(vary_range[param][0], vary_range[param][1], 25)
	X, Y, Z = [], [], []

	'''
        n_levels = 25
        # Data template
        plot_data = np.zeros((4000, 2))
        plot_data[:, 0] = np.concatenate(([i * 0.005 for i in range(2000, 0, -1)], [i * 0.005 for i in range(1, 2001, 1)]))
        plot_data[:, 1] = np.concatenate((np.repeat(-1, 2000), np.repeat(1, 2000)))
        data_var = np.zeros((4000 * n_levels, n_params + 3))
        cnt = 0 
        vary_range = param_bounds[param]
        for par_tmp in np.linspace(vary_range[0], vary_range[1], n_levels):
            tmp_begin = 4000 * cnt
            tmp_end = 4000 * (cnt + 1)
            baseline_params[param] = par_tmp
            data_var[tmp_begin:tmp_end, :len(params)] = baseline_params
            data_var[tmp_begin:tmp_end, len(params):(len(params) + 2)] = plot_data
            # print(data_var.shape)
            data_var[tmp_begin:tmp_end, (len(params) + 2)] = np.squeeze(np.exp(keras_model.predict(data_var[tmp_begin:tmp_end, :-1], 
                                                                                               batch_size = 100)))
            cnt += 1
	'''
	for pt in tqdm.tqdm(range(pts.shape[0])):
	    params_cur = baseline_params
	    params_cur[param] = pts[pt]
	    params_cur = np.expand_dims(np.expand_dims(np.expand_dims(params_cur,axis=0),axis=1), axis=-1)
	    pred_data = i_sampler.forward_sess.run(i_sampler.forward_model.output, feed_dict={i_sampler.forward_input:params_cur})

	    pred_data = pred_data.reshape(-1, cfg.output_hist_dims[-1])
	    X.extend(np.arange(-511,511,1))
	    Y.extend([pts[pt]]*1022)
	    Z.extend(np.flip(pred_data[:-1,0]).tolist() + pred_data[:-1,1].tolist())

	#fig = plt.figure(figsize=(8, 5.5))
	fig = plt.figure()
	ax = fig.add_subplot(111, projection = '3d')
	ax.plot_trisurf(X, 
		    Y, 
		    Z,
		    linewidth = 0., 
		    alpha = 1.0, 
		    cmap = cm.coolwarm)
	ax.set_ylabel(cfg.param_names[param],  
		  fontsize = 16,
		  labelpad = 20)
	ax.set_xlabel('RT',  
		  fontsize = 16, 
		  labelpad = 20)
	ax.set_zlabel('Likelihood',  
		  fontsize = 16, 
		  labelpad = 20)
	'''
	ax.set_zticks(np.round(np.linspace(min(Z), max(Z), 5), 1))
	ax.set_yticks(np.round(np.linspace(min(Y), max(Y), 5), 1))
	ax.set_xticks(np.round(np.linspace(min(X), max(X), 5), 1))
	'''

	ax.tick_params(labelsize = 16)
	ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
	ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
	ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

	#plt.savefig('../figures/cnn_manifold_' + model + '_vary_' + cfg.param_names[param] + '.svg',
	#		format = 'svg', 
	#		transparent = True,
	#		frameon = False)
	plt.savefig('../figures/cnn_manifold_' + model + '_vary_' + cfg.param_names[param] + '.png')
	plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--nbin', type=int)
    parser.add_argument('--N', type=int)
    parser.add_argument('--nsample', type=int)
    parser.add_argument('--proposal', type=str)
    args = parser.parse_args()
    if args.model == 'ddm' or args.model == 'ddm_analytic':
        start_params = [0., 1, 0.5, 1]
        vary_range_vec = [[-2, 2], [0.3, 2], [0.2, 0.8], [0, 2]]
        run(nsample=args.nsample, model=args.model, nbin=args.nbin, N=args.N, proposal=args.proposal, start_params=start_params, vary_range=vary_range_vec)
