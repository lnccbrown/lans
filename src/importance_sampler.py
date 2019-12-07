import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import tqdm, pickle, os, time
from mpl_toolkits.mplot3d import Axes3D
from reverse_model import cnn_reverse_model
from train_detector import cnn_model_struct
import config
import tensorflow as tf
import pandas as pd
import time

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
    def __init__(self, config, max_iters=100, tol=1e-7, nsamples=1e5):	

	self.max_iters = max_iters
	self.tol = tol
	self.N = nsamples

	self.cfg = config
	self.target = []

	self.inf_batch_size = 25000

	# placeholder for forward model
	self.forward_model_inpdims = [self.inf_batch_size] + self.cfg.param_dims[1:]
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
	ckpts = tf.train.latest_checkpoint(os.path.join(self.cfg.base_dir, 
							'models',
							'rev_'+self.cfg.model_name+'_'+self.cfg.model_suffix))
	self.saver1.restore(self.inv_sess, ckpts)

    def initializeMoG(self, mu_initial, std_initial, n_components=3, mu_perturbation=(-2., 2.), spread=10.):
        
	n_dims = mu_initial.shape[0]
	# set the inital component weights
	alpha_p = np.random.randint(low=1, high=n_components-1, size=n_components).astype(np.float32)
	alpha_p[0] = n_components
	alpha_p = alpha_p / alpha_p.sum()

	# initialize the mu and sigma
	mu_p = np.zeros((n_components, n_dims))
	std_p = np.zeros((n_components, n_dims, n_dims))

	# set the first component right on the point estimate
	mu_p[0] = mu_initial
	std_p[0] = np.diag(std_initial)

	# initialize the other components by perturbing them a little bit around the point estimate
	for c in range(1, n_components):
            mu_p[c] = mu_p[0] + np.random.uniform(low=mu_perturbation[0], high=mu_perturbation[1], size=n_dims)*std_initial
            std_p[c] = spread * std_p[0]
        
	# set the members
        self.n_components = n_components
	self.alpha_p = alpha_p
	self.mu_p = mu_p
	self.std_p = std_p

    def klDivergence(self, x, y, eps1=1e-7, eps2=1e-30):
	return np.sum(x * np.log(eps2 + x/(y+eps1)))

    def likelihood(self, x, y, gamma, eps=1e-7):
	return np.sum(np.log(x+eps)*y/gamma, axis=1)

    '''
    feed forward through the inverse model to get a point estimate of the parameters that could've generated a given dataset
    '''
    def getPointEstimate(self, dataset):
	#ndatapoints = dataset.shape[0]
	params = self.inv_sess.run(self.inv_model.output, feed_dict={self.inv_input: dataset.reshape(self.inv_model_inpdims)})
	nparams = np.prod(self.cfg.param_dims[1:])
	#means, stds = params[:ndatapoints+1][:nparams], params[:ndatapoints+1][nparams:]
	means, stds = params[0][:nparams], params[0][nparams:]
	return means, stds

    def getLikelihoodFromProposals(self, params, target, gamma):
	nbatches = params.shape[0] / self.inf_batch_size
	L = []
	for batch in range(nbatches):
	    params_cur = np.expand_dims(np.expand_dims(params[batch*self.inf_batch_size : (batch+1)*self.inf_batch_size, :],-1),1)
	    pred_data = self.forward_sess.run(self.forward_model.output, feed_dict={self.forward_input:params_cur})
	    likelihoods = self.likelihood(pred_data, target.reshape((-1,)), gamma)
	    L.extend(likelihoods)
	return np.array(L)

    '''
    def eval_likelihood(x, mu_l, std_l, alpha_l):
        n_components = alpha_l.shape[0]
        n_dims = mu_l.shape[1]
        target = np.zeros((x.shape[0],))
        for a in range(n_components):
            target += alpha_l[a] * stats.multivariate_normal.pdf(x, mean=mu_l[a], cov=std_l[a])
        return target
    '''

    def evalProposalByComponent(self, x, component):
        return stats.multivariate_normal.pdf(x, mean=self.mu_p[component], cov=self.std_p[component])
    
    def generateFromProposal(self):

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

def plotMarginals(posterior_samples, params, filename):
    plt.figure()
    for k in range(params.shape[0]):
        plt.hist(posterior_samples[:,k], bins=100, alpha=0.2)
	#plt.scatter(params[k],0,s=10)
    plt.savefig(filename)
    plt.show()    

def run(datafile='../data/bg_stn/bg_stn_binned.pickle', sample=0):
    # let's choose the dataset for which we'll try to get posteriors
    my_data = pickle.load(open(datafile ,'rb'))
    data = my_data[0][sample]
    data_norm = data / data.sum()

    # load in the configurations
    cfg = config.Config()

    # initialize the importance sampler
    i_sampler = ImportanceSampler(cfg, max_iters=100, tol=1e-6, nsamples=500000)

    # get an initial point estimate
    mu_initial, std_initial = i_sampler.getPointEstimate(data_norm)
 
    # Initializing the mixture
    i_sampler.initializeMoG(mu_initial, std_initial, n_components=5, mu_perturbation=(-1., 1.), spread=10.)

    # convergence metric
    norm_perplexity, cur_iter = -1.0, 0.

    # annealing factor
    gamma = 64.

    start_time = time.time()
    while (cur_iter < i_sampler.max_iters):

	# sample parameters from the proposal distribution
	X = i_sampler.generateFromProposal()
	# evaluate the likelihood of observering these parameters
	log_target = i_sampler.getLikelihoodFromProposals(X, data, gamma)
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

	entropy = -1*np.sum(w * np.log(w))
	norm_perplexity_cur = np.exp(entropy)/i_sampler.N
	if (norm_perplexity - norm_perplexity_cur)**2 < i_sampler.tol:
	    break

        # update annealing term
	diff = np.sign(norm_perplexity - norm_perplexity_cur)
	if diff < 0:
            gamma = np.maximum(gamma/2. , 1.)

	norm_perplexity = norm_perplexity_cur
	print('Step: {}, Perplexity: {}, Num OOB: {}'.format(cur_iter, norm_perplexity, i_sampler.countOOB(X)))

        # update proposal model parameters; in our case it is the alpha (s), mu (s) and std (s)
	for c in range(i_sampler.n_components):
	    tmp = w * rho[c]
	    i_sampler.alpha_p[c] = np.sum(tmp)
	    i_sampler.mu_p[c] = np.sum(X.transpose() * tmp, axis = 1) / i_sampler.alpha_p[c]
	    cov_mats = np.array([np.outer(x,x) for x in (X - i_sampler.mu_p[c])])
	    i_sampler.std_p[c] =  np.sum(cov_mats * tmp[:,np.newaxis, np.newaxis], axis = 0) / i_sampler.alpha_p[c]

	cur_iter += 1

    end_time = time.time()
    print('Time elapsed: {}'.format(end_time - start_time))

    print('Predicted variances from the reverse model: {}'.format(std_initial))
    post_idx = np.random.choice(w.shape[0], p=w, replace=True, size = 100000)
    posterior_samples = X[post_idx, :]
    print ('Covariance matrix: {}'.format(np.around(np.cov(posterior_samples.transpose()),decimals=6)))
    print ('Correlation matrix: {}'.format(np.around(np.corrcoef(posterior_samples.transpose()),decimals=6)))

    #df = pd.DataFrame(posterior_samples)
    #pd.scatter_matrix(df, figsize=(6,6), alpha=0.01)
    #plt.savefig(os.path.join(cfg.results_dir, 'posteriors_{}_covariances.png'.format(cfg.model_name)))
    #plt.show(block=False)

    #params = np.array([0., 0., 0., 0., 0.])
    #plotMarginals(posterior_samples, params, os.path.join(cfg.results_dir, 'posteriors_{}_marginals.png'.format(cfg.model_name)))    

    results = {'final_x':X, 'final_w':w, 'posterior_samples':posterior_samples, 'alpha':i_sampler.alpha_p, 'mu':i_sampler.mu_p, 'cov':i_sampler.std_p}
    pickle.dump(results, open(os.path.join(cfg.results_dir, 'results_bg_stn_sample_{}'.format(sample)),'wb'))

def main():
    nsamples = 6
    run(sample=3)

if __name__ == '__main__':
    main()
