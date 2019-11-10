import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import tqdm
from mpl_toolkits.mplot3d import Axes3D

def eval_likelihood(x, mu_l, std_l, alpha_l):
    n_components = alpha_l.shape[0]
    n_dims = mu_l.shape[1]
    target = np.zeros((x.shape[0],))
    for a in range(n_components):
        target += alpha_l[a] * stats.multivariate_normal.pdf(x, mean=mu_l[a], cov=std_l[a])
    return target

def eval_proposal_by_component(x, mu_d, std_d):
    target = stats.multivariate_normal.pdf(x, mean=mu_d, cov=std_d)
    return target

def gen_from_proposal(mu_p, std_p, alpha_p, n):
    component_indices = np.random.choice(alpha_p.shape[0], # number of components
		                         p = alpha_p, # component probabilities
		     			 size = n,
		     			 replace = True)
    _, unique_counts = np.unique(component_indices, return_counts=True)
    samples = np.array([])
    for c in range(alpha_p.shape[0]):
	if c >= unique_counts.shape[0]:
	    continue
	cur_samps = np.random.multivariate_normal(size = unique_counts[c], mean = mu_p[c], cov = std_p[c])
	if c == 0:
	    samples = cur_samps
	else:
            samples = np.concatenate([samples, cur_samps], axis=0)
    return samples

def main():
    max_iters = 100
    tol = 1e-7
    N = 100000

    # this is what we want to achieve
    alpha_l = np.array([0.5, 0.5])
    mu_l = np.array([[1., 0.], [5., 0.], [-3., -3]])
    std_l = np.array([
		     [[.1, .0], [.0, .1]],
		     [[.1, .0], [.0, .1]],
                     [[.1, .0], [.0, .1]]
		])

    # proposal
    alpha_p = np.array([0.3, 0.2, 0.2, 0.3])
    mu_p = np.array([[3., -1.], [0., 1.], [-1., 1.], [-3,-3]])
    std_p = np.array([ 
		[[1., 0.5], [0.5, 1.]], 
		[[3., 0.], [0., 3.]],
		[[1., 0.], [0., 1.]],
		[[0.2, 0.],[0., 0.2]]  
		])
    
    # Data from actual dist
    X_true = gen_from_proposal(mu_l, std_l, alpha_l, N)
    # initial data
    X = gen_from_proposal(mu_p, std_p, alpha_p, N)

    n_components = alpha_p.shape[0]
    norm_perplexity = 0
    cur_iter = 0

    while (cur_iter < max_iters):
	print(alpha_p)
	X = gen_from_proposal(mu_p, std_p, alpha_p, N)
	target = eval_likelihood(X, mu_l, std_l, alpha_l)
	
	rho = np.zeros((n_components, N))
        # get rhos
	for c in range(n_components):
	    rho[c] = alpha_p[c] * eval_proposal_by_component(X, mu_p[c], std_p[c])
	rho_sum = np.sum(rho, axis = 0)
	rho = rho / rho_sum
	w = np.exp(np.log(target) - np.log(rho_sum))
	w = w / np.sum(w)

	entropy = -1*np.sum(w * np.log(w))
	norm_perplexity_cur = np.exp(entropy)/N
	if (norm_perplexity - norm_perplexity_cur)**2 < tol:
	    break
	norm_perplexity = norm_perplexity_cur

        # update proposal model parameters; in our case it is the alpha (s), mu (s) and std (s)
	for c in range(n_components):
	    tmp = w * rho[c]
	    alpha_p[c] = np.sum(tmp)
	    mu_p[c] = np.sum(X.transpose() * tmp, axis = 1) / alpha_p[c]
	    cov_mats = np.array([np.outer(x,x) for x in (X - mu_p[c])])
	    std_p[c] =  np.sum(cov_mats * tmp[:,np.newaxis, np.newaxis], axis = 0) / alpha_p[c]

	cur_iter += 1
    return X, w, X_true

if __name__ == '__main__':
    X, w, X_true = main()
    post_idx = np.random.choice(w.shape[0], p = w, replace = True, size = 100000)
    posterior_samples = X[post_idx, :]

    fig = plt.figure()
    #plt.hist2d(X_true[:,0], X_true[:,1], bins = 100, alpha = 0.2)
    plt.hist(posterior_samples[:, 0], bins = 100, alpha = 0.2)
    plt.hist(X_true[:, 0], bins = 100, alpha = 0.2)
    plt.figure()
    plt.hist(posterior_samples[:, 1], bins = 100, alpha = 0.2)
    plt.hist(X_true[:, 1], bins = 100, alpha = 0.2)
    plt.show()

