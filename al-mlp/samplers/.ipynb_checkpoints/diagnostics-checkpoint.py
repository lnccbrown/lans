import numpy as np
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.api import acf, pacf, graphics

# Gelman Rubin Univariate
# ----------------------------------------------------------------------
def _get_W(chains):
    return np.sum(np.square((chains - np.mean(chains, axis = 1, keepdims = 1))), axis = (0, 1)) / (chains.shape[0] * (chains.shape[1] - 1))
    
def _get_B_div_n(chains):
    return np.sum(np.square(np.mean(chains, axis = 1, keepdims = 1) - np.mean(chains, axis = (0,1), keepdims = 1)), axis = (0, 1)) / (chains.shape[0] - 1)
    
def get_gelman_rubin_univ(chains = None, 
                          burn_in = 1000,
                          thresh = 1.01):

    W = _get_W(chains = chains[:, burn_in:, :])
    B_div_n = _get_B_div_n(chains = chains[:, burn_in:, :])
    V = (chains[:, burn_in:, :].shape[1] - 1) / chains[:, burn_in:, :].shape[1] * W + B_div_n 
    
    # Get R_hat statistic
    R_hat = V / W
    
    # Continue if any of the univariate R_hat statistics is above threshold
    continue_ = int(np.sum(R_hat > thresh) > 0)
    return continue_, R_hat
# ----------------------------------------------------------------------


# Gelman Rubin Multivariate
# ----------------------------------------------------------------------
def _get_W_mv(chains):
    tmp = np.zeros((chains.shape[2], chains.shape[2]))
    for i in range(chains.shape[0]):
        tmp += np.cov(chains[i].T)
    return tmp / chains.shape[0]

def _get_B_div_n_mv(chains):
    phi_dot = np.mean(chains, axis = 1)
    #phi_dot_dot = np.mean(phi_dot, axis = 0)
    return np.cov(phi_dot.T) / chains.shape[0]  


def get_gelman_rubin_mv(chains,
                        burn_in = 1000,
                        thresh = 1.01):
    
    # Get component quantities
    W = _get_W_mv(chains = chains[:, burn_in:, :])
    B_div_n = _get_B_div_n_mv(chains = chains[:, burn_in:, :])
    W_inv = np.linalg.inv(W)
    
    # Pick top eigenvalue of W^(-1)B
    lambda_1 = np.sort(np.linalg.eigvals(np.dot(W_inv, B_div_n)))[-1]
    
    # Compute R_hat from it
    R_hat = (chains[:, burn_in:, :].shape[1] - 1) / chains[:, burn_in:, :].shape[1] + ((chains[:, burn_in: :].shape[0] + 1) / chains[:, burn_in:, :].shape[0]) * lambda_1
    
    # Continue sampling if R_hat statistic is above threshold
    continue_ = int(R_hat > thresh)
    return continue_, R_hat
# ----------------------------------------------------------------------


# Geweke statistic (works on single chains too)
# ----------------------------------------------------------------------
def get_geweke_diags(chains, 
                     split = 0.3, 
                     skip = 0.5):
    
    """Function computes geweke statistic for markov chains"""
    # Check dimensionality of chains
    # If single chain add dimesion
    n_dims = len(chains.shape)
    if n_dims == 2:
        chains = np.expand_dims(chains, axis = 0)

    # Compute split demarcations as integers to be used for indexing
    n_floor = int(chains.shape[1] * (split + skip))
    n_skip = int(chains.shape[1] * (skip))
    
    # Initialize the vector in which we store z-scores
    z_scores = np.zeros(chains.shape[0] * chains.shape[2])
    
    # Main loop that computes statistics of interest
    for i in range(chains.shape[0]):
        for j in range(chains.shape[2]):
            # Get Autoregression coefficients for each part of split of chain
            sel_1 = ar_select_order(chains[i, n_skip:n_floor, j], maxlag = 10, seasonal = False)
            sel_2 = ar_select_order(chains[i, n_floor:, j], maxlag = 10, seasonal = False)
            res_1 = sel_1.model.fit()
            res_2 = sel_2.model.fit()
            
            # Compute the Autoregression corrected respective standard deviations
            s_1 = res_1.sigma2 / np.square(1 - np.sum(res_1.params[1:]))
            s_2 = res_2.sigma2 / np.square(1 - np.sum(res_2.params[1:]))
            
            # Compute (absolute) z scores that form the basis of the test of whether or not to continue sampling
            z_scores[i * chains.shape[2] + j] = np.abs((np.mean(chains[i, n_skip:n_floor, j]) - np.mean(chains[i, n_floor:, j])) / np.sqrt((1 / (n_floor - n_skip)) * s_1  + (1 / (chains.shape[1] - n_floor)) * s_2))
    
    # Continuation check: All absolute z scores below 2? If yes stop sampling
    continue_ = int((np.sum(z_scores > 2)) > 0)
    return continue_, z_scores

# ----------------------------------------------------------------------

# Effective sample size 
# ----------------------------------------------------------------------
def neff_mcmc_univ(chains,
                   burn_in = 1000,
                   min_neff = 50,
                   max_lag = 200):
    
    n_dims = len(chains.shape)
    if n_dims == 2:
        chains = np.expand_dims(chains, axis = 0)
    
    neffs = np.zeros(chains.shape[0] * chains.shape[2])

    # Main loop that computes statistics of interest
    for i in range(chains.shape[0]):
        for j in range(chains.shape[2]):
            # Get Autoregression coefficients for each part of split of chain
            acf_vals_tmp = acf(t_swapped[0, 1000:, 0], nlags = max_lag)
            pos_id = np.argwhere(acf_vals_tmp < 0)
            if len(pos_id) > 1:
                max_lag_tmp = np.min(pos_id)
            else:
                max_lag_tmp = max_lag
            
            neffs[i + chains.shape[2] + j] = chains.shape[1] / (1 + (2 * np.sum(acf_vals_tmp[:max_lag])))
    
    # Continuation check: All absolute z scores below 2? If yes stop sampling
    continue_ = int((np.sum(neffs < 50)) > 0)
    return continue_, neffs
# -----------------------------------------------------------------------