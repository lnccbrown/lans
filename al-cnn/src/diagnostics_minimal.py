import numpy as np

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
