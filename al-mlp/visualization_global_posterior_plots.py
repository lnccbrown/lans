import os
import pickle
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import re
import argparse
import seaborn as sns
import yaml
from string import ascii_letters
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression
import cddm_data_simulation as cds
import boundary_functions as bf
from datetime import datetime
from statsmodels.distributions.empirical_distribution import ECDF
import scipy as scp
from basic_simulator import simulator
from functools import partial

# Config -----
config = {'ddm': {'params':['v', 'a', 'w', 'ndt'],
                  'param_bounds': [[-2.7, 0.4, 0.15, 0.05], [2.7, 2.2, 0.85, 2.0]],
                 },
          'angle':{'params': ['v', 'a', 'w', 'ndt', 'theta'],
                   'param_bounds': [[-2.7, 0.4, 0.3, 0.1, - .1], [2.7, 1.7, 0.7, 1.9, np.pi / 2 - 0.3]],
                  },
          'weibull_cdf':{'params': ['v', 'a', 'w', 'ndt', 'alpha', 'beta'],
                         'param_bounds': [[-2.7, 0.4, 0.3, 0.1, 0.5, 0.5], [2.7, 1.7, 0.7, 1.9, 4.5, 6.5]]
                        },
          'levy':{'params':['v', 'a', 'w','alpha-diff', 'ndt'],
                  'param_bounds':[[-2.7, 0.4, 0.3, 1.1, 0.1], [2.7, 1.7, 0.7, 1.9, 1.9]]
                 },
          'ddm_sdv':{'params':['v', 'a', 'w', 'ndt', 'sdv'],
                     'param_bounds':[[-2.7, 0.5, 0.15, 0.05, 0.3],[2.7, 2.2, 0.85, 1.95, 2.2]]
                    },
          'full_ddm':{'params':['v', 'a', 'w', 'ndt', 'dw', 'sdv', 'dndt'],
                      'param_bounds':[[-2.5, 0.4, 0.25, 0.3, 0.05, 0.3],[2.5, 1.8, 0.65, 2.2, 0.25, 1.7]]
                     },
         }

def get_r2_vec(estimates = [0, 0, 0],
               ground_truths = [0, 0, 0]):
    """Function reads in parameter estimates and group truths and returns regression function"""
    r2_vec = []
    for i in range(estimates.shape[1]):
        reg = LinearRegression().fit(np.asmatrix(estimates[:, i]).T, np.asmatrix(ground_truths[:, i]).T)
        r2_vec.append(str(round(reg.score(np.asmatrix(estimates[:, i]).T, np.asmatrix(ground_truths[:, i]).T), 2)))
    return r2_vec

def hdi_eval(posterior_samples = [],
             ground_truths = []):
    
    vec_dim_1 = posterior_samples.shape[0]
    vec_dim_2 = posterior_samples.shape[2]
    
    vec = np.zeros((vec_dim_1, vec_dim_2))

    for i in range(vec_dim_1):
        for j in range(vec_dim_2):
            my_cdf = ECDF(posterior_samples[i, :, j])
            vec[i,j] = my_cdf(ground_truths[i, j])

        if i % 100 == 0:
            print(i)
  
    # Get calibration statistics
    prop_covered_by_param = []
    for i in range(vec.shape[1]):
        print(vec[:, i])
        prop_covered_by_param.append(np.sum((vec[:, i] > 0.01) * (vec[:, i] < 0.99)) / vec[:, :].shape[0])
    
    prop_covered_all = (vec[:, 0] > 0.01) * (vec[:, 0] < 0.99)
    for i in range(1, vec.shape[1], 1):
        prop_covered_all = prop_covered_all * (vec[:, i] > 0.01) * (vec[:, i] < 0.99)
    prop_covered_all = np.sum(prop_covered_all) / vec.shape[0]
    
    return vec, prop_covered_by_param, prop_covered_all

def sbc_eval(posterior_samples = [],
             ground_truths = []):
    vec_dim_1 = posterior_samples.shape[0]
    vec_dim_2 = posterior_samples.shape[2]
    n_post_samples = posterior_samples.shape[1]
    
    rank_mat = np.zeros((vec_dim_1, vec_dim_2))
    
    for i in range(vec_dim_1):
        for j in range(vec_dim_2):
            samples_tmp = posterior_samples[i, np.random.choice(n_post_samples, size = 100), j]
            samples_tmp.sort()
            rank_mat[i, j] = np.sum(samples_tmp <= ground_truths[i, j])
    return rank_mat

# PREPARE mcmc_dict for plotting

def clean_mcmc_dict(mcmc_dict = {},
                    filter_ = 'choice_p', # 'boundary', 'choice_p' 'none'
                    choice_p_lim = 0.95,
                    param_lims = [],
                    method = []):
    
    # Filter out cases that have choice which are too imbalanced
    
    # Get indices of imbalanced samples
    n_data = mcmc_dict['data'].shape[1]
    n_params = mcmc_dict['data'].shape[0]
    n_choices = np.unique(mcmc_dict['data'][0, :, 1]).shape[0]
    test_choice = np.unique(mcmc_dict['data'][0, : , 1])[0]
    ok_ids = np.zeros(n_params, dtype = np.bool)
    
    if filter_ == 'choice_p':
        if method == 'mlp' or method == 'navarro':
            for i in range(n_params):
                ok_ids[i] = (np.sum(mcmc_dict['data'][i, :, 1] == test_choice) < (n_data * choice_p_lim) and (np.sum(mcmc_dict['data'][i, :, 1] == test_choice) > (n_data * (1 - choice_p_lim))))
        if method == 'cnn':
            for i in range(n_params):
                ok_ids[i] = (np.sum(mcmc_dict['data'][i, :, 1]) < (choice_p_lim) and (np.sum(mcmc_dict['data'][i, :, 1]) > (1 - choice_p_lim)))
                # print(ok_ids[i])
                # print(np.sum(mcmc_dict['data'][i, :, 1]))
            
#             if i == 100:
#                 print(mcmc_dict['data'])
#                 print(mcmc_dict['data'].shape)
#             if not ok_ids[i]:
#                 print('rejected')
                       
    # Filter out severe boundary cases
    if filter_ == 'boundary':
        cnt = 0
        adj_size = 0.1
        for param_bnd_tmp in param_lims:
            if ax_titles[cnt] == 'ndt':
                cnt += 1
            else:
                if cnt == 0:
                    bool_vec = ( mcmc_dict['means'][:, cnt] > param_bnd_tmp[1] - 0.1 ) + ( mcmc_dict['means'][:, cnt] < param_bnd_tmp[0] + 0.1 ) 
                    cnt += 1
                else:
                    bool_vec = (bool_vec + (( mcmc_dict['means'][:, cnt] > param_bnd_tmp[1] - 0.1 ) + ( mcmc_dict['means'][:, cnt] < param_bnd_tmp[0] + 0.1 ))) > 0
                    cnt += 1
            print(np.sum(1 - bool_vec))
        print(np.sum(1 - bool_vec))

        ok_ids = (1 - bool_vec) > 0
        
    if filter_ == 'none':
        ok_ids = (1 - ok_ids) > 0

    for tmp_key in mcmc_dict.keys():
        print(tmp_key)
        #print(np.array(mcmc_dict[tmp_key]))
        mcmc_dict[tmp_key] = np.array(mcmc_dict[tmp_key])[ok_ids]

    # Calulate quantities from posterior samples
    mcmc_dict['sds'] = np.std(mcmc_dict['posterior_samples'][:, :, :], axis = 1)
    mcmc_dict['sds_mean_in_row'] = np.min(mcmc_dict['sds'], axis = 1)
    mcmc_dict['gt_cdf_score'], mcmc_dict['p_covered_by_param'], mcmc_dict['p_covered_all'] = hdi_eval(posterior_samples = mcmc_dict['posterior_samples'],
                                                                                                      ground_truths = mcmc_dict['gt'])
    mcmc_dict['gt_ranks'] = sbc_eval(posterior_samples = mcmc_dict['posterior_samples'],
                                     ground_truths = mcmc_dict['gt'])

    # Get regression coefficients on mcmc_dict 
    mcmc_dict['r2_means'] = get_r2_vec(estimates = mcmc_dict['means'], 
                              ground_truths = mcmc_dict['gt'])

    mcmc_dict['r2_maps'] = get_r2_vec(estimates = mcmc_dict['maps'], 
                              ground_truths = mcmc_dict['gt'])

    #mcmc_dict['gt'][mcmc_dict['r_hats'] < r_hat_cutoff, :]

    mcmc_dict['boundary_rmse'], mcmc_dict['boundary_dist_param_euclid'] = compute_boundary_rmse(mode = 'max_t_global',
                                                                                                boundary_fun = bf.weibull_cdf,
                                                                                                parameters_estimated = mcmc_dict['means'],
                                                                                                parameters_true = mcmc_dict['gt'],
                                                                                                data = mcmc_dict['data'],
                                                                                                model = model,
                                                                                                max_t = 20,
                                                                                                n_probes = 1000)

    mcmc_dict['euc_dist_means_gt'] = np.linalg.norm(mcmc_dict['means'] - mcmc_dict['gt'], axis = 1)
    mcmc_dict['euc_dist_maps_gt'] = np.linalg.norm(mcmc_dict['maps'] - mcmc_dict['gt'], axis = 1)
    mcmc_dict['euc_dist_means_gt_sorted_id'] = np.argsort(mcmc_dict['euc_dist_means_gt'])
    mcmc_dict['euc_dist_maps_gt_sorted_id'] = np.argsort(mcmc_dict['euc_dist_maps_gt'])
    mcmc_dict['boundary_rmse_sorted_id'] = np.argsort(mcmc_dict['boundary_rmse'])
    mcmc_dict['method'] = method
    
    return mcmc_dict

def a_of_t_data_prep(mcmc_dict = {},
                     model = 'weibull_cdf',
                     n_eval_points = 1000,
                     max_t = 20,
                     p_lims = [0.2, 0.8],
                     n_posterior_subsample = 10,
                     split_ecdf = False,
                     bnd_epsilon = 0.2):
    
    n_posterior_samples = mcmc_dict['posterior_samples'].shape[1]
    n_param_sets = mcmc_dict['gt'].shape[0]
    n_choices = 2
    cdf_list = []
    eval_points = np.linspace(0, max_t, n_eval_points)
    
    # boundary_evals = 
    dist_in = np.zeros(n_param_sets)
    dist_out = np.zeros(n_param_sets)
    
    gt_bnd_pos_mean_in = np.zeros(n_param_sets)
    gt_bnd_pos_mean_out = np.zeros(n_param_sets)
    post_bnd_pos_mean_in = np.zeros(n_param_sets)
    post_bnd_pos_mean_out = np.zeros(n_param_sets)
    
    for i in range(n_param_sets):
        if (i % 10) == 0:
            print(i)
        
        if model == 'weibull_cdf' or model == 'weibull_cdf2':
            out = cds.ddm_flexbound(v = mcmc_dict['gt'][i, 0],
                                    a = mcmc_dict['gt'][i, 1],
                                    w = mcmc_dict['gt'][i, 2],
                                    ndt = 0,
                                    #ndt = mcmc_dict['gt'][i, 3],
                                    delta_t = 0.001, 
                                    s = 1,
                                    max_t = 20, 
                                    n_samples = 2500,
                                    boundary_fun = bf.weibull_cdf,
                                    boundary_multiplicative = True,
                                    boundary_params = {'alpha': mcmc_dict['gt'][i, 4],
                                                       'beta': mcmc_dict['gt'][i, 5]})

            in_ = np.zeros(n_eval_points, 
                           dtype = np.bool)
            
            if split_ecdf:
                
                bin_c = [0, 0]
                if np.sum(out[1] == - 1) > 0:
                    bin_c[0] = 1
                    out_cdf_0 = ECDF(out[0][out[1] == - 1])
                    out_cdf_0_eval = out_cdf_0(eval_points)
                if np.sum(out[1] == 1) > 0:
                    bin_c[1] = 1
                    out_cdf_1 = ECDF(out[0][out[1] == 1])
                    out_cdf_1_eval = out_cdf_1(eval_points)

                cnt = 0

                for c in bin_c:
                    if c == 1:
                        if cnt == 0:
                            in_ += ((out_cdf_0_eval > p_lims[0]) * (out_cdf_0_eval < p_lims[1]))
                        if cnt == 1:
                            in_ += ((out_cdf_1_eval > p_lims[0]) * (out_cdf_1_eval < p_lims[1]))
                    cnt += 0
                 
            else:
                
                out_cdf = ECDF(out[0][:, 0])
                out_cdf_eval = out_cdf(eval_points)
                in_ = ((out_cdf_eval > p_lims[0]) * (out_cdf_eval < p_lims[1]))
                
            out_ = np.invert(in_)
            gt_bnd = mcmc_dict['gt'][i, 1] * bf.weibull_cdf(eval_points, 
                                                            alpha = mcmc_dict['gt'][i, 4],
                                                            beta = mcmc_dict['gt'][i, 5])
            
            gt_bnd_pos = np.maximum(gt_bnd, 0)
            
            tmp_dist_in = np.zeros(n_posterior_subsample)
            tmp_dist_out = np.zeros(n_posterior_subsample)
            
            post_bnd_pos_tmp = np.zeros(len(eval_points))
            post_bnd_pos_mean_tmp_in = np.zeros(n_posterior_subsample)
            post_bnd_pos_mean_tmp_out = np.zeros(n_posterior_subsample)
            
            for j in range(n_posterior_subsample):
                idx = np.random.choice(n_posterior_samples)
                post_bnd_pos_tmp[:] = np.maximum(mcmc_dict['posterior_samples'][i, idx, 1] * bf.weibull_cdf(eval_points,
                                                                                                            alpha = mcmc_dict['posterior_samples'][i, idx , 4],
                                                                                                            beta = mcmc_dict['posterior_samples'][i, idx , 5]),
                                                 0)
                
                post_bnd_pos_mean_tmp_in[j] = np.mean(post_bnd_pos_tmp[in_] [(gt_bnd_pos[in_] > bnd_epsilon) * (post_bnd_pos_tmp[in_] > bnd_epsilon)] )
                post_bnd_pos_mean_tmp_out[j] = np.mean(post_bnd_pos_tmp[out_] [(gt_bnd_pos[out_] > bnd_epsilon) * (post_bnd_pos_tmp[out_] > bnd_epsilon)] )
                tmp_dist_in[j] = np.mean(  np.square( gt_bnd_pos[in_] - post_bnd_pos_tmp[in_] ) [(gt_bnd_pos[in_] > bnd_epsilon) * (post_bnd_pos_tmp[in_] > bnd_epsilon)] )
                tmp_dist_out[j] = np.mean(  np.square( gt_bnd_pos[out_] - post_bnd_pos_tmp[out_] ) [(gt_bnd_pos[out_] > bnd_epsilon) * (post_bnd_pos_tmp[out_] > bnd_epsilon)] )
            
            
            gt_bnd_pos_mean_in[i] = np.mean(gt_bnd_pos[in_][(gt_bnd_pos[in_] > bnd_epsilon)])
            gt_bnd_pos_mean_out[i] = np.mean(gt_bnd_pos[out_][(gt_bnd_pos[out_] > bnd_epsilon)])
            post_bnd_pos_mean_in[i] = np.mean(post_bnd_pos_mean_tmp_in)
            post_bnd_pos_mean_out[i] = np.mean(post_bnd_pos_mean_tmp_out)
            
            dist_in[i] = np.mean(tmp_dist_in)
            dist_out[i] = np.mean(tmp_dist_out)
            
    return dist_in, dist_out, gt_bnd_pos_mean_in, gt_bnd_pos_mean_out, post_bnd_pos_mean_in, post_bnd_pos_mean_out

# A of T statistics (considering a of t only for timerange that spans observed data)
def compute_boundary_rmse(mode = 'max_t_global', # max_t_global, max_t_local, quantile
                          boundary_fun = bf.weibull_cdf, # bf.angle etc.
                          parameters_estimated =  [], #mcmc_dict['means'][mcmc_dict['r_hats'] < r_hat_cutoff, :],
                          parameters_true = [], # mcmc_dict['gt'][mcmc_dict['r_hats'] < r_hat_cutoff, :],
                          data = [],
                          model = 'weibull_cdf',
                          max_t = 20,
                          n_probes = 1000):
    

    parameters_estimated_tup = tuple(map(tuple, parameters_estimated[:, 4:]))
    
    #print(parameters_estimated_tup)
    parameters_true_tup = tuple(map(tuple, parameters_true[:, 4:]))
    #t_probes = np.linspace(0, max_t, n_probes)
    bnd_est = np.zeros((len(parameters_estimated), n_probes))
    bnd_true = np.zeros((len(parameters_estimated), n_probes))
    
    # get bound estimates
    for i in range(len(parameters_estimated)):
        #print(parameters_estimated[i])
        max_t = np.max(data[i, :, 0])
        t_probes = np.linspace(0, max_t, n_probes)
        
        if model == 'weibull_cdf' or model == 'weibull_cdf2':
            bnd_est[i] = np.maximum(parameters_estimated[i, 1] * boundary_fun(*(t_probes, ) + parameters_estimated_tup[i]), 0)
            bnd_true[i] = np.maximum(parameters_true[i, 1] * boundary_fun(*(t_probes, ) + parameters_true_tup[i]), 0)
        if model == 'angle':
            bnd_est[i] = np.maximum(parameters_estimated[i, 1] + boundary_fun(*(t_probes, ) + parameters_estimated_tup[i]), 0)
            bnd_true[i] = np.maximum(parameters_true[i, 1] + boundary_fun(*(t_probes, ) + parameters_true_tup[i]), 0)
            #print(parameters_estimated[i, 1] * boundary_fun(*(t_probes, ) + parameters_estimated_tup[i]))
            #print(bnd_true[i])
        else:
            bnd_est[i] = parameters_estimated[i, 1]
            bnd_true[i] = parameters_estimated[i, 1]
            
#         if i % 100 == 0:
#             print(i)
    
    # compute rmse
    rmse_vec = np.zeros((len(parameters_estimated_tup)))
    dist_param_euclid = np.zeros((len(parameters_estimated_tup)))
    for i in range(len(parameters_estimated)):
        rmse_vec[i] = np.sqrt(np.sum(np.square(bnd_est[i] - bnd_true[i])) / n_probes)
        dist_param_euclid[i] = np.sqrt(np.sum(np.square(parameters_estimated[i] - parameters_true[i])))
    
    return rmse_vec, dist_param_euclid

# SUPPORT FUNCTIONS GRAPHS
def parameter_recovery_plot(ax_titles = ['v', 'a', 'w', 'ndt', 'angle'], 
                            title = 'Parameter Recovery: ABC-NN',
                            ground_truths = [0, 0, 0],
                            estimates = [0, 0, 0],
                            estimate_variances = [0, 0, 0],
                            r2_vec = [0, 0, 0],
                            cols = 3,
                            save = True,
                            model = '', 
                            machine = '',
                            method = 'cnn',
                            statistic = 'mean',
                            data_signature = '',
                            fileidentifier = '',
                            train_data_type = '',
                            plot_format = 'svg'): # color_param 'none' 
    
    matplotlib.rcParams['text.usetex'] = True
    #matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['svg.fonttype'] = 'none'
    
    grayscale_map = plt.get_cmap('gray')
    
    normalized_sds = np.zeros(estimates.shape)
    for i in range(estimates.shape[1]):
        normalized_sds[:, i] = (estimate_variances[:, i] - np.min(estimate_variances[:, i])) \
        / (np.max(estimate_variances[:, i]) - np.min(estimate_variances[:, i]))

    rows = int(np.ceil(len(ax_titles) / cols))
    sns.set(style = "white", 
            palette = "muted", 
            color_codes = True)
    
    print('n_rows: ', rows)
    print('n_cols: ', cols)
    
    fig, ax = plt.subplots(rows, 
                           cols, 
                           figsize = (12, 12), 
                           sharex = False, 
                           sharey = False)
    
    #fig.suptitle(title, fontsize = 24)
    sns.despine(right = True)

    for i in range(estimates.shape[1]):
        row_tmp = int(np.floor(i / cols))
        col_tmp = i - (cols * row_tmp)
        print('row: ', row_tmp)
        print('col: ', col_tmp)
        sns.regplot(ground_truths[:, i], estimates[:, i], 
                    marker =  '.',
                    fit_reg = False,
                    ax = ax[row_tmp, col_tmp],
                    scatter_kws = {'s': 120, 'alpha': 0.3, 'color': grayscale_map(normalized_sds[:, i]), 'edgecolor': 'face'})
        unity_coords = np.linspace(*ax[row_tmp, col_tmp].get_xlim())
        ax[row_tmp, col_tmp].plot(unity_coords, unity_coords, color = 'red')
        
        ax[row_tmp, col_tmp].text(0.6, 0.1, '$R^2$: ' + r2_vec[i], 
                                  transform = ax[row_tmp, col_tmp].transAxes, 
                                  fontsize = 22)
        ax[row_tmp, col_tmp].set_xlabel(ax_titles[i] + ' - ground truth', 
                                        fontsize = 20);
        ax[row_tmp, col_tmp].set_ylabel(ax_titles[i] + ' - posterior mean', 
                                        fontsize = 20);
        ax[row_tmp, col_tmp].tick_params(axis = "x", 
                                         labelsize = 18)

    for i in range(estimates.shape[1], rows * cols, 1):
        row_tmp = int(np.floor(i / cols))
        col_tmp = i - (cols * row_tmp)
        ax[row_tmp, col_tmp].axis('off')

    plt.setp(ax, yticks = [])
    
    if save == True:
        if machine == 'home':
            fig_dir = "/users/afengler/OneDrive/git_repos/nn_likelihoods/figures/" + method + "/parameter_recovery"
            if not os.path.isdir(fig_dir):
                os.mkdir(fig_dir)
        
        figure_name = 'parameter_recovery_plot_' + fileidentifier + '_' + statistic + '_'
        plt.subplots_adjust(top = 0.9)
        plt.subplots_adjust(hspace = 0.3, wspace = 0.3)
        if plot_format == 'png':
            plt.savefig(fig_dir + '/' + figure_name + model + data_signature + '_' + train_data_type + '.png', dpi = 300)
        if plot_format == 'svg':
            plt.savefig(fig_dir + '/' + figure_name + model + data_signature + '_' + train_data_type + '.svg', 
                        format = 'svg', 
                        transparent = True,
                        frameon = False)
        plt.close()
    return #plt.show(block = False)

# SUPPORT FUNCTIONS GRAPHS
def parameter_recovery_hist(ax_titles = ['v', 'a', 'w', 'ndt', 'angle'],
                            estimates = [0, 0, 0],
                            r2_vec = [0, 0, 0],
                            cols = 3,
                            save = True,
                            model = '',
                            machine = '',
                            posterior_stat = 'mean', # can be 'mean' or 'map'
                            data_signature = '',
                            train_data_type = '',
                            method = 'cnn',
                            plot_format = 'svg'): # color_param 'none' 
    
    
    matplotlib.rcParams['text.usetex'] = True
    #matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['svg.fonttype'] = 'none'
    
    rows = int(np.ceil(len(ax_titles) / cols))
    sns.set(style = "white", 
            palette = "muted", 
            color_codes = True)

    fig, ax = plt.subplots(rows, 
                           cols, 
                           figsize = (12, 12), 
                           sharex = False, 
                           sharey = False)
    
    fig.suptitle('Ground truth - Posterior ' + posterior_stat + ': ' + model.upper(), fontsize = 24)
    sns.despine(right = True)

    for i in range(estimates.shape[1]):
        row_tmp = int(np.floor(i / cols))
        col_tmp = i - (cols * row_tmp)

        sns.distplot(estimates[:, i], 
                     color = 'black',
                     bins = 50,
                     kde = False,
                     rug = False,
                     rug_kws = {'alpha': 0.2, 'color': 'grey'},
                     hist_kws = {'alpha': 1, 
                                 'range': (-0.5, 0.5), 
                                 'edgecolor': 'black',
                                 'histtype': 'step'},
                     ax = ax[row_tmp, col_tmp])
        
        ax[row_tmp, col_tmp].axvline(x = 0, linestyle = '--', color = 'red', label = 'ground truth') # ymin=0, ymax=1)
        ax[row_tmp, col_tmp].axvline(x = np.mean(estimates[:, i]), linestyle = '--', color = 'blue', label = 'mean offset')
        
        ax[row_tmp, col_tmp].set_xlabel(ax_titles[i], 
                                        fontsize = 16);
        ax[row_tmp, col_tmp].tick_params(axis = "x", 
                                         labelsize = 14);
        
        if row_tmp == 0 and col_tmp == 0:
            ax[row_tmp, col_tmp].legend(labels = ['ground_truth', 'mean_offset'], fontsize = 14)


    for i in range(estimates.shape[1], rows * cols, 1):
        row_tmp = int(np.floor(i / cols))
        col_tmp = i - (cols * row_tmp)
        ax[row_tmp, col_tmp].axis('off')

    plt.setp(ax, yticks = [])
    
    if save == True:
        if machine == 'home':
            fig_dir = '/users/afengler/OneDrive/git_repos/nn_likelihoods/figures/' + method + '/parameter_recovery'
            if not os.path.isdir(fig_dir):
                os.mkdir(fig_dir)

        figure_name = 'parameter_recovery_hist_'
        plt.subplots_adjust(top = 0.9)
        plt.subplots_adjust(hspace = 0.3, wspace = 0.3)
        
        if plot_format == 'svg':
            plt.savefig(fig_dir + '/' + figure_name + model + data_signature + '_' + train_data_type + '.svg',
                        format = 'svg', 
                        transparent = True,
                        frameon = False)
        if plot_format == 'png':
            plt.savefig(fig_dir + '/' + figure_name + model + data_signature + '_' + train_data_type + '.png',
                        dpi = 300)
        plt.close()
    return #plt.show(block=False)

def posterior_variance_plot(ax_titles = ['v', 'a', 'w', 'ndt', 'angle'], 
                            posterior_variances = [0, 0, 0],
                            cols = 3,
                            save = True,
                            data_signature = '',
                            train_data_type = '',
                            model = '',
                            method = 'cnn',
                            range_max = 0.4,
                            plot_format = 'svg'):
    
    matplotlib.rcParams['text.usetex'] = True
    #matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['svg.fonttype'] = 'none'

    rows = int(np.ceil(len(ax_titles) / cols))
    sns.set(style = "white", 
            palette = "muted", 
            color_codes = True)

    fig, ax = plt.subplots(rows, 
                           cols, 
                           figsize = (12, 12), 
                           sharex = False, 
                           sharey = False)
    
    fig.suptitle('Posterior Variance: ' + model.upper(), fontsize = 40)
    sns.despine(right = True)

    for i in range(posterior_variances.shape[1]):
        row_tmp = int(np.floor(i / cols))
        col_tmp = i - (cols * row_tmp)
        
        sns.distplot(posterior_variances[:, i], 
                     color = 'black',
                     bins = 50,
                     kde = False,
                     rug = True,
                     rug_kws = {'alpha': 0.2, 'color': 'black'},
                     hist_kws = {'alpha': 1, 
                                 'range': (0, range_max), 
                                 'edgecolor': 'black',
                                 'histtype:': 'step'},
                     ax = ax[row_tmp, col_tmp])
        
        ax[row_tmp, col_tmp].set_xlabel(ax_titles[i], 
                                        fontsize = 24);
        
        ax[row_tmp, col_tmp].tick_params(axis = "x", 
                                         labelsize = 24);
        

    for i in range(posterior_variances.shape[1], rows * cols, 1):
        row_tmp = int(np.floor(i / cols))
        col_tmp = i - (cols * row_tmp)
        ax[row_tmp, col_tmp].axis('off')

    plt.setp(ax, yticks = [])
    
    if save == True:
        if machine == 'home':
            fig_dir = "/users/afengler/OneDrive/git_repos/nn_likelihoods/figures/" + method + "/posterior_variance"
            if not os.path.isdir(fig_dir):
                os.mkdir(fig_dir)
        
        figure_name = 'posterior_variance_plot_'
        plt.subplots_adjust(top = 0.9)
        plt.subplots_adjust(hspace = 0.3, wspace = 0.3)
        
        if plot_format == 'png':
            plt.savefig(fig_dir + '/' + figure_name + model + data_signature + '_' + train_data_type + '.png', dpi = 300, )
        if plot_format == 'svg':
            plt.savefig(fig_dir + '/' + figure_name + model + data_signature + '_' + train_data_type + '.svg',
                        format = 'svg', 
                        transparent = True,
                        frameon = False)
        plt.close()
    return #plt.show(block = False)


def hdi_p_plot(ax_titles = ['v', 'a', 'w', 'ndt', 'angle'], 
               p_values = [0, 0, 0],
               cols = 3,
               save = True,
               model = '',
               data_signature = '',
               train_data_type = '',
               method = 'cnn',
               plot_format = 'svg'):
    
    matplotlib.rcParams['text.usetex'] = True
    #matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['svg.fonttype'] = 'none'

    rows = int(np.ceil(len(ax_titles) / cols))
    sns.set(style = "white", 
            palette = "muted", 
            color_codes = True)

    fig, ax = plt.subplots(rows, 
                           cols, 
                           figsize = (12, 12), 
                           sharex = False, 
                           sharey = False)
    
    fig.suptitle('Bayesian p value of ground truth: ' + model.upper(), fontsize = 24)
    sns.despine(right = True)

    for i in range(p_values.shape[1]):
        row_tmp = int(np.floor(i / cols))
        col_tmp = i - (cols * row_tmp)
        
        sns.distplot(p_values[:, i], 
                     bins = 20,
                     color = 'black',
                     kde = False,
                     rug = False,
                     rug_kws = {'alpha': 0.2, 'color': 'grey'},
                     hist_kws = {'alpha': 1, 
                                 'edgecolor': 'black',
                                 'histtype': 'step'},
                     ax = ax[row_tmp, col_tmp])
        
        ax[row_tmp, col_tmp].set_xlabel(ax_titles[i], 
                                        fontsize = 16);
        
        ax[row_tmp, col_tmp].tick_params(axis = "x", 
                                         labelsize = 14);

    for i in range(p_values.shape[1], rows * cols, 1):
        row_tmp = int(np.floor(i / cols))
        col_tmp = i - (cols * row_tmp)
        ax[row_tmp, col_tmp].axis('off')

    plt.setp(ax, yticks = [])
    
    if save == True:
        if machine == 'home':
            fig_dir = "/users/afengler/OneDrive/git_repos/nn_likelihoods/figures/" + method + "/hdi_p"
            if not os.path.isdir(fig_dir):
                os.mkdir(fig_dir)
                
        figure_name = 'hdi_p_plot_'
        plt.subplots_adjust(top = 0.9)
        plt.subplots_adjust(hspace = 0.3, wspace = 0.3)
        
        if plot_format == 'png':
            plt.savefig(fig_dir + '/' + figure_name + model + data_signature + '_' + train_data_type + '.png', dpi = 300, )
        if plot_format == 'svg':
            plt.savefig(fig_dir + '/' + figure_name + model + data_signature + '_' + train_data_type + '.svg',
                        format = 'svg', 
                        transparent = True,
                        frameon = False)
        
        plt.close()
    return #plt.show(block = False)

def sbc_plot(ax_titles = ['v', 'a', 'w', 'ndt', 'angle'], 
             ranks = [0, 0, 0],
             cols = 3,
             save = True,
             model = '',
             data_signature = '',
             train_data_type = '',
             method = 'cnn'):
    
    matplotlib.rcParams['text.usetex'] = True
    #matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['svg.fonttype'] = 'none'

    rows = int(np.ceil(len(ax_titles) / cols))
    sns.set(style = "white", 
            palette = "muted", 
            color_codes = True)

    fig, ax = plt.subplots(rows, 
                           cols, 
                           figsize = (12, 12), 
                           sharex = False, 
                           sharey = False)
    
    fig.suptitle('Bayesian p value of ground truth: ' + model.upper(), fontsize = 24)
    sns.despine(right = True)

    for i in range(ranks.shape[1]):
        row_tmp = int(np.floor(i / cols))
        col_tmp = i - (cols * row_tmp)
        
        sns.distplot(ranks[:, i], 
                     bins = np.arange(0, 101),
                     color = 'black',
                     kde = False,
                     rug = False,
                     rug_kws = {'alpha': 0.2, 'color': 'grey'},
                     hist_kws = {'alpha': 1, 'edgecolor': 'black'},
                     ax = ax[row_tmp, col_tmp])
        
        ax[row_tmp, col_tmp].set_xlabel(ax_titles[i], 
                                        fontsize = 16);
        
        ax[row_tmp, col_tmp].tick_params(axis = "x", 
                                         labelsize = 14);

    for i in range(ranks.shape[1], rows * cols, 1):
        row_tmp = int(np.floor(i / cols))
        col_tmp = i - (cols * row_tmp)
        ax[row_tmp, col_tmp].axis('off')

    plt.setp(ax, yticks = [])
    
    if save == True:
        if machine == 'home':
            fig_dir = "/users/afengler/OneDrive/git_repos/nn_likelihoods/figures/" + method + "/sbc"
            if not os.path.isdir(fig_dir):
                os.mkdir(fig_dir)
                
        figure_name = 'sbc_plot_'
        plt.subplots_adjust(top = 0.9)
        plt.subplots_adjust(hspace = 0.3, 
                            wspace = 0.3)
        
        if plot_format == 'png':
            plt.savefig(fig_dir + '/' + figure_name + model + data_signature + '_' + train_data_type + '.png', dpi = 300, )
        if plot_format == 'svg':
            plt.savefig(fig_dir + '/' + figure_name + model + data_signature + '_' + train_data_type + '.svg',
                        format = 'svg', 
                        transparent = True,
                        frameon = False)
        plt.close()
    return #plt.show(block = False)

def hdi_coverage_plot(ax_titles = [],
                      coverage_probabilities = [],
                      save = True,
                      model = '',
                      data_signature = '',
                      train_data_type = '',
                      method = 'cnn',
                      plot_format = 'svg'):
    
    matplotlib.rcParams['text.usetex'] = True
    #matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['svg.fonttype'] = 'none'
    
    plt.bar(ax_titles, 
            coverage_probabilities,
            color = 'black')
    plt.title( model.upper() + ': Ground truth in HDI?', 
              size = 20)
    plt.xticks(size = 20)
    plt.yticks(np.arange(0, 1, step = 0.2),
               size = 20)
    plt.ylabel('Prob. HDI covers', size = 20)
    
    if save == True:
        if machine == 'home':
            fig_dir = "/users/afengler/OneDrive/git_repos/nn_likelihoods/figures/" + method + "/hdi_coverage"
            if not os.path.isdir(fig_dir):
                os.mkdir(fig_dir)
        
        figure_name = 'hdi_coverage_plot_'
        if plot_format == 'png':
            plt.savefig(fig_dir + '/' + figure_name + model + data_signature + '_' + train_data_type + '.png', dpi = 300, )
        if plot_format == 'svg':
            plt.savefig(fig_dir + '/' + figure_name + model + data_signature + '_' + train_data_type + '.svg',
                        format = 'svg', 
                        transparent = True,
                        frameon = False)
        
        plt.close()
    return #plt.show(block = False)

# def a_of_t_histogram(mcmc_dict = None,
#                      model = 'None',
#                      save = True,
#                      data_signature = '',
#                      train_data_type = '',
#                      method = 'mlp'):
    
#     plt.hist(mcmc_dict['a_of_t_dist_out'][ ~np.isnan(mcmc_dict['a_of_t_dist_out'])], 
#              bins = 40, 
#              alpha = 1.0,
#              color = 'black', 
#              histtype = 'step',
#              edgecolor = 'black',
#              label = 'Out of Data')
#     plt.hist(mcmc_dict['a_of_t_dist_in'][ ~np.isnan(mcmc_dict['a_of_t_dist_in'])], 
#              bins = 40, 
#              alpha = 1.0, 
#              color = 'red',
#              histtype = 'step',
#              edgecolor = 'red',
#              label = 'At Data')
#     plt.title(model.upper() + ': Boundary RMSE', size = 20)
#     plt.legend()
    
#     if save == True:
#         if machine == 'home':
#             fig_dir = "/users/afengler/OneDrive/git_repos/nn_likelihoods/figures/" + method + "/a_of_t"
#             if not os.path.isdir(fig_dir):
#                 os.mkdir(fig_dir)
            
#             figure_name = 'a_of_t_plot_'
#             plt.savefig(fig_dir + '/' + figure_name + model + data_signature + '_' + train_data_type + '.png', dpi = 300)
#             plt.close()
#     return


def a_of_t_panel(mcmc_dict = None,
                 model = 'None',
                 save = True,
                 data_signature = '',
                 train_data_type = '',
                 method = 'mlp',
                 z_score_xy = (0.14, 30),
                 plot_format = 'svg'):
    
    matplotlib.rcParams['text.usetex'] = True
    #matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['svg.fonttype'] = 'none'

    # Basic plot hyperparameters
    fig, ax = plt.subplots(2, 
                           2, 
                           figsize = (10, 10), 
                           sharex = False, 
                           sharey = False)
    
    # Run regressions first 
    # Out of data
    print('Size of a of t gt out: ', mcmc_dict['a_of_t_gt_out'].shape)
    print('Size of post out: ', mcmc_dict['a_of_t_post_out'].shape)
    print('Size of gt in: ', mcmc_dict['a_of_t_gt_in'].shape)
    print('Size of post in: ', mcmc_dict['a_of_t_post_in'].shape)
    
    n_parameter_sets = mcmc_dict['a_of_t_gt_in'].shape[0]
    
    reg_out = LinearRegression().fit(np.asmatrix(mcmc_dict['a_of_t_gt_out']).T, np.asmatrix(mcmc_dict['a_of_t_post_out']).T)
    r_out = reg_out.score(np.asmatrix(mcmc_dict['a_of_t_gt_out']).T, np.asmatrix(mcmc_dict['a_of_t_post_out']).T)
    # At data
    reg_in = LinearRegression().fit(np.asmatrix(mcmc_dict['a_of_t_gt_in']).T, np.asmatrix(mcmc_dict['a_of_t_post_in']).T)
    r_in = reg_in.score(np.asmatrix(mcmc_dict['a_of_t_gt_in']).T, np.asmatrix(mcmc_dict['a_of_t_post_in']).T)
    
    # Run Z-test
    r_prime_out = 1/2 * np.log((1 + r_out) / (1 - r_out))
    r_prime_in = 1/2 * np.log((1 + r_in) / (1 - r_in))
    S = np.sqrt((1 / (n_parameter_sets - 3) + 1 / (n_parameter_sets - 3)))
    z = np.abs((r_prime_out - r_prime_in) / S)
    
    # Get Bootstrap R_squared differences
    print('Running bootstrapping part...')
    
    # fix number of bootstrap samples to take
    B = 10000
    r_diff = []
    
    for i in range(B):
        # Get bootstrap sample indices
        sample = np.random.choice(n_parameter_sets, 
                                  size = n_parameter_sets,
                                  replace = True)
        #max(sample)
        
        # Compute R_squared for the bootstrap indices
        r_out_tmp = reg_out.score(np.asmatrix(mcmc_dict['a_of_t_gt_out'][sample]).T, np.asmatrix(mcmc_dict['a_of_t_post_out'][sample]).T)
        r_in_tmp = reg_in.score(np.asmatrix(mcmc_dict['a_of_t_gt_in'][sample]).T, np.asmatrix(mcmc_dict['a_of_t_post_in'][sample]).T)
        
        r_diff.append(r_in_tmp - r_out_tmp)
        if i % 100 == 0:
            print(i)
            
    r_diff_cdf = ECDF(r_diff)
    
    # Regression part: At data
    ax[0, 0].scatter(mcmc_dict['a_of_t_gt_in'], 
                     mcmc_dict['a_of_t_post_in'], 
                     color = 'black', 
                     alpha = 0.5)
    ax[0, 0].set_title('Boundary Recovery: At Data', fontsize = 16)
    ax[0, 0].text(0.7, 0.1, 
                  '$R^2$: ' + str(round(r_in,2)), 
                  transform = ax[0, 0].transAxes, 
                  fontsize = 14)
    ax[0, 0].set_xlabel('True', fontsize = 14)
    ax[0, 0].set_ylabel('Recovered', fontsize = 14)
    ax[0, 0].tick_params(labelsize = 12)

    # Regression part: Out of Data
    ax[0, 1].scatter(mcmc_dict['a_of_t_gt_out'],
                     mcmc_dict['a_of_t_post_out'], 
                     color = 'black', 
                     alpha = 0.5)
    ax[0, 1].set_title('Boundary Recovery: Out of Data', fontsize = 16)
    ax[0, 1].text(0.7, 0.1,
                 '$R^2$: ' + str(round(r_out, 2)), 
                  transform = ax[0, 1].transAxes, 
                  fontsize = 14)
    ax[0, 1].set_xlabel('True', fontsize = 14)
    ax[0, 1].set_ylabel('Recovered', fontsize = 14)
    ax[0, 1].tick_params(labelsize = 12)

    # Correlation difference part
    ax[1, 0].hist(r_diff, density = True, bins = 50, histtype = 'step', color = 'black')
    ax[1, 0].axvline(x = np.linspace(-0.5, 0.5, 10000)[r_diff_cdf(np.linspace(-0.5, 0.5, 10000)) < 0.05][-1], 
                     color = 'red', 
                     linestyle = '-.')
    ax[1, 0].axvline(x = np.linspace(-0.5, 0.5, 10000)[r_diff_cdf(np.linspace(-0.5, 0.5, 10000)) > 0.95][0], 
                     color = 'red', 
                     linestyle = '-.')
    ax[1, 0].set_title('Bootstrap Correlation Difference', 
                       fontsize = 16)
    ax[1, 0].set_xlim(left = 0.0) #, 0.2)
    ax[1, 0].set_xlim(right = ax[1, 0].get_xlim()[1] * 2)
    
    mylims_x = ax[1, 0].get_xlim()
    mylims_y = ax[1, 0].get_ylim()
    ax[1, 0].text(mylims_x[0] + (2 / 3) * (mylims_x[1] - mylims_x[0]), 
                  mylims_y[0] + (1/2) * (mylims_y[1] - mylims_y[0]), 
                  'z-score: ' + str(round(z, 2)) ,
                  ha = 'center', 
                  fontsize = 14)
    ax[1, 0].tick_params(axis = 'x', 
                         labelrotation = -45) # to Rotate Xticks Label Text
    ax[1, 0].tick_params(labelsize = 12)

    # Average squared distance histograms
    ax[1, 1].set_title(r'$\frac{1}{T} \int_T(GT(t) - Recovered(t))^2 dt$', fontsize = 16)
    ax[1, 1].hist(mcmc_dict['a_of_t_dist_in'][~np.isnan(mcmc_dict['a_of_t_dist_in'])], 
                  bins = np.linspace(0, 0.5, 100), 
                  alpha = 0.2, 
                  color = 'red',
                  label = 'At Data', 
                  density = True)
    
    ax[1, 1].hist(mcmc_dict['a_of_t_dist_out'][~np.isnan(mcmc_dict['a_of_t_dist_out'])], 
                  bins = np.linspace(0, 0.5, 100), 
                  alpha = 0.2,
                  color = 'black', 
                  label = 'Out of Data', 
                  density = True)
    
    ax[1, 1].tick_params(labelsize = 12)
    ax[1, 1].legend(fontsize = 12)

    plt.tight_layout()
    
    if save == True:
        if machine == 'home':
            fig_dir = "/users/afengler/OneDrive/git_repos/nn_likelihoods/figures/" + method + "/a_of_t"
            if not os.path.isdir(fig_dir):
                os.mkdir(fig_dir)
            
            figure_name = 'a_of_t_plot_'
            if plot_format == 'png':
                plt.savefig(fig_dir + '/' + figure_name + model + data_signature + '_' + train_data_type + '.png', dpi = 300)
            if plot_format == 'svg':
                plt.savefig(fig_dir + '/' + figure_name + model + data_signature + '_' + train_data_type + '.svg',
                            format = 'svg', 
                            transparent = True,
                            frameon = False)
            plt.close()

    return

def posterior_predictive_plot(ax_titles = [], 
                              title = 'POSTERIOR PREDICTIVES: ',
                              x_labels = [],
                              posterior_samples = [],
                              ground_truths = [],
                              cols = 3,
                              model = 'angle',
                              data_signature = '',
                              train_data_type = '',
                              n_post_params = 100,
                              samples_by_param = 10,
                              save = False,
                              show = False,
                              machine = 'home',
                              method = 'cnn',
                              plot_format = 'svg'):
    
    matplotlib.rcParams['text.usetex'] = True
    #matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['svg.fonttype'] = 'none'
    
    rows = int(np.ceil(len(ax_titles) / cols))
    sns.set(style = "white", 
            palette = "muted", 
            color_codes = True,
            font_scale = 1)

    fig, ax = plt.subplots(rows, cols, 
                           figsize = (12, 12), 
                           sharex = False, 
                           sharey = False)
    fig.suptitle(title + model.upper(), fontsize = 24)
    sns.despine(right = True)

    tmp_simulator = partial(simulator, model = model, delta_t = 0.001, max_t = 20, bin_dim = None)
    
    for i in range(len(ax_titles)):
        row_tmp = int(np.floor(i / cols))
        col_tmp = i - (cols * row_tmp)
        
        post_tmp = np.zeros((n_post_params * samples_by_param, 2))
        idx = np.random.choice(posterior_samples.shape[1], size = n_post_params, replace = False)

        # Run Model simulations for posterior samples
        for j in range(n_post_params):
            out = tmp_simulator(theta = posterior_samples[i, idx[j], :], 
                                n_samples = samples_by_param)
          
            post_tmp[(samples_by_param * j):(samples_by_param * (j + 1)), :] = np.concatenate([out[0], out[1]], axis = 1)
        
        # Run Model simulations for true parameters
        out = tmp_simulator(theta = ground_truths[i, :],
                            n_samples = 20000)
  
        gt_tmp = np.concatenate([out[0], out[1]], axis = 1)
        print('passed through')
            
        sns.distplot(post_tmp[:, 0] * post_tmp[:, 1], 
                     bins = 50, 
                     kde = False, 
                     rug = False, 
                     hist_kws = {'alpha': 1, 'color': 'black', 'fill': 'black', 'density': 1, 'edgecolor': 'black'},
                     ax = ax[row_tmp, col_tmp]);
        sns.distplot(gt_tmp[:, 0] * gt_tmp[:, 1], 
                     hist_kws = {'alpha': 0.5, 'color': 'grey', 'fill': 'grey', 'density': 1, 'edgecolor': 'grey'}, 
                     bins = 50, 
                     kde = False, 
                     rug = False,
                     ax = ax[row_tmp, col_tmp])
        
        
        if row_tmp == 0 and col_tmp == 0:
            ax[row_tmp, col_tmp].legend(labels = [model, 'posterior'], 
                                        fontsize = 12, loc = 'upper right')
        
        if row_tmp == (rows - 1):
            ax[row_tmp, col_tmp].set_xlabel('RT', 
                                            fontsize = 14);
        
        if col_tmp == 0:
            ax[row_tmp, col_tmp].set_ylabel('Density', 
                                            fontsize = 14);
        
        ax[row_tmp, col_tmp].set_title(ax_titles[i],
                                       fontsize = 16)
        ax[row_tmp, col_tmp].tick_params(axis = 'y', size = 12)
        ax[row_tmp, col_tmp].tick_params(axis = 'x', size = 12)
        
    for i in range(len(ax_titles), rows * cols, 1):
        row_tmp = int(np.floor(i / cols))
        col_tmp = i - (cols * row_tmp)
        ax[row_tmp, col_tmp].axis('off')

    #plt.setp(ax, yticks = [])
    if save == True:
        if machine == 'home':
            fig_dir = "/users/afengler/OneDrive/git_repos/nn_likelihoods/figures/" + method + "/posterior_predictive"
            if not os.path.isdir(fig_dir):
                os.mkdir(fig_dir)
                
        figure_name = 'posterior_predictive_'
        #plt.tight_layout()
        plt.subplots_adjust(top = 0.9)
        plt.subplots_adjust(hspace = 0.3, wspace = 0.3)
        if plot_format == 'png':
            plt.savefig(fig_dir + '/' + figure_name + model + data_signature + '_' + train_data_type + '.png', dpi = 300) #  bbox_inches = 'tight')
        if plot_format == 'svg':
            plt.savefig(fig_dir + '/' + figure_name + model + data_signature + '_' + train_data_type + '.svg',
                            format = 'svg', 
                            transparent = True,
                            frameon = False)
        plt.close()
    if show:
        return #plt.show(block = False)

# Posterior predictive RACE / LCA (generally --> n_choices > 2)
def posterior_predictive_plot_race_lca(ax_titles = ['hiconf_go_stnhi.txt',
                                                    'hiconf_go_stnlo.txt',
                                                    'hiconf_go_stnmid.txt',
                                                    'loconf7_go_stnhi.txt',
                                                    'loconf7_go_stnlo.txt',
                                                    'loconf7_go_stnmid.txt'], 
                                        title = 'BG-STN: POSTERIOR PREDICTIVE',
                                        x_labels = [],
                                        posterior_samples = [],
                                        ground_truths = [],
                                        cols = 3,
                                        model = 'angle',
                                        data_signature = '',
                                        n_post_params = 500,
                                        samples_by_param = 10,
                                        show = False,
                                        save = False,
                                        max_t = 10,
                                        method = [],
                                        train_data_type = '',
                                        plot_format = 'svg'):
    
    matplotlib.rcParams['text.usetex'] = True
    #matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['svg.fonttype'] = 'none'

    rows = int(np.ceil(len(ax_titles) / cols))
    print('nrows: ', rows)
    sns.set(style = "white", 
            palette = "muted", 
            color_codes = True,
            font_scale = 2)

    fig, ax = plt.subplots(rows, cols, 
                           figsize = (20, 20), 
                           sharex = True, 
                           sharey = False)
    fig.suptitle(title, fontsize = 40)
    sns.despine(right = True)

    for i in range(len(ax_titles)):
        row_tmp = int(np.floor(i / cols))
        col_tmp = i - (cols * row_tmp)
        
        tmp = np.zeros((n_post_params * samples_by_param, 2))
        idx = np.random.choice(posterior_samples.shape[1], size = n_post_params, replace = False)

        # Run Model simulations
        for j in range(n_post_params):
            # Get posterior model simulations
            if model == 'race_model_3':
                out = cds.race_model(v = np.float32(posterior_samples[i, idx[j], 0:3]),
                                        a = np.float32(posterior_samples[i, idx[j], 3]),
                                        w = np.float32(posterior_samples[i, idx[j], 4:7]),
                                        ndt = np.float32(posterior_samples[i, idx[j], 7]),
                                        s = np.array([1., 1., 1., 1.],dtype=np.float32),
                                        delta_t = 0.001, 
                                        max_t = max_t,
                                        n_samples = samples_by_param,
                                        print_info = False,
                                        boundary_fun = bf.constant,
                                        boundary_multiplicative = True,
                                        boundary_params = {})
#             #tmp[(10 * j):(10 * (j + 1)), :] = np.concatenate([out[0], out[1], out[2], out[3]], axis = 1)
#             tmp[(10 * j):(10 * (j + 1)), :] = np.concatenate([out[0], out[1]], axis = 1)
#             #print('posterior samples gathered: ', j')
          
            if model == 'race_model_4':
                out = cds.race_model(v = np.float32(posterior_samples[i, idx[j], 0:4]),
                                        a = np.float32(posterior_samples[i, idx[j], 4]),
                                        w = np.float32(posterior_samples[i, idx[j], 5:9]),
                                        ndt = np.float32(posterior_samples[i, idx[j], 9]),
                                        s = np.array([1., 1., 1., 1.],dtype=np.float32),
                                        delta_t = 0.001, 
                                        max_t = max_t,
                                        n_samples = samples_by_param,
                                        print_info = False,
                                        boundary_fun = bf.constant,
                                        boundary_multiplicative = True,
                                        boundary_params = {})
#             #tmp[(10 * j):(10 * (j + 1)), :] = np.concatenate([out[0], out[1], out[2], out[3]], axis = 1)
#             tmp[(10 * j):(10 * (j + 1)), :] = np.concatenate([out[0], out[1]], axis = 1)
#             #print('posterior samples gathered: ', j')
            
            if model == 'lca_3':
                out = cds.lca(v = np.float32(posterior_samples[i, idx[j], 0:3]),
                             a = np.float32(posterior_samples[i, idx[j], 3]),
                             w = np.float32(posterior_samples[i, idx[j], 4:7]),
                             g = np.float32(posterior_samples[i, idx[j], 7]),
                             b = np.float32(posterior_samples[i, idx[j], 8]),
                             ndt = np.float32(posterior_samples[i, idx[j], 9]),
                             #s = np.array([1., 1., 1., 1.],dtype = np.float32),
                             s = 1.0,
                             delta_t = 0.001, 
                             max_t = max_t,
                             n_samples = samples_by_param,
                             print_info = False,
                             boundary_fun = bf.constant,
                             boundary_multiplicative = True,
                             boundary_params = {})
                
                if np.std(out[0]) == 0:
                    print(posterior_samples[i, idx[j], :])
                    print(np.mean(out[0]))

            
#             #tmp[(10 * j):(10 * (j + 1)), :] = np.concatenate([out[0], out[1], out[2], out[3]], axis = 1)
#             tmp[(10 * j):(10 * (j + 1)), :] = np.concatenate([out[0], out[1]], axis = 1)
#             #print('posterior samples gathered: ', j')
            
            if model == 'lca_4':
                out = cds.lca(v = np.float32(posterior_samples[i, idx[j], 0:4]),
                              a = np.float32(posterior_samples[i, idx[j], 4]),
                              w = np.float32(posterior_samples[i, idx[j], 5:9]),
                              g = np.float32(posterior_samples[i, idx[j], 9]),
                              b = np.float32(posterior_samples[i, idx[j], 10]),
                              ndt = np.float32(posterior_samples[i, idx[j], 11]),
                              #s = np.array([1., 1., 1., 1.], dtype = np.float32),
                              s = 1.0,
                              delta_t = 0.001,
                              max_t = max_t,
                              n_samples = samples_by_param,
                              print_info = False,
                              boundary_fun = bf.constant,
                              boundary_multiplicative = True,
                              boundary_params = {})
                if np.std(out[0]) == 0:
                    print(posterior_samples[i, idx[j], :])
                    print(np.mean(out[0]))

            
            
            #tmp[(10 * j):(10 * (j + 1)), :] = np.concatenate([out[0], out[1], out[2], out[3]], axis = 1)
            tmp[(10 * j):(10 * (j + 1)), :] = np.concatenate([out[0], out[1]], axis = 1)
            #print('posterior samples gathered: ', j')


        # Get model simulations from true data      
        if model == 'race_model_3':
            out = cds.race_model(v = np.float32(ground_truths[i, 0:3]),
                                 a = np.float32(ground_truths[i, 3]),
                                 w = np.float32(ground_truths[i, 4:7]),
                                 ndt = np.float32(ground_truths[i, 7]),
                                 s = np.array([1., 1., 1.], dtype = np.float32),
                                 delta_t = 0.001, 
                                 max_t = max_t,
                                 n_samples = 20000,
                                 print_info = False,
                                 boundary_fun = bf.constant,
                                 boundary_multiplicative = True,
                                 boundary_params = {})

        if model == 'race_model_4':
            out = cds.race_model(v = np.float32(ground_truths[i, 0:4]),
                                 a = np.float32(ground_truths[i, 4]),
                                 w = np.float32(ground_truths[i, 5:9]),
                                 ndt = np.float32(ground_truths[i, 9]),
                                 s = np.array([1., 1., 1., 1.],dtype = np.float32),
                                 delta_t = 0.001, 
                                 max_t = max_t,
                                 n_samples = 20000,
                                 print_info = False,
                                 boundary_fun = bf.constant,
                                 boundary_multiplicative = True,
                                 boundary_params = {})
            
        if model == 'lca_3':
            out = cds.lca(v = np.float32(ground_truths[i, 0:3]),
                          a = np.float32(ground_truths[i, 3]),
                          w = np.float32(ground_truths[i, 4:7]),
                          g = np.float32(ground_truths[i, 7]),
                          b = np.float32(ground_truths[i, 8]),
                          ndt = np.float32(ground_truths[i, 9]),
                          s = 1.0,
                          #s = np.array([1., 1., 1.],dtype = np.float32),
                          delta_t = 0.001, 
                          max_t = max_t,
                          n_samples = 20000,
                          print_info = False,
                          boundary_fun = bf.constant,
                          boundary_multiplicative = True,
                          boundary_params = {})
            
            if np.std(out[0]) == 0:
                print(ground_truths[i, :])
                print(np.mean(out[0]))
                        
            
        if model == 'lca_4':
            out = cds.lca(v = np.float32(ground_truths[i, 0:4]),
                           a = np.float32(ground_truths[i, 4]),
                           w = np.float32(ground_truths[i, 5:9]),
                           g = np.float32(ground_truths[i, 9]),
                           b = np.float32(ground_truths[i, 10]),
                           ndt = np.float32(ground_truths[i, 11]),
                           s = 1.0,
                           #s = np.array([1., 1., 1., 1.],dtype = np.float32),
                           delta_t = 0.001, 
                           max_t = max_t,
                           n_samples = 20000,
                           print_info = False,
                           boundary_fun = bf.constant,
                           boundary_multiplicative = True,
                           boundary_params = {})
            
            if np.std(out[0]) == 0:
                print(ground_truths[i, :])   
                print(np.mean(out[0]))
                  

        tmp_true = np.concatenate([out[0], out[1]], axis = 1)
        print('passed through')
        
        plot_colors = ['blue', 'red', 'orange', 'black', 'grey', 'green', 'brown']
        for c in range(6):
            if np.sum(tmp[:, 1] == c) > 0:
                choice_p_c = np.sum(tmp[:, 1] == c) / tmp.shape[0]
                counts, bins = np.histogram(tmp[tmp[:, 1] == c, 0],
                                            bins = np.linspace(0, max_t, 100),
                                            density = True)
                
                ax[row_tmp, col_tmp].hist(bins[:-1],
                                          bins,
                                          weights = choice_p_c * counts,
                                          histtype = 'step',
                                          alpha = 0.5,
                                          color = plot_colors[c],
                                          linestyle = 'dashed',
                                          edgecolor = plot_colors[c],
                                          label = 'Choice: ' + str(c) + ' Posterior')
                
                choice_p_c_true = np.sum(tmp_true[:, 1] == c) / tmp_true.shape[0]
                counts_2, bins = np.histogram(tmp_true[tmp_true[:, 1] == c, 0],
                                              bins = np.linspace(0, max_t, 100),
                                              density = True)
                
                ax[row_tmp, col_tmp].hist(bins[:-1],
                                          bins,
                                          weights = choice_p_c_true * counts_2,
                                          histtype = 'step',
                                          alpha = 0.5,
                                          color = plot_colors[c],
                                          linestyle = 'solid',
                                          edgecolor = plot_colors[c],
                                          label = 'Choice: ' + str(c) + ' Ground Truth')
                
               # ax[row_tmp, col_tmp].legend()

#                 sns.distplot(tmp[np.where(tmp[:, 1] == c)[0], 0], 
#                              bins = 50, 
#                              hist = False,
#                              kde = True, 
#                              rug = False, 
#                              hist_kws = {'alpha': 0.2, 'color': plot_colors[c], 'density': 1},
#                              kde_kws = {'color': plot_colors[c], 'label': 'Ground Truth'},
#                              ax = ax[row_tmp, col_tmp])
#                 sns.distplot(tmp_true[np.where(tmp_true[:, 1] == c)[0], 0], 
#                              bins = 50, 
#                              hist = False,
#                              kde = True, 
#                              rug = False, 
#                              hist_kws = {'alpha': 0.2, 'color': plot_colors[c], 'density': 1},
#                              kde_kws={'linestyle':'--', 'color': plot_colors[c], 'label': 'CNN'},
#                              ax = ax[row_tmp, col_tmp])

        if row_tmp == 0 and col_tmp == 0:
            ax[row_tmp, col_tmp].legend(fontsize = 12, loc = 'upper right')
        #    ax[row_tmp, col_tmp].legend(labels = ['Simulations', 'E-D CNN'], fontsize = 20)
        else:
            pass
            #ax[row_tmp, col_tmp].get_legend().remove()
            
        ax[row_tmp, col_tmp].set_xlabel('', 
                                        fontsize = 24);
        ax[row_tmp, col_tmp].set_ylabel('density', 
                                        fontsize = 24);
        ax[row_tmp, col_tmp].set_title(ax_titles[i],
                                       fontsize = 24)
        ax[row_tmp, col_tmp].tick_params(axis = 'y',
                                         size = 24)
        ax[row_tmp, col_tmp].tick_params(axis = 'x', 
                                         size = 24)
        
    for i in range(ground_truths.shape[0], rows * cols, 1):
        row_tmp = int(np.floor(i / cols))
        col_tmp = i - (cols * row_tmp)
        ax[row_tmp, col_tmp].axis('off')

  #plt.setp(ax, yticks = [])
    if save == True:
        if machine == 'home':
            fig_dir = "/users/afengler/OneDrive/git_repos/nn_likelihoods/figures/" + method + "/posterior_predictive"
            if not os.path.isdir(fig_dir):
                os.mkdir(fig_dir)
                
        figure_name = 'posterior_predictive_'
        #plt.tight_layout()
        plt.subplots_adjust(top = 0.9)
        plt.subplots_adjust(hspace = 0.3, wspace = 0.3)
        if plot_format == 'png':
            plt.savefig(fig_dir + '/' + figure_name + model + data_signature + '_' + train_data_type + '.png', dpi = 300) #  bbox_inches = 'tight')
        if plot_format == 'svg':
            plt.savefig(fig_dir + '/' + figure_name + model + data_signature + '_' + train_data_type + '.svg',
                            format = 'svg', 
                            transparent = True,
                            frameon = False)
        plt.close()
    if show:
        return #plt.show(block = False)
    return

# Plot bound
# Mean posterior predictives

def model_plot(posterior_samples = None,
               ground_truths = [],
               cols = 3,
               model = 'weibull_cdf',
               n_post_params = 500,
               n_plots = 4,
               samples_by_param = 10,
               max_t = 5,
               input_hddm_trace = False,
               datatype = 'single_subject', # 'hierarchical', 'single_subject', 'condition'
               show_model = True,
               show = False,
               save = False,
               machine = 'home',
               data_signature = '',
               train_data_type = '',
               method = 'cnn',
               plot_format = 'svg'):
    
    matplotlib.rcParams['text.usetex'] = True
    #matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['svg.fonttype'] = 'none'
    
    if 'weibull_cdf' in model:
        model = 'weibull_cdf'
    if 'angle' in model:
        model = 'angle'
    
    # Inputs are hddm_traces --> make plot ready
    if input_hddm_trace and posterior_samples is not None:
        if datatype == 'hierarchical':
            posterior_samples = _make_trace_plotready_hierarchical(posterior_samples, 
                                                                   model = model)
            n_plots = posterior_samples.shape[0]
#             print(posterior_samples)
            
        if datatype == 'single_subject':
            posterior_samples = _make_trace_plotready_single_subject(posterior_samples, 
                                                                     model = model)
        if datatype == 'condition':
            posterior_samples = _make_trace_plotready_condition(posterior_samples, 
                                                                model = model)
            n_plots = posterior_samples.shape[0]
            #print(posterior_samples)
            #n_plots = posterior_samples.shape[0]

    tmp_simulator = partial(simulator, 
                            model = model, 
                            delta_t = 0.001, 
                            max_t = 20, 
                            bin_dim = None)

    # Taking care of special case with 1 plot
    if n_plots == 1:
        ground_truths = np.expand_dims(ground_truths, 0)
        if posterior_samples is not None:
            posterior_samples = np.expand_dims(posterior_samples, 0)
            
    plot_titles = {'ddm': 'DDM', 
                   'angle': 'ANGLE',
                   'full_ddm': 'FULL DDM',
                   'weibull_cdf': 'WEIBULL',
                   'levy': 'LEVY',
                   'ornstein': 'ORNSTEIN UHLENBECK',
                   'ddm_sdv': 'DDM RANDOM SLOPE',
                  }
    
    title = 'Model Plot: '
    ax_titles = config[model]['params']

    rows = int(np.ceil(n_plots / cols))

    sns.set(style = "white", 
            palette = "muted", 
            color_codes = True,
            font_scale = 2)

    fig, ax = plt.subplots(rows, cols, 
                           figsize = (20, 20), 
                           sharex = False, 
                           sharey = False)
    
    my_suptitle = fig.suptitle(title + plot_titles[model], fontsize = 40)
    sns.despine(right = True)
    
    t_s = np.arange(0, max_t, 0.01)
    
    for i in range(n_plots):
        print('Making plot: ', i)
        row_tmp = int(np.floor(i / cols))
        col_tmp = i - (cols * row_tmp)
        
        if rows > 1 and cols > 1:
            ax[row_tmp, col_tmp].set_xlim(0, max_t)
            ax[row_tmp, col_tmp].set_ylim(-2, 2)
        elif (rows == 1 and cols > 1) or (rows > 1 and cols == 1):
            ax[i].set_xlim(0, max_t)
            ax[i].set_ylim(-2, 2)
        else:
            ax.set_xlim(0, max_t)
            ax.set_ylim(-2, 2)
        
        # Run simulations and add histograms
        # True params
        print('Running ground truth simulator: ')
        out = tmp_simulator(theta = ground_truths[i, :],
                            n_samples = 20000)
        
        print('Simulator finished')
        tmp_true = np.concatenate([out[0], out[1]], axis = 1)
        choice_p_up_true = np.sum(tmp_true[:, 1] == 1) / tmp_true.shape[0]
        
        if posterior_samples is not None:
            # Run Model simulations for posterior samples
            tmp_post = np.zeros((n_post_params * samples_by_param, 2))
            idx = np.random.choice(posterior_samples.shape[1], size = n_post_params, replace = False)
            
            print('Running posterior simulator: ')
            for j in range(n_post_params):
                
                out = tmp_simulator(theta = posterior_samples[i, idx[j], :],
                                   n_samples = samples_by_param)
                                
                tmp_post[(samples_by_param * j):(samples_by_param * (j + 1)), :] = np.concatenate([out[0], out[1]], axis = 1)
            print('Posterior simulator finished')
            
         #ax.set_ylim(-4, 2)
        if rows > 1 and cols > 1:
            ax_tmp = ax[row_tmp, col_tmp].twinx()
        elif (rows == 1 and cols > 1) or (rows > 1 and cols == 1):
            ax_tmp = ax[i].twinx()
        else:
            ax_tmp = ax.twinx()
        
        ax_tmp.set_ylim(-2, 2)
        ax_tmp.set_yticks([])
        
        if posterior_samples is not None:
            choice_p_up_post = np.sum(tmp_post[:, 1] == 1) / tmp_post.shape[0]



            counts, bins = np.histogram(tmp_post[tmp_post[:, 1] == 1, 0],
                                        bins = np.linspace(0, max_t, 100))

            counts_2, bins = np.histogram(tmp_post[tmp_post[:, 1] == 1, 0],
                                          bins = np.linspace(0, max_t, 100),
                                          density = True)
            
            if j == (n_post_params - 1) and row_tmp == 0 and col_tmp == 0:
                ax_tmp.hist(bins[:-1], 
                            bins, 
                            weights = choice_p_up_post * counts_2,
                            histtype = 'step',
                            alpha = 0.5, 
                            color = 'black',
                            edgecolor = 'black',
                            zorder = -1,
                            label = 'Posterior Predictive')
                
            else:
                ax_tmp.hist(bins[:-1], 
                            bins, 
                            weights = choice_p_up_post * counts_2,
                            histtype = 'step',
                            alpha = 0.5, 
                            color = 'black',
                            edgecolor = 'black',
                            zorder = -1)
                        
        counts, bins = np.histogram(tmp_true[tmp_true[:, 1] == 1, 0],
                                bins = np.linspace(0, max_t, 100))

        counts_2, bins = np.histogram(tmp_true[tmp_true[:, 1] == 1, 0],
                                      bins = np.linspace(0, max_t, 100),
                                      density = True)
        
        if row_tmp == 0 and col_tmp == 0:
            ax_tmp.hist(bins[:-1], 
                        bins, 
                        weights = choice_p_up_true * counts_2,
                        histtype = 'step',
                        alpha = 0.5, 
                        color = 'red',
                        edgecolor = 'red',
                        zorder = -1,
                        label = 'Ground Truth Data')
            ax_tmp.legend(loc = 'lower right')
        else:
            ax_tmp.hist(bins[:-1], 
                    bins, 
                    weights = choice_p_up_true * counts_2,
                    histtype = 'step',
                    alpha = 0.5, 
                    color = 'red',
                    edgecolor = 'red',
                    zorder = -1)
             
        #ax.invert_xaxis()
        if rows > 1 and cols > 1:
            ax_tmp = ax[row_tmp, col_tmp].twinx()
        elif (rows == 1 and cols > 1) or (rows > 1 and cols == 1):
            ax_tmp = ax[i].twinx()
        else:
            ax_tmp = ax.twinx()
            
        ax_tmp.set_ylim(2, -2)
        ax_tmp.set_yticks([])
        
        if posterior_samples is not None:
            counts, bins = np.histogram(tmp_post[tmp_post[:, 1] == -1, 0],
                            bins = np.linspace(0, max_t, 100))

            counts_2, bins = np.histogram(tmp_post[tmp_post[:, 1] == -1, 0],
                                          bins = np.linspace(0, max_t, 100),
                                          density = True)
            ax_tmp.hist(bins[:-1], 
                        bins, 
                        weights = (1 - choice_p_up_post) * counts_2,
                        histtype = 'step',
                        alpha = 0.5, 
                        color = 'black',
                        edgecolor = 'black',
                        zorder = -1)
        
        counts, bins = np.histogram(tmp_true[tmp_true[:, 1] == -1, 0],
                                    bins = np.linspace(0, max_t, 100))
    
        counts_2, bins = np.histogram(tmp_true[tmp_true[:, 1] == -1, 0],
                                      bins = np.linspace(0, max_t, 100),
                                      density = True)
        ax_tmp.hist(bins[:-1], 
                    bins, 
                    weights = (1 - choice_p_up_true) * counts_2,
                    histtype = 'step',
                    alpha = 0.5, 
                    color = 'red',
                    edgecolor = 'red',
                    zorder = -1)
        
        # Plot posterior samples of bounds and slopes (model)
        print('Making bounds')
        if show_model:
            if posterior_samples is not None:
                for j in range(n_post_params):
                    if model == 'weibull_cdf' or model == 'weibull_cdf2':
                        b = posterior_samples[i, idx[j], 1] * bf.weibull_cdf(t = t_s, 
                                                                             alpha = posterior_samples[i, idx[j], 4],
                                                                             beta = posterior_samples[i, idx[j], 5])
                    if model == 'angle' or model == 'angle2':
                        b = np.maximum(posterior_samples[i, idx[j], 1] + bf.angle(t = t_s, 
                                                                                  theta = posterior_samples[i, idx[j], 4]), 0)
                    if model == 'ddm':
                        b = posterior_samples[i, idx[j], 1] * np.ones(t_s.shape[0])


                    start_point_tmp = - posterior_samples[i, idx[j], 1] + \
                                      (2 * posterior_samples[i, idx[j], 1] * posterior_samples[i, idx[j], 2])

                    slope_tmp = posterior_samples[i, idx[j], 0]

                    if rows > 1 and cols > 1:
                        ax[row_tmp, col_tmp].plot(t_s + posterior_samples[i, idx[j], 3], b, 'black',
                                                  t_s + posterior_samples[i, idx[j], 3], - b, 'black', 
                                                  alpha = 0.05,
                                                  zorder = 1000)
                    elif (rows == 1 and cols > 1) or (rows > 1 and cols == 1):
                        ax[i].plot(t_s + posterior_samples[i, idx[j], 3], b, 'black',
                                   t_s + posterior_samples[i, idx[j], 3], - b, 'black', 
                                   alpha = 0.05,
                                   zorder = 1000)
                    else:
                        ax.plot(t_s + posterior_samples[i, idx[j], 3], b, 'black',
                                t_s + posterior_samples[i, idx[j], 3], - b, 'black', 
                                alpha = 0.05,
                                zorder = 1000)
                    

                    for m in range(len(t_s)):
                        if (start_point_tmp + (slope_tmp * t_s[m])) > b[m] or (start_point_tmp + (slope_tmp * t_s[m])) < -b[m]:
                            maxid = m
                            break
                        maxid = m

                    if rows > 1 and cols > 1:
                        ax[row_tmp, col_tmp].plot(t_s[:maxid] + posterior_samples[i, idx[j], 3],
                                                  start_point_tmp + slope_tmp * t_s[:maxid], 
                                                  'black', 
                                                  alpha = 0.05,
                                                  zorder = 1000)
                        if j == (n_post_params - 1):
                            ax[row_tmp, col_tmp].plot(t_s[:maxid] + posterior_samples[i, idx[j], 3],
                                                      start_point_tmp + slope_tmp * t_s[:maxid], 
                                                      'black', 
                                                      alpha = 0.05,
                                                      zorder = 1000,
                                                      label = 'Model Samples')
                    elif (rows == 1 and cols > 1) or (rows > 1 and cols == 1):
                        ax[i].plot(t_s[:maxid] + posterior_samples[i, idx[j], 3],
                                   start_point_tmp + slope_tmp * t_s[:maxid], 
                                   'black', 
                                   alpha = 0.05,
                                   zorder = 1000)
                        if j == (n_post_params - 1):
                            ax[i].plot(t_s[:maxid] + posterior_samples[i, idx[j], 3],
                                       start_point_tmp + slope_tmp * t_s[:maxid], 
                                       'black', 
                                       alpha = 0.05,
                                       zorder = 1000,
                                       label = 'Model Samples')

                    else:
                        ax.plot(t_s[:maxid] + posterior_samples[i, idx[j], 3],
                                start_point_tmp + slope_tmp * t_s[:maxid], 
                                'black', 
                                alpha = 0.05,
                                zorder = 1000)
                        if j ==(n_post_params - 1):
                            ax.plot(t_s[:maxid] + posterior_samples[i, idx[j], 3],
                                    start_point_tmp + slope_tmp * t_s[:maxid], 
                                    'black', 
                                    alpha = 0.05,
                                    zorder = 1000,
                                    label = 'Model Samples')
                            
                            
                    if rows > 1 and cols > 1:
                        ax[row_tmp, col_tmp].axvline(x = posterior_samples[i, idx[j], 3], 
                                                     ymin = - 2, 
                                                     ymax = 2, 
                                                     c = 'black', 
                                                     linestyle = '--',
                                                     alpha = 0.05)
                        
                    elif (rows == 1 and cols > 1) or (rows > 1 and cols == 1):
                        ax[i].axvline(x = posterior_samples[i, idx[j], 3],
                                                     ymin = - 2,
                                                     ymax = 2,
                                                     c = 'black',
                                                     linestyle = '--',
                                                     alpha = 0.05)
                    else:
                        ax.axvline(x = posterior_samples[i, idx[j], 3], 
                                   ymin = -2, 
                                   ymax = 2, 
                                   c = 'black', 
                                   linestyle = '--',
                                   alpha = 0.05)
                            
        # Plot ground_truths bounds
        if show_model:
            if model == 'weibull_cdf' or model == 'weibull_cdf2':
                b = ground_truths[i, 1] * bf.weibull_cdf(t = t_s,
                                                         alpha = ground_truths[i, 4],
                                                         beta = ground_truths[i, 5])

            if model == 'angle' or model == 'angle2':
                b = np.maximum(ground_truths[i, 1] + bf.angle(t = t_s, theta = ground_truths[i, 4]), 0)

            if model == 'ddm':
                b = ground_truths[i, 1] * np.ones(t_s.shape[0])

            start_point_tmp = - ground_truths[i, 1] + \
                              (2 * ground_truths[i, 1] * ground_truths[i, 2])
            slope_tmp = ground_truths[i, 0]

            if rows > 1 and cols > 1:
                if row_tmp == 0 and col_tmp == 0:
                    ax[row_tmp, col_tmp].plot(t_s + ground_truths[i, 3], b, 'red', 
                                              alpha = 1, 
                                              linewidth = 3, 
                                              zorder = 1000)
                    ax[row_tmp, col_tmp].plot(t_s + ground_truths[i, 3], -b, 'red', 
                                              alpha = 1,
                                              linewidth = 3,
                                              zorder = 1000, 
                                              label = 'Grund Truth Model')
                    ax[row_tmp, col_tmp].legend(loc = 'upper right')
                else:
                    ax[row_tmp, col_tmp].plot(t_s + ground_truths[i, 3], b, 'red', 
                              t_s + ground_truths[i, 3], -b, 'red', 
                              alpha = 1,
                              linewidth = 3,
                              zorder = 1000)
                    
            elif (rows == 1 and cols > 1) or (rows > 1 and cols == 1):
                if row_tmp == 0 and col_tmp == 0:
                    ax[i].plot(t_s + ground_truths[i, 3], b, 'red', 
                                              alpha = 1, 
                                              linewidth = 3, 
                                              zorder = 1000)
                    ax[i].plot(t_s + ground_truths[i, 3], -b, 'red', 
                                              alpha = 1,
                                              linewidth = 3,
                                              zorder = 1000, 
                                              label = 'Grund Truth Model')
                    ax[i].legend(loc = 'upper right')
                else:
                    ax[i].plot(t_s + ground_truths[i, 3], b, 'red', 
                              t_s + ground_truths[i, 3], -b, 'red', 
                              alpha = 1,
                              linewidth = 3,
                              zorder = 1000)
            else:
                ax.plot(t_s + ground_truths[i, 3], b, 'red', 
                        alpha = 1, 
                        linewidth = 3, 
                        zorder = 1000)
                ax.plot(t_s + ground_truths[i, 3], -b, 'red', 
                        alpha = 1,
                        linewidth = 3,
                        zorder = 1000,
                        label = 'Ground Truth Model')
                print('passed through legend part')
                print(row_tmp)
                print(col_tmp)
                ax.legend(loc = 'upper right')

            # Ground truth slope:
            # TODO: Can skip for weibull but can also make faster in general
            for m in range(len(t_s)):
                if (start_point_tmp + (slope_tmp * t_s[m])) > b[m] or (start_point_tmp + (slope_tmp * t_s[m])) < -b[m]:
                    maxid = m
                    break
                maxid = m

            # print('maxid', maxid)
            if rows > 1 and cols > 1:
                ax[row_tmp, col_tmp].plot(t_s[:maxid] + ground_truths[i, 3], 
                                          start_point_tmp + slope_tmp * t_s[:maxid], 
                                          'red', 
                                          alpha = 1, 
                                          linewidth = 3, 
                                          zorder = 1000)

                ax[row_tmp, col_tmp].set_zorder(ax_tmp.get_zorder() + 1)
                ax[row_tmp, col_tmp].patch.set_visible(False)
            elif (rows == 1 and cols > 1) or (rows > 1 and cols == 1):
                ax[i].plot(t_s[:maxid] + ground_truths[i, 3], 
                                          start_point_tmp + slope_tmp * t_s[:maxid], 
                                          'red', 
                                          alpha = 1, 
                                          linewidth = 3, 
                                          zorder = 1000)

                ax[i].set_zorder(ax_tmp.get_zorder() + 1)
                ax[i].patch.set_visible(False)
            else:
                ax.plot(t_s[:maxid] + ground_truths[i, 3], 
                        start_point_tmp + slope_tmp * t_s[:maxid], 
                        'red', 
                        alpha = 1, 
                        linewidth = 3, 
                        zorder = 1000)

                ax.set_zorder(ax_tmp.get_zorder() + 1)
                ax.patch.set_visible(False)
               
        # Set plot title
        title_tmp = ''
        for k in range(len(ax_titles)):
            title_tmp += ax_titles[k] + ': '
            title_tmp += str(round(ground_truths[i, k], 2)) + ', ' 

        if rows > 1 and cols > 1:
            if row_tmp == rows:
                ax[row_tmp, col_tmp].set_xlabel('rt', 
                                                 fontsize = 20);
            ax[row_tmp, col_tmp].set_ylabel('', 
                                            fontsize = 20);


            ax[row_tmp, col_tmp].set_title(title_tmp,
                                           fontsize = 24)
            ax[row_tmp, col_tmp].tick_params(axis = 'y', size = 20)
            ax[row_tmp, col_tmp].tick_params(axis = 'x', size = 20)

            # Some extra styling:
            ax[row_tmp, col_tmp].axvline(x = ground_truths[i, 3], ymin = -2, ymax = 2, c = 'red', linestyle = '--')
            ax[row_tmp, col_tmp].axhline(y = 0, xmin = 0, xmax = ground_truths[i, 3] / max_t, c = 'red',  linestyle = '--')
        
        elif (rows == 1 and cols > 1) or (rows > 1 and cols == 1):
            if row_tmp == rows:
                ax[i].set_xlabel('rt', 
                                                 fontsize = 20);
            ax[i].set_ylabel('', 
                                            fontsize = 20);


            ax[i].set_title(title_tmp,
                                           fontsize = 24)
            ax[i].tick_params(axis = 'y', size = 20)
            ax[i].tick_params(axis = 'x', size = 20)

            # Some extra styling:
            ax[i].axvline(x = ground_truths[i, 3], ymin = -2, ymax = 2, c = 'red', linestyle = '--')
            ax[i].axhline(y = 0, xmin = 0, xmax = ground_truths[i, 3] / max_t, c = 'red',  linestyle = '--')
        
        else:
            if row_tmp == rows:
                ax.set_xlabel('rt', 
                              fontsize = 20);
            ax.set_ylabel('', 
                          fontsize = 20);

            ax.set_title(title_tmp,
                         fontsize = 24)

            ax.tick_params(axis = 'y', size = 20)
            ax.tick_params(axis = 'x', size = 20)

            # Some extra styling:
            ax.axvline(x = ground_truths[i, 3], ymin = -2, ymax = 2, c = 'red', linestyle = '--')
            ax.axhline(y = 0, xmin = 0, xmax = ground_truths[i, 3] / max_t, c = 'red',  linestyle = '--')

    
    if rows > 1 and cols > 1:
        for i in range(n_plots, rows * cols, 1):
            row_tmp = int(np.floor(i / cols))
            col_tmp = i - (cols * row_tmp)
            ax[row_tmp, col_tmp].axis('off')

    plt.tight_layout(rect = [0, 0.03, 1, 0.9])
     
        #plt.setp(ax, yticks = [])
    if save == True:
        if machine == 'home':
            if show_model:
                fig_dir = "/users/afengler/OneDrive/git_repos/nn_likelihoods/figures/" + method + "/model_uncertainty_alt"
                figure_name = 'model_uncertainty_alt_'
            else:
                fig_dir = "/users/afengler/OneDrive/git_repos/nn_likelihoods/figures/" + method + "/posterior_predictive_alt"
                figure_name = 'posterior_predictive_alt_'
            
            
        if not os.path.isdir(fig_dir):
            os.mkdir(fig_dir) 
        #plt.tight_layout()
        plt.subplots_adjust(top = 0.9)
        plt.subplots_adjust(hspace = 0.3, wspace = 0.3)
        
        if plot_format == 'png':
            plt.savefig(fig_dir + '/' + figure_name + model + data_signature + '_' + train_data_type + '.png', dpi = 300) #  bbox_inches = 'tight')
        if plot_format == 'svg':
            plt.savefig(fig_dir + '/' + figure_name + model + data_signature + '_' + train_data_type + '.svg',
                            format = 'svg', 
                            transparent = True,
                            frameon = False)
        plt.close()
    if show:
        return plt.show(block = False)
    return



# Plot bound
# Mean posterior predictives
def boundary_posterior_plot(ax_titles = ['hi-hi', 'hi-lo', 'hi-mid', 'lo-hi', 'lo-mid'], 
                            title = 'Model uncertainty plot: ',
                            posterior_samples = [],
                            ground_truths = [],
                            cols = 3,
                            model = 'weibull_cdf',
                            data_signature = '',
                            train_data_type = '',
                            n_post_params = 500,
                            samples_by_param = 10,
                            max_t = 2,
                            show = True,
                            save = False,
                            machine = 'home',
                            method = 'cnn',
                            plot_format = 'svg'):
    
    matplotlib.rcParams['text.usetex'] = True
    #matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['svg.fonttype'] = 'none'
    
    tmp_simulator = partial(simulator, model = model, delta_t = 0.001, max_t = 20, bin_dim = None)
    
    rows = int(np.ceil(len(ax_titles) / cols))
    sub_idx = np.random.choice(posterior_samples.shape[1], size = n_post_params)
    posterior_samples = posterior_samples[:, sub_idx, :]
    
    sns.set(style = "white", 
            palette = "muted", 
            color_codes = True,
            font_scale = 2)

    fig, ax = plt.subplots(rows, cols, 
                           figsize = (20, 20), 
                           sharex = False, 
                           sharey = False)
    
    my_suptitle = fig.suptitle(title + model, fontsize = 40)
    sns.despine(right = True)
    
    t_s = np.arange(0, max_t, 0.01)
    for i in range(len(ax_titles)):
        row_tmp = int(np.floor(i / cols))
        col_tmp = i - (cols * row_tmp)
        
        idx = np.random.choice(posterior_samples.shape[1], size = n_post_params, replace = False)

        ax[row_tmp, col_tmp].set_xlim(0, max_t)
        ax[row_tmp, col_tmp].set_ylim(-2, 2)
        
        # Run simulations and add histograms
        # True params
        
        out = tmp_simulator(theta = ground_truths[i, :],
                            n_samples = 20000)
#         if model == 'angle' or model == 'angle2':
#             out = cds.ddm_flexbound(v = ground_truths[i, 0],
#                                     a = ground_truths[i, 1],
#                                     w = ground_truths[i, 2],
#                                     ndt = ground_truths[i, 3],
#                                     s = 1,
#                                     delta_t = 0.01, 
#                                     max_t = 20,
#                                     n_samples = 10000,
#                                     print_info = False,
#                                     boundary_fun = bf.angle,
#                                     boundary_multiplicative = False,
#                                     boundary_params = {'theta': ground_truths[i, 4]})
            
#         if model == 'weibull_cdf' or model == 'weibull_cdf2':
#             out = cds.ddm_flexbound(v = ground_truths[i, 0],
#                                     a = ground_truths[i, 1],
#                                     w = ground_truths[i, 2],
#                                     ndt = ground_truths[i, 3],
#                                     s = 1,
#                                     delta_t = 0.01, 
#                                     max_t = 20,
#                                     n_samples = 10000,
#                                     print_info = False,
#                                     boundary_fun = bf.weibull_cdf,
#                                     boundary_multiplicative = True,
#                                     boundary_params = {'alpha': ground_truths[i, 4],
#                                                        'beta': ground_truths[i, 5]})
        
#         if model == 'ddm':
#             out = cds.ddm_flexbound(v = ground_truths[i, 0],
#                                     a = ground_truths[i, 1],
#                                     w = ground_truths[i, 2],
#                                     ndt = ground_truths[i, 3],
#                                     s = 1,
#                                     delta_t = 0.01,
#                                     max_t = 20, 
#                                     n_samples = 10000,
#                                     print_info = False,
#                                     boundary_fun = bf.constant,
#                                     boundary_multiplicative = True,
#                                     boundary_params = {})
            
        
        tmp_true = np.concatenate([out[0], out[1]], axis = 1)
        choice_p_up_true = np.sum(tmp_true[:, 1] == 1) / tmp_true.shape[0]
        
        # Run Model simulations for posterior samples
        tmp_post = np.zeros((n_post_params*samples_by_param, 2))
        for j in range(n_post_params):
            out = tmp_simulator(theta = posterior_sampels[i, idx[j], :],
                                n_samples = samples_by_param)
            
#             if model == 'angle' or model == 'angle2':
#                 out = cds.ddm_flexbound(v = posterior_samples[i, idx[j], 0],
#                                         a = posterior_samples[i, idx[j], 1],
#                                         w = posterior_samples[i, idx[j], 2],
#                                         ndt = posterior_samples[i, idx[j], 3],
#                                         s = 1,
#                                         delta_t = 0.01, 
#                                         max_t = 20,
#                                         n_samples = samples_by_param,
#                                         print_info = False,
#                                         boundary_fun = bf.angle,
#                                         boundary_multiplicative = False,
#                                         boundary_params = {'theta': posterior_samples[i, idx[j], 4]})
            
#             if model == 'weibull_cdf' or model == 'weibull_cdf2':
#                 out = cds.ddm_flexbound(v = posterior_samples[i, idx[j], 0],
#                                         a = posterior_samples[i, idx[j], 1],
#                                         w = posterior_samples[i, idx[j], 2],
#                                         ndt = posterior_samples[i, idx[j], 3],
#                                         s = 1,
#                                         delta_t = 0.01, 
#                                         max_t = 20,
#                                         n_samples = samples_by_param,
#                                         print_info = False,
#                                         boundary_fun = bf.weibull_cdf,
#                                         boundary_multiplicative = True,
#                                         boundary_params = {'alpha': posterior_samples[i, idx[j], 4],
#                                                            'beta': posterior_samples[i, idx[j], 5]})
                
#             if model == 'ddm':
#                 out = cds.ddm_flexbound(v = posterior_samples[i, idx[j], 0],
#                                         a = posterior_samples[i, idx[j], 1],
#                                         w = posterior_samples[i, idx[j], 2],
#                                         ndt = posterior_samples[i, idx[j], 3],
#                                         s = 1,
#                                         delta_t = 0.01,
#                                         max_t = 20, 
#                                         n_samples = samples_by_param,
#                                         print_info = False,
#                                         boundary_fun = bf.constant,
#                                         boundary_multiplicative = True,
#                                         boundary_params = {})
            
            tmp_post[(10 * j):(10 * (j + 1)), :] = np.concatenate([out[0], out[1]], axis = 1)
        
        choice_p_up_post = np.sum(tmp_post[:, 1] == 1) / tmp_post.shape[0]
        
        #ax.set_ylim(-4, 2)
        ax_tmp = ax[row_tmp, col_tmp].twinx()
        ax_tmp.set_ylim(-2, 2)
        ax_tmp.set_yticks([])
        
        counts, bins = np.histogram(tmp_post[tmp_post[:, 1] == 1, 0],
                                    bins = np.linspace(0, 10, 100))
    
        counts_2, bins = np.histogram(tmp_post[tmp_post[:, 1] == 1, 0],
                                      bins = np.linspace(0, 10, 100),
                                      density = True)
        ax_tmp.hist(bins[:-1], 
                    bins, 
                    weights = choice_p_up_post * counts_2,
                    alpha = 0.2, 
                    color = 'black',
                    edgecolor = 'none',
                    zorder = -1)
        
        counts, bins = np.histogram(tmp_true[tmp_true[:, 1] == 1, 0],
                                bins = np.linspace(0, 10, 100))
    
        counts_2, bins = np.histogram(tmp_true[tmp_true[:, 1] == 1, 0],
                                      bins = np.linspace(0, 10, 100),
                                      density = True)
        ax_tmp.hist(bins[:-1], 
                    bins, 
                    weights = choice_p_up_true * counts_2,
                    alpha = 0.2, 
                    color = 'red',
                    edgecolor = 'none',
                    zorder = -1)
        
             
        #ax.invert_xaxis()
        ax_tmp = ax[row_tmp, col_tmp].twinx()
        ax_tmp.set_ylim(2, -2)
        ax_tmp.set_yticks([])
        
        counts, bins = np.histogram(tmp_post[tmp_post[:, 1] == -1, 0],
                        bins = np.linspace(0, 10, 100))
    
        counts_2, bins = np.histogram(tmp_post[tmp_post[:, 1] == -1, 0],
                                      bins = np.linspace(0, 10, 100),
                                      density = True)
        ax_tmp.hist(bins[:-1], 
                    bins, 
                    weights = (1 - choice_p_up_post) * counts_2,
                    alpha = 0.2, 
                    color = 'black',
                    edgecolor = 'none',
                    zorder = -1)
        
        counts, bins = np.histogram(tmp_true[tmp_true[:, 1] == -1, 0],
                                bins = np.linspace(0, 10, 100))
    
        counts_2, bins = np.histogram(tmp_true[tmp_true[:, 1] == -1, 0],
                                      bins = np.linspace(0, 10, 100),
                                      density = True)
        ax_tmp.hist(bins[:-1], 
                    bins, 
                    weights = (1 - choice_p_up_true) * counts_2,
                    alpha = 0.2, 
                    color = 'red',
                    edgecolor = 'none',
                    zorder = -1)
        
        # Plot posterior samples 
        for j in range(n_post_params):
            if model == 'weibull_cdf' or model == 'weibull_cdf2':
                b = posterior_samples[i, idx[j], 1] * bf.weibull_cdf(t = t_s, 
                                                                        alpha = posterior_samples[i, idx[j], 4],
                                                                        beta = posterior_samples[i, idx[j], 5])
            if model == 'angle' or model == 'angle2':
                b = np.maximum(posterior_samples[i, idx[j], 1] + bf.angle(t = t_s, 
                                                                             theta = posterior_samples[i, idx[j], 4]), 0)
            if model == 'ddm':
                b = posterior_samples[i, idx[j], 1] * np.ones(t_s.shape[0])
            
            
            start_point_tmp = - posterior_samples[i, idx[j], 1] + \
                              (2 * posterior_samples[i, idx[j], 1] * posterior_samples[i, idx[j], 2])
            
            slope_tmp = posterior_samples[i, idx[j], 0]
            
            ax[row_tmp, col_tmp].plot(t_s + posterior_samples[i, idx[j], 3], b, 'black',
                                      t_s + posterior_samples[i, idx[j], 3], - b, 'black', 
                                      alpha = 0.05,
                                      zorder = 1000)
            
            for m in range(len(t_s)):
                if (start_point_tmp + (slope_tmp * t_s[m])) > b[m] or (start_point_tmp + (slope_tmp * t_s[m])) < -b[m]:
                    maxid = m
                    break
                maxid = m
            
            ax[row_tmp, col_tmp].plot(t_s[:maxid] + posterior_samples[i, idx[j], 3],
                                      start_point_tmp + slope_tmp * t_s[:maxid], 
                                      'black', 
                                      alpha = 0.05,
                                      zorder = 1000)
            
        # Plot true ground_truths  
        if model == 'weibull_cdf' or model == 'weibull_cdf2':
            b = ground_truths[i, 1] * bf.weibull_cdf(t = t_s, 
                                                     alpha = ground_truths[i, 4],
                                                     beta = ground_truths[i, 5])
            
        if model == 'angle' or model == 'angle2':
            b = np.maximum(ground_truths[i, 1] + bf.angle(t = t_s, theta = ground_truths[i, 4]), 0)
        
        if model == 'ddm':
            b = ground_truths[i, 1] * np.ones(t_s.shape[0])

        start_point_tmp = - ground_truths[i, 1] + \
                          (2 * ground_truths[i, 1] * ground_truths[i, 2])
        slope_tmp = ground_truths[i, 0]

        ax[row_tmp, col_tmp].plot(t_s + ground_truths[i, 3], b, 'red', 
                                  t_s + ground_truths[i, 3], -b, 'red', 
                                  alpha = 1,
                                  linewidth = 3,
                                  zorder = 1000)
        
        for m in range(len(t_s)):
            if (start_point_tmp + (slope_tmp * t_s[m])) > b[m] or (start_point_tmp + (slope_tmp * t_s[m])) < -b[m]:
                maxid = m
                break
            maxid = m
        
        print('maxid', maxid)
        ax[row_tmp, col_tmp].plot(t_s[:maxid] + ground_truths[i, 3], 
                                  start_point_tmp + slope_tmp * t_s[:maxid], 
                                  'red', 
                                  alpha = 1, 
                                  linewidth = 3, 
                                  zorder = 1000)
        
        ax[row_tmp, col_tmp].set_zorder(ax_tmp.get_zorder() + 1)
        ax[row_tmp, col_tmp].patch.set_visible(False)
        print('passed through')
        
        #ax[row_tmp, col_tmp].legend(labels = [model, 'bg_stn'], fontsize = 20)
        if row_tmp == rows:
            ax[row_tmp, col_tmp].set_xlabel('rt', 
                                            fontsize = 20);
        ax[row_tmp, col_tmp].set_ylabel('', 
                                        fontsize = 20);
        ax[row_tmp, col_tmp].set_title(ax_titles[i],
                                       fontsize = 20)
        ax[row_tmp, col_tmp].tick_params(axis = 'y', size = 20)
        ax[row_tmp, col_tmp].tick_params(axis = 'x', size = 20)
        
    for i in range(len(ax_titles), rows * cols, 1):
        row_tmp = int(np.floor(i / cols))
        col_tmp = i - (cols * row_tmp)
        ax[row_tmp, col_tmp].axis('off')
    
    plt.tight_layout(rect = [0, 0.03, 1, 0.9])
    if save:
        if machine == 'home':
            fig_dir = "/users/afengler/OneDrive/git_repos/nn_likelihoods/figures/" + method + "/model_uncertainty"
            if not os.path.isdir(fig_dir):
                os.mkdir(fig_dir)
                
        figure_name = 'model_uncertainty_plot_'
        plt.savefig(fig_dir + '/' + figure_name + model + data_signature + '_' + train_data_type + '.png',
                    dpi = 150, 
                    transparent = False,
                    bbox_inches = 'tight',
                    bbox_extra_artists = [my_suptitle])
        plt.close()
    if show:
        return #plt.show(block = False)
    
# Posterior Pair Plot
def make_posterior_pair_grid_alt(posterior_samples = [],
                                 axes_limits = 'model', # 'model', 'samples'
                                 height = 10,
                                 aspect = 1,
                                 n_subsample = 1000,
                                 data_signature = None,
                                 gt_available = False,
                                 gt = [],
                                 model = None,
                                 machine = 'home',
                                 method = 'cnn',
                                 save = True,
                                 title_signature = None,
                                 train_data_type = '',
                                 plot_format = 'svg'):
    
    matplotlib.rcParams['text.usetex'] = True
    #matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['svg.fonttype'] = 'none'
    
    #sns.set(font_scale = 1.5)
    if 'weibull_cdf' in model:
        model = 'weibull_cdf'
    if 'angle' in model:
        model = 'angle'
    
    # some preprocessing (HDDM)
    #posterior_samples = posterior_samples.get_traces().copy()
    #posterior_samples['z'] = 1 / ( 1 + np.exp(- posterior_samples['z_trans']))
    #posterior_samples = posterior_samples.drop('z_trans', axis = 1)

    g = sns.PairGrid(posterior_samples.sample(n_subsample), 
                     height = height / len(list(posterior_samples.keys())),
                     aspect = 1,
                     diag_sharey = False)
    g = g.map_diag(sns.kdeplot, color = 'black', shade = False) # shade = True, 
    g = g.map_lower(sns.kdeplot, 
                    shade_lowest = False,
                    n_levels = 50,
                    shade = False,
                    cmap = 'Purples_d') # 'Greys'
    
    for i, j in zip(*np.triu_indices_from(g.axes, 1)):
        g.axes[i, j].set_visible(False)
    
    
    if axes_limits == 'model':
        xlabels,ylabels = [],[]

        for ax in g.axes[-1, :]:
            xlabel = ax.xaxis.get_label_text()
            #ax.set_xlabel(fontsize = 20)
            xlabels.append(xlabel)

        for ax in g.axes[:, 0]:
            ylabel = ax.yaxis.get_label_text()
            #ax.set_ylabel(fontsize = 20)
            ylabels.append(ylabel)

        for i in range(len(xlabels)):
            for j in range(len(ylabels)):
                g.axes[j,i].set_xlim(config[model]['param_bounds'][0][config[model]['params'].index(xlabels[i])], 
                                     config[model]['param_bounds'][1][config[model]['params'].index(xlabels[i])])
                g.axes[j,i].set_ylim(config[model]['param_bounds'][0][config[model]['params'].index(ylabels[j])], 
                                     config[model]['param_bounds'][1][config[model]['params'].index(ylabels[j])])

    
    for ax in g.axes.flat:
        plt.setp(ax.get_xticklabels(), rotation = 45)

    my_suptitle = g.fig.suptitle(model.upper() + ': Posterior Pair Plot', 
                                 y = 1.03, 
                                 fontsize = 24)
    
    # If ground truth is available add it in:
    if gt_available:
        for i in range(g.axes.shape[0]):
            for j in range(i + 1, g.axes.shape[0], 1):
                g.axes[j,i].plot(gt[config[model]['params'].index(xlabels[i])], 
                                 gt[config[model]['params'].index(ylabels[j])], 
                                 '.', 
                                 color = 'red',
                                 markersize = 7)

        for i in range(g.axes.shape[0]):
            g.axes[i,i].plot(gt[i],
                             g.axes[i,i].get_ylim()[0], 
                             '.', 
                             color = 'red',
                             markersize = 7)
              
    if save == True:
        if machine == 'home':
            fig_dir = "/users/afengler/OneDrive/git_repos/nn_likelihoods/figures/" + method + "/posterior_covariance_alt"
        if not os.path.isdir(fig_dir):
            os.mkdir(fig_dir)
    
    if plot_format == 'png':
        plt.savefig(fig_dir + '/' + 'cov_' + model + data_signature + '_' + train_data_type + '.png', 
                    dpi = 300, 
                    transparent = False,
                    bbox_inches = 'tight',
                    bbox_extra_artists = [my_suptitle])
    if plot_format == 'svg':
        plt.savefig(fig_dir + '/' + 'cov_' + model + data_signature + '_' + train_data_type + '.svg',
                    format = 'svg', 
                    transparent = True,
                    bbox_inches = 'tight',
                    bbox_extra_artists = [my_suptitle],
                    frameon = False)
    plt.close()

    # Show
    return #plt.show(block = False)

def caterpillar_plot(posterior_samples = [],
                     ground_truths = [],
                     model = None,
                     data_signature = '',
                     train_data_type = '',
                     machine = 'home',
                     method = 'cnn',
                     save = True,
                     plot_format = 'svg'):
    
    matplotlib.rcParams['text.usetex'] = True
    #matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['svg.fonttype'] = 'none'
    
    sns.set(style = "white", 
            palette = "muted", 
            color_codes = True,
            font_scale = 2)

    fig, ax = plt.subplots(1, 1, 
                           figsize = (10, 10), 
                           sharex = False, 
                           sharey = False)
    
    my_suptitle = fig.suptitle('Caterpillar plot: ' + model.upper(), 
                               fontsize = 40)
    
    sns.despine(right = True)
           
    ecdfs = {}
    plot_vals = {} # [0.01, 0.9], [0.01, 0.99], [mean]
    
    for k in config[model]['params']:
        ecdfs[k] = ECDF(posterior_samples[:, config[model]['params'].index(k)])
        tmp_sorted = sorted(posterior_samples[:, config[model]['params'].index(k)])
        _p01 =  tmp_sorted[np.sum(ecdfs[k](tmp_sorted) <= 0.01) - 1]
        _p99 = tmp_sorted[np.sum(ecdfs[k](tmp_sorted) <= 0.99) - 1]
        _p1 = tmp_sorted[np.sum(ecdfs[k](tmp_sorted) <= 0.1) - 1]
        _p9 = tmp_sorted[np.sum(ecdfs[k](tmp_sorted) <= 0.9) - 1]
        _pmean = trace[k].mean()
        plot_vals[k] = [[_p01, _p99], [_p1, _p9], _pmean]
        
    x = [plot_vals[k][2] for k in plot_vals.keys()]
    ax.scatter(x, plot_vals.keys(), c = 'black', marker = 's', alpha = 0)
    
    for k in plot_vals.keys():
        ax.plot(plot_vals[k][1], [k, k], c = 'grey', zorder = -1, linewidth = 5)
        ax.plot(plot_vals[k][0] , [k, k], c = 'black', zorder = -1)
        ax.scatter(ground_truths[config[model]['params'].index(k)], k,  c = 'red', marker = "|")
                                 
    if save == True:
        if machine == 'home':
            fig_dir = "/users/afengler/OneDrive/git_repos/nn_likelihoods/figures/" + method + "/caterpillar"
            if not os.path.isdir(fig_dir):
                os.mkdir(fig_dir)
        
        if plot_format == 'png':
            plt.savefig(fig_dir + '/' + 'caterpillar_' + model + data_signature + '_' + train_data_type + '.png', 
                        dpi = 300, 
                        transparent = False,
                        bbox_inches = 'tight',
                        bbox_extra_artists = [my_suptitle])
        if plot_format == 'svg':
            plt.savefig(fig_dir + '/' + 'caterpillar_' + model + data_signature + '_' + train_data_type + '.svg',
                            format = 'svg', 
                            transparent = True,
                            bbox_inches = 'tight',
                            bbox_extra_artists = [my_suptitle],
                            frameon = False)
        
        plt.close()
    # Show
    return #plt.show(block = False)

    return plt.show()

    
def make_posterior_pair_grid(posterior_samples = [],
                             height = 10,
                             aspect = 1,
                             n_subsample = 1000,
                             title = "Posterior Pairwise: ",
                             data_signature = '',
                             train_data_type = '',
                             title_signature= '',
                             gt_available = False,
                             gt = [],
                             save = True,
                             model = None,
                             machine = 'home',
                             method = 'cnn',
                             plot_format = 'svg'):
    
    matplotlib.rcParams['text.usetex'] = True
    #matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['svg.fonttype'] = 'none'
    
    g = sns.PairGrid(posterior_samples.sample(n_subsample), 
                     height = height / len(list(posterior_samples.keys())),
                     aspect = 1,
                     diag_sharey = False)
    g = g.map_diag(sns.kdeplot, color = 'black', shade = True) # shade = True, 
    g = g.map_lower(sns.kdeplot, 
                    shade_lowest = False,
                    n_levels = 30,
                    shade = True, 
                    cmap = 'Greys')
    g = g.map_lower(sns.regplot,
                    color = 'grey',
                    lowess = False,
                    x_ci = None,
                    marker = None,
                    scatter = False,
                    line_kws = {'alpha': 0.5})
    g = g.map_upper(sns.regplot,
                    scatter = True,
                    fit_reg = False,
                    color = 'grey',
                    scatter_kws = {'alpha': 0.01})

    #  g = g.map_upper(sns.scatterplot)
    #  g = g.map_upper(corrdot)
    #  g = g.map_upper(corrfunc)

    [plt.setp(ax.get_xticklabels(), rotation = 45) for ax in g.axes.flat]
    print(g.axes)
    my_suptitle = g.fig.suptitle(title + model.upper() + title_signature, y = 1.03, fontsize = 24)
    
    # If ground truth is available add it in:
    if gt_available:
        for i in range(g.axes.shape[0]):
            for j in range(i + 1, g.axes.shape[0], 1):
                g.axes[i,j].plot(gt[j], gt[i], '.', color = 'red')

        for i in range(g.axes.shape[0]):
            g.axes[i,i].plot(gt[i], g.axes[i,i].get_ylim()[0], '.', color = 'red')

    
    if save == True:
        if machine == 'home':
            fig_dir = "/users/afengler/OneDrive/git_repos/nn_likelihoods/figures/" + method + "/posterior_covariance"
            if not os.path.isdir(fig_dir):
                os.mkdir(fig_dir)
        
        if plot_format == 'png':
            plt.savefig(fig_dir + '/' + 'covalt_' + model + data_signature + '_' + train_data_type + '.png', 
                        dpi = 300, 
                        transparent = False,
                        bbox_inches = 'tight',
                        bbox_extra_artists = [my_suptitle])
        if plot_format == 'svg':
            plt.savefig(fig_dir + '/' + figure_name + model + data_signature + '_' + train_data_type + '.svg',
                        format = 'svg', 
                        transparent = True,
                        bbox_inches = 'tight',
                        bbox_extra_artists = [my_suptitle],
                        frameon = False)
        plt.close()
    # Show
    return #plt.show(block = False)


if __name__ == "__main__":
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--model",
                     type = str,
                     default = 'ddm')
    CLI.add_argument("--machine",
                     type = str,
                     default = 'home')
    CLI.add_argument("--method",
                     type = str,
                     default = "mlp") # "mlp", "cnn", "navarro", 'sbi'
    CLI.add_argument("--networkidx",
                     type = int,
                     default = -1)
    CLI.add_argument("--traindattype",
                     type = str,
                     default = "kde") # "kde", "analytic"
    CLI.add_argument("--n",
                     type = int,
                     default = 1024)
    CLI.add_argument("--analytic",
                     type = int,
                     default = 0)
    CLI.add_argument("--rhatcutoff",
                     type = float,
                     default = 1.1)
    CLI.add_argument("--npostpred",
                     type = int,
                     default = 9)
    CLI.add_argument("--npostpair",
                     type = int,
                     default = 9)
    CLI.add_argument("--plots",
                     nargs = "*",
                     type = str,
                     default = [])
    CLI.add_argument("--datafilter",
                     type = str,
                     default = 'choice_p')
    CLI.add_argument("--fileidentifier",
                     type = str,
                     default = 'elife_slice_')
    CLI.add_argument("--modelidentifier",
                     type = str,
                     default = 'None')
    
    args = CLI.parse_args()
    print(args)
    print(args.plots)
    
    model = args.model
    machine = args.machine
    method = args.method
    n = args.n
    rhatcutoff = args.rhatcutoff
    network_idx = args.networkidx
    traindattype = args.traindattype
    now = datetime.now().strftime("%m_%d_%Y")
    npostpred = args.npostpred
    npostpair = args.npostpair
    datafilter = args.datafilter
    
    if args.modelidentifier != 'None':
        modelidentifier = args.modelidentifier
    else:
        modelidentifier = ''
    
    if args.fileidentifier != 'None':
        fileidentifier = args.fileidentifier
    else:
        fileidentifier = ''
    
    matplotlib.rcParams['text.usetex'] = True
    #matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['svg.fonttype'] = 'none'

# Folder data 
# model = 'ddm_sdv'
# machine = 'home'
# method = 'mlp' # "mlp", "cnn", "navarro"
# n = 4096
# r_hat_cutoff = 1.1
# now = datetime.now().strftime("%m_%d_%Y")
# network_idx = 2
# train_dat = 'analytic'

    # Get model metadata
    info = pickle.load(open('kde_stats.pickle', 'rb'))
    ax_titles = info[model]['param_names'] + info[model]['boundary_param_names']
    print('ax_titles: ', ax_titles)
    print('length ax_titles: ', len(ax_titles))
    param_lims = info[model]['param_bounds_network'] + info[model]['boundary_param_bounds_network']
    
    if method != 'cnn' and method != 'sbi':
        if method != 'navarro':
            with open("model_paths.yaml") as tmp_file:
                if network_idx == -1:
                    network_path = yaml.load(tmp_file)[model + modelidentifier]
                    network_id = network_path[list(re.finditer('/', network_path))[-2].end():]
                else:
                    if traindattype == 'analytic':
                        network_path = yaml.load(tmp_file)[model + '_analytic' + '_batch'][network_idx]
                    else:
                        network_path = yaml.load(tmp_file)[model + '_batch'][network_idx]

                    network_id = network_path[list(re.finditer('/', network_path))[-2].end():]


            method_comparison_folder = '/Users/afengler/OneDrive/project_nn_likelihoods/data/' + traindattype + '/' + model + '/method_comparison/' + network_id + '/'

        else:
            method_comparison_folder = '/Users/afengler/OneDrive/project_nn_likelihoods/data/analytic/' + model + '/method_comparison/' + '/analytic/'

        # Get trained networks for model
        if fileidentifier == 'elife_slice_' or fileidentifier == 'elife_diffevo_':
            file_signature = 'post_samp_data_param_recov_unif_reps_1_n_' + str(n) + '_init_mle_'
        else:
            file_signature = 'post_samp_data_param_recov_unif_reps_1_n_' + str(n) + '_init_mle_1_'
        
        summary_file = method_comparison_folder +  'summary_' + fileidentifier + file_signature[:-1] + '.pickle'
    elif method == 'cnn':
        summary_file = '/users/afengler/OneDrive/project_nn_likelihoods/eLIFE_exps/summaries/IS_summary_' + fileidentifier + model + '_N_' + str(n) + '.pickle'
    elif method == 'sbi':
        if model == 'ddm':
            if n == 1000:
                summary_file = '/users/afengler/OneDrive/project_sbi_experiments/posterior_samples/'
            if n == 2000:
                summary_file = '/users/afengler/OneDrive/project_sbi_experiments/posterior_samples/'
        if model == 'angle':
            if n == 1000:
                summary_file = '/users/afengler/OneDrive/project_sbi_experiments/posterior_samples/collapsed_angle_bindim_64_abcmethod_SNPE_nsimruns_100000_nsamplesl_1000_nsamplesh_50000_nobs_1000.pickle'
            if n == 2000:
                summary_file = '/users/afengler/OneDrive/project_sbi_experiments/posterior_samples/collapsed_angle_bindim_64_abcmethod_SNPE_nsimruns_100000_nsamplesl_1000_nsamplesh_50000_nobs_2000.pickle'
        if model == 'full_ddm':
            if n == 1000:
                summary_file = '/users/afengler/OneDrive/project_sbi_experiments/posterior_samples/collapsed_full_ddm_bindim_64_abcmethod_SNPE_nsimruns_100000_nsamplesl_1000_nsamplesh_50000_nobs_1000.pickle'
            if n == 2000:
                summary_file = '/users/afengler/OneDrive/project_sbi_experiments/posterior_samples/collapsed_full_ddm_bindim_64_abcmethod_SNPE_nsimruns_100000_nsamplesl_1000_nsamplesh_50000_nobs_2000.pickle'
        if model == 'levy':
            if n == 1000:
                summary_file = '/users/afengler/OneDrive/project_sbi_experiments/posterior_samples/collapsed_levy_bindim_64_abcmethod_SNPE_nsimruns_100000_nsamplesl_1000_nsamplesh_50000_nobs_1000.pickle'
            if n == 2000:
                summary_file = '/users/afengler/OneDrive/project_sbi_experiments/posterior_samples/collapsed_levy_bindim_64_abcmethod_SNPE_nsimruns_100000_nsamplesl_1000_nsamplesh_50000_nobs_2000.pickle'
        if model == 'ornstein':
            if n == 1000:
                summary_file = '/users/afengler/OneDrive/project_sbi_experiments/posterior_samples/collapsed_ornstein_bindim_64_abcmethod_SNPE_nsimruns_100000_nsamplesl_1000_nsamplesh_50000_nobs_1000.pickle'
            if n == 2000:
                summary_file = '/users/afengler/OneDrive/project_sbi_experiments/posterior_samples/collapsed_ornstein_bindim_64_abcmethod_SNPE_nsimruns_100000_nsamplesl_1000_nsamplesh_50000_nobs_2000.pickle'
        if model == 'weibull_cdf':
            if n == 1000:
                summary_file = '/users/afengler/OneDrive/project_sbi_experiments/posterior_samples/collapsed_weibull_cdf_bindim_64_abcmethod_SNPE_nsimruns_100000_nsamplesl_1000_nsamplesh_50000_nobs_1000.pickle'
            if n == 2000:
                summary_file = '/users/afengler/OneDrive/project_sbi_experiments/posterior_samples/collapsed_weibull_cdf_bindim_64_abcmethod_SNPE_nsimruns_100000_nsamplesl_1000_nsamplesh_50000_nobs_2000.pickle'
        if model == 'ddm_sdv':
            if n == 1000:
                summary_file = '/users/afengler/OneDrive/project_sbi_experiments/posterior_samples/collapsed_ddm_sdv_bindim_64_abcmethod_SNPE_nsimruns_100000_nsamplesl_1000_nsamplesh_50000_nobs_1000.pickle'
            if n == 2000:
                summary_file = '/users/afengler/OneDrive/project_sbi_experiments/posterior_samples/collapsed_ddm_sdv_bindim_64_abcmethod_SNPE_nsimruns_100000_nsamplesl_1000_nsamplesh_50000_nobs_2000.pickle'

    # READ IN SUMMARY FILE
    mcmc_dict = pickle.load(open(summary_file, 'rb'))
    
    print(mcmc_dict.keys())
    print(mcmc_dict['posterior_samples'].shape)
    
    mcmc_dict = clean_mcmc_dict(mcmc_dict = mcmc_dict,
                                filter_ = datafilter,
                                method = method)
    
#     GENERATE PLOTS:
#     POSTERIOR VARIANCE PLOT MLP
    if "posterior_variance" in args.plots:
        print('Making Posterior Variance Plot...')
        posterior_variance_plot(ax_titles = ax_titles, 
                            posterior_variances = mcmc_dict['sds'],
                            cols = 2,
                            save = True,
                            data_signature = '_n_' + str(n) + '_' + now,
                            model = model,
                            method = mcmc_dict['method'],
                            train_data_type = traindattype)
    
    # HDI_COVERAGE PLOT
    if "hdi_coverage" in args.plots:
        print('Making HDI Coverage Plot...')
        hdi_coverage_plot(ax_titles = ax_titles,
                      model = model,
                      coverage_probabilities = mcmc_dict['p_covered_by_param'],
                      data_signature = '_n_' + str(n) + '_' + now,
                      save = True,
                      method = mcmc_dict['method'],
                      train_data_type = traindattype)
    
    # HDI P PLOT
    if "hdi_p" in args.plots:
        print('Making HDI P plot')
        hdi_p_plot(ax_titles = ax_titles,
               p_values = mcmc_dict['gt_cdf_score'],
               cols = 2,
               save = True,
               model = model,
               data_signature = '_n_' + str(n) + '_' + now,
               method = mcmc_dict['method'],
               train_data_type = traindattype)
        
    if "sbc" in args.plots:
        print('Making SBC plot')
        sbc_plot(ax_titles = ax_titles,
                 ranks = mcmc_dict['gt_ranks'],
                 cols = 2,
                 save = True,
                 model = model,
                 data_signature = '_n_' + str(n) + '_' + now,
                 method = mcmc_dict['method'],
                 train_data_type = traindattype)
        
    if 'a_of_t' in args.plots:
        print('Data preparation for a_of_t plot')
        mcmc_dict['a_of_t_dist_in'], mcmc_dict['a_of_t_dist_out'], mcmc_dict['a_of_t_gt_in'], mcmc_dict['a_of_t_gt_out'], mcmc_dict['a_of_t_post_in'], mcmc_dict['a_of_t_post_out'] = a_of_t_data_prep(mcmc_dict = mcmc_dict,
                                                                                                                                                                                                       model = model,
                                                                                                                                                                                                       n_eval_points = 1000,
                                                                                                                                                                                                       max_t = 20,
                                                                                                                                                                                                       p_lims = [0.05, 0.95],
                                                                                                                                                                                                       n_posterior_subsample = 10,
                                                                                                                                                                                                       split_ecdf = False,
                                                                                                                                                                                                       bnd_epsilon = 0.05)
        
        print('Making a_of_t plot')
        a_of_t_panel(mcmc_dict = mcmc_dict,
                     model = model,
                     save = True,
                     data_signature = '_n_' + str(n) + '_' + now,
                     train_data_type = traindattype,
                     method = mcmc_dict['method'])
                           
    if 'caterpillar' in args.plots:
                                 
        print('Making caterpillar plots ...')

        idx_vecs = [mcmc_dict['euc_dist_means_gt_sorted_id'][:npostpair], 
                    mcmc_dict['euc_dist_means_gt_sorted_id'][np.arange(int(len(mcmc_dict['euc_dist_means_gt_sorted_id']) / 2 - math.ceil(npostpair / 2)), int(len(mcmc_dict['euc_dist_means_gt_sorted_id']) / 2 + math.ceil(npostpair / 2)), 1)],
                    mcmc_dict['euc_dist_means_gt_sorted_id'][-npostpair:],
                    np.random.choice(mcmc_dict['gt'].shape[0], size = npostpair)]

        data_signatures = ['_n_' + str(n) + '_euc_dist_mean_low_idx_',
                          '_n_' + str(n) + '_euc_dist_mean_medium_idx_',
                          '_n_' + str(n) + '_euc_dist_mean_high_idx_',
                          '_n_' + str(n) + '_euc_dist_mean_random_idx_']

        title_signatures = [', ' + str(n) + ', Recovery Good',
                  ', ' + str(n) + ', Reocvery Medium',
                  ', ' + str(n) + ', Reocvery Bad',
                   ', ' + str(n) + ', Random ID']

        cnt = 0
        tot_cnt = 0
        for idx_vec in idx_vecs:
            tot_cnt = 0
            for idx in idx_vec:
                print('Making Caterpillar plots: ', title_signatures[cnt])
                caterpillar_plot(posterior_samples = mcmc_dict['posterior_samples'][idx, :, :],
                                 ground_truths = mcmc_dict['gt'][idx, :],
                                 model = model,
                                 data_signature = data_signatures[cnt] + '_' + str(tot_cnt) + '_idx_' + str(idx),
                                 train_data_type = traindattype,
                                 machine = 'home',
                                 method = mcmc_dict['method'],
                                 save = True)
                tot_cnt += 1
            cnt += 1

    # PARAMETER RECOVERY PLOTS: KDE MLP
    if "parameter_recovery_scatter" in args.plots:
        print('Making Parameter Recovery Plot ...')
        parameter_recovery_plot(ax_titles = ax_titles,
                                title = 'Parameter Recovery: ' + model,
                                ground_truths = mcmc_dict['gt'],
                                estimates = mcmc_dict['means'],
                                estimate_variances = mcmc_dict['sds'],
                                r2_vec = mcmc_dict['r2_means'],
                                cols = 3,
                                save = True,
                                machine = 'home',
                                data_signature = '_n_' + str(n) + '_' + now,
                                fileidentifier = fileidentifier,
                                method = mcmc_dict['method'],
                                model = model,
                                statistic = 'mean',
                                train_data_type = traindattype)
        
        parameter_recovery_plot(ax_titles = ax_titles,
                                title = 'Parameter Recovery: ' + model,
                                ground_truths = mcmc_dict['gt'],
                                estimates = mcmc_dict['maps'],
                                estimate_variances = mcmc_dict['sds'],
                                r2_vec = mcmc_dict['r2_maps'],
                                cols = 3,
                                save = True,
                                machine = 'home',
                                data_signature = '_n_' + str(n) + '_' + now,
                                fileidentifier = fileidentifier,
                                method = mcmc_dict['method'],
                                model = model,
                                statistic = 'maps',
                                train_data_type = traindattype)

    
    # Parameter recovery hist MLP
    if "parameter_recovery_hist" in args.plots:
        parameter_recovery_hist(ax_titles = ax_titles,
                                estimates = mcmc_dict['means'] - mcmc_dict['gt'], 
                                cols = 2,
                                save = True,
                                model = model,
                                machine = 'home',
                                posterior_stat = 'mean', # can be 'mean' or 'map'
                                data_signature =  '_n_' + str(n) + '_' + now,
                                method = mcmc_dict['method'],
                                train_data_type = traindattype)
    
    # EUC DIST MEANS GT SORTED ID: MLP
    # n_plots = 9
    # random_idx = np.random.choice(mcmc_dict['gt'][mcmc_dict['r_hats'] < r_hat_cutoff, 0].shape[0], size = n_plots)
    if "posterior_pair" in args.plots:
        print('Making Posterior Pair Plots...')
        idx_vecs = [mcmc_dict['euc_dist_means_gt_sorted_id'][:npostpair], 
                    mcmc_dict['euc_dist_means_gt_sorted_id'][np.arange(int(len(mcmc_dict['euc_dist_means_gt_sorted_id']) / 2 - math.ceil(npostpair / 2)), int(len(mcmc_dict['euc_dist_means_gt_sorted_id']) / 2 + math.ceil(npostpair / 2)), 1)],
                    mcmc_dict['euc_dist_means_gt_sorted_id'][-npostpair:],
                    np.random.choice(mcmc_dict['gt'].shape[0], size = npostpair)]

        data_signatures = ['_n_' + str(n) + '_euc_dist_mean_low_idx_',
                          '_n_' + str(n) + '_euc_dist_mean_medium_idx_',
                          '_n_' + str(n) + '_euc_dist_mean_high_idx_',
                          '_n_' + str(n) + '_euc_dist_mean_random_idx_']

        title_signatures = [', ' + str(n) + ', Recovery Good',
                  ', ' + str(n) + ', Reocvery Medium',
                  ', ' + str(n) + ', Reocvery Bad',
                   ', ' + str(n) + ', Random ID']

        cnt = 0
        tot_cnt = 0
        for idx_vec in idx_vecs:
            for idx in idx_vec:
                print('Making Posterior Pair Plot: ', tot_cnt)
                make_posterior_pair_grid(posterior_samples =  pd.DataFrame(mcmc_dict['posterior_samples'][idx, :, :],
                                                                           columns = ax_titles),
                                         gt =  mcmc_dict['gt'][idx, :],
                                         height = 8,
                                         aspect = 1,
                                         n_subsample = 2000,
                                         data_signature = data_signatures[cnt] + str(idx),
                                         title_signature = title_signatures[cnt],
                                         gt_available = True,
                                         save = True,
                                         model = model,
                                         method = mcmc_dict['method'],
                                         train_data_type = traindattype)
                tot_cnt += 1
            cnt += 1
    
    if "posterior_pair_alt" in args.plots:
        print('Making Posterior Pair Plots...')
        idx_vecs = [mcmc_dict['euc_dist_means_gt_sorted_id'][:npostpair], 
                    mcmc_dict['euc_dist_means_gt_sorted_id'][np.arange(int(len(mcmc_dict['euc_dist_means_gt_sorted_id']) / 2 - math.ceil(npostpair / 2)), int(len(mcmc_dict['euc_dist_means_gt_sorted_id']) / 2 + math.ceil(npostpair / 2)), 1)],
                    mcmc_dict['euc_dist_means_gt_sorted_id'][-npostpair:],
                    np.random.choice(mcmc_dict['gt'].shape[0], size = npostpair)]

        data_signatures = ['_n_' + str(n) + '_euc_dist_mean_low_idx_',
                           '_n_' + str(n) + '_euc_dist_mean_medium_idx_',
                           '_n_' + str(n) + '_euc_dist_mean_high_idx_',
                           '_n_' + str(n) + '_euc_dist_mean_random_idx_']

        title_signatures = [', ' + str(n) + ', Recovery Good',
                            ', ' + str(n) + ', Reocvery Medium',
                            ', ' + str(n) + ', Reocvery Bad',
                            ', ' + str(n) + ', Random ID']

        cnt = 0
        tot_cnt = 0
        for idx_vec in idx_vecs:
            tot_cnt = 0
            for idx in idx_vec:
                print('Making Posterior Pair Plot: ', title_signatures[cnt])
                make_posterior_pair_grid_alt(posterior_samples =  pd.DataFrame(mcmc_dict['posterior_samples'][idx, :, :],
                                                                               columns = ax_titles),
                                             gt =  mcmc_dict['gt'][idx, :],
                                             height = 8,
                                             aspect = 1,
                                             n_subsample = 2000,
                                             data_signature = data_signatures[cnt] + '_' + str(tot_cnt) + '_idx_' + str(idx),
                                             title_signature = title_signatures[cnt],
                                             gt_available = True,
                                             save = True,
                                             model = model,
                                             method = mcmc_dict['method'],
                                             train_data_type = traindattype)
                tot_cnt += 1
            cnt += 1
    
    # MODEL UNCERTAINTY PLOTS
    if "model_uncertainty" in args.plots:
        if model == 'angle' or model == 'weibull_cdf' or model == 'angle2' or model == 'ddm' or model == 'weibull_cdf2':
            print('Making Model Uncertainty Plots...')
            idx_vecs = [mcmc_dict['euc_dist_means_gt_sorted_id'][:npostpred], 
                        mcmc_dict['euc_dist_means_gt_sorted_id'][np.arange(int(len(mcmc_dict['euc_dist_means_gt_sorted_id']) / 2 - math.ceil(npostpred / 2)), int(len(mcmc_dict['euc_dist_means_gt_sorted_id']) / 2 + math.ceil(npostpred / 2)), 1)],
                        mcmc_dict['euc_dist_means_gt_sorted_id'][-npostpred:]]

            data_signatures = ['_n_' + str(n) + '_euc_dist_mean_low_',
                               '_n_' + str(n) + '_euc_dist_mean_medium_',
                               '_n_' + str(n) + '_euc_dist_mean_high_',]

            cnt = 0
            for idx_vec in idx_vecs:
                print('Making Model Uncertainty Plots... sets: ', cnt)
                boundary_posterior_plot(ax_titles = [str(i) for i in idx_vec], 
                                        title = 'Model Uncertainty: ',
                                        posterior_samples = mcmc_dict['posterior_samples'][idx_vec, :, :], # dat_total[1][bottom_idx, 5000:, :],
                                        ground_truths = mcmc_dict['gt'][idx_vec, :], #dat_total[0][bottom_idx, :],
                                        cols = 3,
                                        model = model, # 'weibull_cdf',
                                        data_signature = data_signatures[cnt],
                                        n_post_params = 2000,
                                        samples_by_param = 10,
                                        max_t = 10,
                                        show = True,
                                        save = True,
                                        method = mcmc_dict['method'],
                                        train_data_type = traindattype)
                cnt += 1
            
    
    # POSTERIOR PREDICTIVE PLOTS
    if "posterior_predictive" in args.plots:
        print('Making Posterior Predictive Plots...')
        idx_vecs = [mcmc_dict['euc_dist_means_gt_sorted_id'][:npostpred], 
                    mcmc_dict['euc_dist_means_gt_sorted_id'][np.arange(int(len(mcmc_dict['euc_dist_means_gt_sorted_id']) / 2 - math.ceil(npostpred / 2)), int(len(mcmc_dict['euc_dist_means_gt_sorted_id']) / 2 + math.ceil(npostpred / 2)), 1)],
                    mcmc_dict['euc_dist_means_gt_sorted_id'][-npostpred:]]

        data_signatures = ['_n_' + str(n) + '_euc_dist_mean_low_',
                           '_n_' + str(n) + '_euc_dist_mean_medium_',
                           '_n_' + str(n) + '_euc_dist_mean_high_',]

        cnt = 0
        for idx_vec in idx_vecs:
            print('Making Posterior Predictive Plots... set: ', cnt)
            if 'race' in model or 'lca' in model:
                if cnt != 0 or 'race' in model:
                    posterior_predictive_plot_race_lca(ax_titles = [str(i) for i in idx_vec], 
                                                    title = 'Posterior Predictive',
                                                    x_labels = [],
                                                    posterior_samples = mcmc_dict['posterior_samples'][idx_vec, :, :],
                                                    ground_truths = mcmc_dict['gt'][idx_vec, :],
                                                    cols = 3,
                                                    model = model,
                                                    data_signature = data_signatures[cnt],
                                                    n_post_params = 2000,
                                                    samples_by_param = 10,
                                                    show = True,
                                                    save = True, 
                                                    method = mcmc_dict['method'],
                                                    train_data_type = traindattype)
                                                                
            else:
                posterior_predictive_plot(ax_titles =[str(i) for i in idx_vec],
                                          title = 'Posterior Predictive: ',
                                          posterior_samples = mcmc_dict['posterior_samples'][idx_vec, :, :],
                                          ground_truths =  mcmc_dict['gt'][idx_vec, :],
                                          cols = 3,
                                          model = model,
                                          data_signature = data_signatures[cnt],
                                          n_post_params = 2000,
                                          samples_by_param = 10,
                                          show = True,
                                          save = True,
                                          method = mcmc_dict['method'],
                                          train_data_type = traindattype)
            cnt += 1                             
                                            
                                 
    # POSTERIOR PREDICTIVE PLOTS
    if "posterior_predictive_alt" in args.plots:
        print('Making Posterior Predictive Plots...')
        idx_vecs = [mcmc_dict['euc_dist_means_gt_sorted_id'][:npostpred], 
                    mcmc_dict['euc_dist_means_gt_sorted_id'][np.arange(int(len(mcmc_dict['euc_dist_means_gt_sorted_id']) / 2 - math.ceil(npostpred / 2)), int(len(mcmc_dict['euc_dist_means_gt_sorted_id']) / 2 + math.ceil(npostpred / 2)), 1)],
                    mcmc_dict['euc_dist_means_gt_sorted_id'][-npostpred:]]

        data_signatures = ['_n_' + str(n) + '_euc_dist_mean_low_',
                           '_n_' + str(n) + '_euc_dist_mean_medium_',
                           '_n_' + str(n) + '_euc_dist_mean_high_',]

        cnt = 0
        for idx_vec in idx_vecs:
            print('Making Posterior Predictive Plots... set: ', cnt)
            if 'race' in model or 'lca' in model:
                if cnt != 0 or 'race' in model:
                    posterior_predictive_plot_race_lca(ax_titles = [str(i) for i in idx_vec], 
                                                    title = 'Posterior Predictive',
                                                    x_labels = [],
                                                    posterior_samples = mcmc_dict['posterior_samples'][idx_vec, :, :],
                                                    ground_truths = mcmc_dict['gt'][idx_vec, :],
                                                    cols = 3,
                                                    model = model,
                                                    data_signature = data_signatures[cnt],
                                                    n_post_params = 2000,
                                                    samples_by_param = 10,
                                                    show = True,
                                                    save = True, 
                                                    method = mcmc_dict['method'],
                                                    train_data_type = traindattype)
                                                                
            else:
                model_plot(posterior_samples = mcmc_dict['posterior_samples'][idx_vec, :, :],
                          ground_truths = mcmc_dict['gt'][idx_vec, :, :],
                          cols = 3,
                          model = model,
                          n_post_params = 2000,
                          n_plots = npostpred,
                          samples_by_param = 10,
                          max_t = 5,
                          input_hddm_trace = False,
                          #datatype = 'single_subject', # 'hierarchical', 'single_subject', 'condition'
                          show_model = False,
                          save = True,
                          machine = 'home',
                          data_signature = data_signatures[cnt],
                          train_data_type = traindattype,
                          method = mcmc_dict['method'])
            cnt += 1  
                                 
                                 
    # MODEL UNCERTAINTY PLOTS
    if "model_uncertainty_alt" in args.plots:
        if model == 'angle' or model == 'weibull_cdf' or model == 'angle2' or model == 'ddm' or model == 'weibull_cdf2':
            print('Making Model Uncertainty Plots...')
            idx_vecs = [mcmc_dict['euc_dist_means_gt_sorted_id'][:npostpred], 
                        mcmc_dict['euc_dist_means_gt_sorted_id'][np.arange(int(len(mcmc_dict['euc_dist_means_gt_sorted_id']) / 2 - math.ceil(npostpred / 2)), int(len(mcmc_dict['euc_dist_means_gt_sorted_id']) / 2 + math.ceil(npostpred / 2)), 1)],
                        mcmc_dict['euc_dist_means_gt_sorted_id'][-npostpred:]]

            data_signatures = ['_n_' + str(n) + '_euc_dist_mean_low_',
                               '_n_' + str(n) + '_euc_dist_mean_medium_',
                               '_n_' + str(n) + '_euc_dist_mean_high_',]

            cnt = 0
            for idx_vec in idx_vecs:
                print('Making Model Uncertainty Plots... sets: ', cnt)
                                                 
                model_plot(posterior_samples = mcmc_dict['posterior_samples'][idx_vec, :, :],
                          ground_truths = mcmc_dict['gt'][idx_vec, :],
                          cols = 3,
                          model = model,
                          n_post_params = 1000,
                          n_plots = npostpred,
                          samples_by_param = 10,
                          max_t = 5,
                          input_hddm_trace = False,
                          #datatype = 'single_subject', # 'hierarchical', 'single_subject', 'condition'
                          show_model = True,
                          show = False,
                          save = True,
                          machine = 'home',
                          data_signature = data_signatures[cnt],
                          train_data_type = traindattype,
                          method = mcmc_dict['method'])

                cnt += 1