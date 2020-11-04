import argparse
import tensorflow as tf
from tensorflow import keras
import numpy as np
import yaml
import pandas as pd
from itertools import product
import pickle
import uuid
import seaborn as sns
import os
import cddm_data_simulation as cds
import cdwiener as cdw
import boundary_functions as bf
import kde_training_utilities as kde_utils
import kde_class as kdec
import ckeras_to_numpy as ktnp
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from matplotlib import cm

# INITIALIZATIONS -------------------------------------------------------------
def kde_vs_mlp_likelihoods(ax_titles = [], 
                           title = 'Likelihoods KDE - MLP',
                           network_dir = '',
                           x_labels = [],
                           parameter_matrix = [],
                           cols = 3,
                           model = 'angle',
                           data_signature = '',
                           n_samples = 10,
                           nreps = 10,
                           save = True,
                           show = False,
                           machine = 'home',
                           method = 'mlp',
                           traindatanalytic = 0,
                           plot_format = 'svg'):
    
    mpl.rcParams['text.usetex'] = True
    #matplotlib.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['svg.fonttype'] = 'none'
    
    # Initialize rows and graph parameters
    rows = int(np.ceil(len(ax_titles) / cols))
    sns.set(style = "white", 
            palette = "muted", 
            color_codes = True,
            font_scale = 2)

    fig, ax = plt.subplots(rows, cols, 
                           figsize = (10, 10), 
                           sharex = True, 
                           sharey = False)
    
    fig.suptitle(title + ': ' + model.upper().replace('_', '-'), fontsize = 40)
    sns.despine(right = True)
    
    # Data template
    plot_data = np.zeros((4000, 2))
    plot_data[:, 0] = np.concatenate(([i * 0.0025 for i in range(2000, 0, -1)], [i * 0.0025 for i in range(1, 2001, 1)]))
    plot_data[:, 1] = np.concatenate((np.repeat(-1, 2000), np.repeat(1, 2000)))
    
    # Load Keras model and initialize batch container
    keras_model = keras.models.load_model(network_dir + 'model_final.h5')
    keras_input_batch = np.zeros((4000, parameter_matrix.shape[1] + 2))
    keras_input_batch[:, parameter_matrix.shape[1]:] = plot_data

    for i in range(len(ax_titles)):
        
        print('Making Plot: ', i)
        
        row_tmp = int(np.floor(i / cols))
        col_tmp = i - (cols * row_tmp)
        
        # Get predictions from keras model
        keras_input_batch[:, :parameter_matrix.shape[1]] = parameter_matrix[i, :]
        ll_out_keras = keras_model.predict(keras_input_batch, 
                                           batch_size = 100)
        
        # Get prediction from navarro if traindatanalytic = 1
        if traindatanalytic:
            ll_out_gt = cdw.batch_fptd(plot_data[:, 0] * plot_data[:, 1], 
                                   v = parameter_matrix[i, 0],
                                   a = parameter_matrix[i, 1],
                                   w = parameter_matrix[i, 2],
                                   ndt = parameter_matrix[i, 3])

            sns.lineplot(plot_data[:, 0] * plot_data[:, 1], 
                     ll_out_gt,
                     color = 'black',
                     alpha = 0.5,
                     label = 'TRUE',
                     ax = ax[row_tmp, col_tmp])
        
        # Get predictions from simulations /kde
        
        if not traindatanalytic:
            for j in range(nreps):
                if model == 'ddm' or model == 'ddm_analytic':
                    out = cds.ddm_flexbound(v = parameter_matrix[i, 0],
                                            a = parameter_matrix[i, 1],
                                            w = parameter_matrix[i, 2],
                                            ndt = parameter_matrix[i, 3],
                                            s = 1,
                                            delta_t = 0.001,
                                            max_t = 20, 
                                            n_samples = n_samples,
                                            print_info = False,
                                            boundary_fun = bf.constant,
                                            boundary_multiplicative = True,
                                            boundary_params = {})

                if model == 'ddm_sdv':
                    out = cds.ddm_sdv(v = parameter_matrix[i, 0],
                                            a = parameter_matrix[i, 1],
                                            w = parameter_matrix[i, 2],
                                            ndt = parameter_matrix[i, 3],
                                            sdv = parameter_matrix[i, 4],
                                            s = 1,
                                            delta_t = 0.001,
                                            max_t = 20, 
                                            n_samples = n_samples,
                                            print_info = False,
                                            boundary_fun = bf.constant,
                                            boundary_multiplicative = True,
                                            boundary_params = {})

                if model == 'full_ddm' or model == 'full_ddm2':
                    out = cds.full_ddm(v = parameter_matrix[i, 0],
                                       a = parameter_matrix[i, 1],
                                       w = parameter_matrix[i, 2],
                                       ndt = parameter_matrix[i, 3],
                                       dw = parameter_matrix[i, 4],
                                       sdv = parameter_matrix[i, 5],
                                       dndt = parameter_matrix[i, 6],
                                       s = 1,
                                       delta_t = 0.001,
                                       max_t = 20,
                                       n_samples = n_samples,
                                       print_info = False,
                                       boundary_fun = bf.constant,
                                       boundary_multiplicative = True,
                                       boundary_params = {})

                if model == 'angle' or model == 'angle2':
                    out = cds.ddm_flexbound(v = parameter_matrix[i, 0],
                                            a = parameter_matrix[i, 1],
                                            w = parameter_matrix[i, 2],
                                            ndt = parameter_matrix[i, 3],
                                            s = 1,
                                            delta_t = 0.001, 
                                            max_t = 20,
                                            n_samples = n_samples,
                                            print_info = False,
                                            boundary_fun = bf.angle,
                                            boundary_multiplicative = False,
                                            boundary_params = {'theta': parameter_matrix[i, 4]})

                if model == 'weibull_cdf' or model == 'weibull_cdf2':
                    out = cds.ddm_flexbound(v = parameter_matrix[i, 0],
                                            a = parameter_matrix[i, 1],
                                            w = parameter_matrix[i, 2],
                                            ndt = parameter_matrix[i, 3],
                                            s = 1,
                                            delta_t = 0.001, 
                                            max_t = 20,
                                            n_samples = n_samples,
                                            print_info = False,
                                            boundary_fun = bf.weibull_cdf,
                                            boundary_multiplicative = True,
                                            boundary_params = {'alpha': parameter_matrix[i, 4],
                                                               'beta': parameter_matrix[i, 5]})

                if model == 'levy':
                    out = cds.levy_flexbound(v = parameter_matrix[i, 0],
                                             a = parameter_matrix[i, 1],
                                             w = parameter_matrix[i, 2],
                                             alpha_diff = parameter_matrix[i, 3],
                                             ndt = parameter_matrix[i, 4],
                                             s = 1,
                                             delta_t = 0.001,
                                             max_t = 20,
                                             n_samples = n_samples,
                                             print_info = False,
                                             boundary_fun = bf.constant,
                                             boundary_multiplicative = True, 
                                             boundary_params = {})

                if model == 'ornstein':
                    out = cds.ornstein_uhlenbeck(v = parameter_matrix[i, 0],
                                                 a = parameter_matrix[i, 1],
                                                 w = parameter_matrix[i, 2],
                                                 g = parameter_matrix[i, 3],
                                                 ndt = parameter_matrix[i, 4],
                                                 s = 1,
                                                 delta_t = 0.001, 
                                                 max_t = 20,
                                                 n_samples = n_samples,
                                                 print_info = False,
                                                 boundary_fun = bf.constant,
                                                 boundary_multiplicative = True,
                                                 boundary_params = {})      

                mykde = kdec.logkde((out[0], out[1], out[2]))
                ll_out_gt = mykde.kde_eval((plot_data[:, 0], plot_data[:, 1]))

                # Plot kde predictions
                if j == 0:
                    sns.lineplot(plot_data[:, 0] * plot_data[:, 1], 
                                 np.exp(ll_out_gt),
                                 color = 'black',
                                 alpha = 0.5,
                                 label = 'KDE',
                                 ax = ax[row_tmp, col_tmp])
                elif j > 0:
                    sns.lineplot(plot_data[:, 0] * plot_data[:, 1], 
                                 np.exp(ll_out_gt),
                                 color = 'black',
                                 alpha = 0.5,
                                 ax = ax[row_tmp, col_tmp])

            # Plot keras predictions
            sns.lineplot(plot_data[:, 0] * plot_data[:, 1], 
                         np.exp(ll_out_keras[:, 0]),
                         color = 'green',
                         label = 'MLP',
                         alpha = 1,
                         ax = ax[row_tmp, col_tmp])

        # Legend adjustments
        if row_tmp == 0 and col_tmp == 0:
            ax[row_tmp, col_tmp].legend(loc = 'upper left', 
                                        fancybox = True, 
                                        shadow = True,
                                        fontsize = 12)
        else: 
            ax[row_tmp, col_tmp].legend().set_visible(False)
        
        
        if row_tmp == rows - 1:
            ax[row_tmp, col_tmp].set_xlabel('rt', 
                                            fontsize = 24);
        else:
            ax[row_tmp, col_tmp].tick_params(color = 'white')
        
        if col_tmp == 0:
            ax[row_tmp, col_tmp].set_ylabel('likelihood', 
                                            fontsize = 24);
        
        
        ax[row_tmp, col_tmp].set_title(ax_titles[i],
                                       fontsize = 20)
        ax[row_tmp, col_tmp].tick_params(axis = 'y', size = 16)
        ax[row_tmp, col_tmp].tick_params(axis = 'x', size = 16)
        
    for i in range(len(ax_titles), rows * cols, 1):
        row_tmp = int(np.floor(i / cols))
        col_tmp = i - (cols * row_tmp)
        ax[row_tmp, col_tmp].axis('off')

    if save == True:
        if machine == 'home':
            fig_dir = "/users/afengler/OneDrive/git_repos/nn_likelihoods/figures/" + method + "/likelihoods/"
            if not os.path.isdir(fig_dir):
                os.mkdir(fig_dir)
                
        figure_name = 'mlp_vs_kde_likelihood_'
        plt.subplots_adjust(top = 0.9)
        plt.subplots_adjust(hspace = 0.3, wspace = 0.3)
        
        if traindatanalytic == 1:
            if plot_format == 'svg':
                plt.savefig(fig_dir + '/' + figure_name + model + data_signature + '_' + train_data_type + '.svg', 
                            format = 'svg', 
                            transparent = True,
                            frameon = False)
            if plot_format == 'png':
                plt.savefig(fig_dir + '/' + figure_name + model + '_analytic' + '.png', 
                            dpi = 300) #, bbox_inches = 'tight')

        else:
            if plot_format == 'svg':
                plt.savefig(fig_dir + '/' + figure_name + model + '_kde' + '.svg', 
                            format = 'svg', 
                            transparent = True,
                            frameon = False)
                
            if plot_format == 'png':
                plt.savefig(fig_dir + '/' + figure_name + model + '_kde' + '.png', 
                            dpi = 300) #, bbox_inches = 'tight')
        
    if show:
        return plt.show()
    else:
        plt.close()
        return 'finished'
    
# Predict
def mlp_manifold(params = [],
                 vary_idx = [],
                 vary_range = [],
                 vary_name = [],
                 n_levels = 25,
                 network_dir = [],
                 save = True,
                 show = True,
                 title = 'MLP Manifold',
                 model = 'ddm',
                 traindatanalytic = 0,
                 plot_format = 'svg',
                ):
    
    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rcParams['text.usetex'] = True
    #matplotlib.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['svg.fonttype'] = 'none'
    
    # Load Keras model and initialize batch container
    keras_model = keras.models.load_model(network_dir + 'model_final.h5')

    # Prepare data structures
    
    # Data template
    plot_data = np.zeros((4000, 2))
    plot_data[:, 0] = np.concatenate(([i * 0.005 for i in range(2000, 0, -1)], [i * 0.005 for i in range(1, 2001, 1)]))
    plot_data[:, 1] = np.concatenate((np.repeat(-1, 2000), np.repeat(1, 2000)))

    data_var = np.zeros((4000 * n_levels, len(params) + 3))
    
    cnt = 0 
    for par_tmp in np.linspace(vary_range[0], vary_range[1], n_levels):
        tmp_begin = 4000 * cnt
        tmp_end = 4000 * (cnt + 1)
        params[vary_idx] = par_tmp
        data_var[tmp_begin:tmp_end, :len(params)] = params
        data_var[tmp_begin:tmp_end, len(params):(len(params) + 2)] = plot_data
        # print(data_var.shape)
        data_var[tmp_begin:tmp_end, (len(params) + 2)] = np.squeeze(np.exp(keras_model.predict(data_var[tmp_begin:tmp_end, :-1], 
                                                                                               batch_size = 100)))
        cnt += 1

    fig = plt.figure(figsize=(8, 5.5))
    ax = fig.add_subplot(111, projection = '3d')
    ax.plot_trisurf(data_var[:, -2] * data_var[:, -3], 
                    data_var[:, vary_idx], 
                    data_var[:, -1],
                    linewidth = 0.5, 
                    alpha = 1.0, 
                    cmap = cm.coolwarm)
    
    ax.set_ylabel(vary_name.upper().replace('_', '-'),  
                  fontsize = 16,
                  labelpad = 20)
    
    ax.set_xlabel('RT',  
                  fontsize = 16, 
                  labelpad = 20)
    
    ax.set_zlabel('Likelihood',  
                  fontsize = 16, 
                  labelpad = 20)
    
    ax.set_zticks(np.round(np.linspace(min(data_var[:, -1]), 
                                       max(data_var[:, -1]), 
                                       5), 
                                1))

    ax.set_yticks(np.round(np.linspace(min(data_var[:, vary_idx]), 
                                       max(data_var[:, vary_idx]), 
                                       5),
                                1))

    ax.set_xticks(np.round(np.linspace(min(data_var[:, -2] * data_var[:, -3]), 
                                       max(data_var[:, -2] * data_var[:, -3]), 
                                       5), 
                                1))
    
    ax.tick_params(labelsize = 16)
    ax.set_title(model.upper().replace('_', '-') + ' - MLP: Manifold', 
                 fontsize = 20, 
                 pad = 20)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
   
    if traindatanalytic:
        if plot_format == 'svg':
            plt.savefig('./figures/mlp/manifolds/mlp_manifold_' + model + '_vary_' + vary_name +  '_analytic' + '.svg',
                        format = 'svg', 
                        transparent = True,
                        frameon = False)
                        
        if plot_format == 'png':
            plt.savefig('./figures/mlp/manifolds/mlp_manifold_' + model + '_vary_' + vary_name +  '_analytic' + '.png', 
                        bbox_inches = 'tight',
                        dpi = 300)
    else:
        if plot_format == 'svg':
            plt.savefig('./figures/mlp/manifolds/mlp_manifold_' + model + '_vary_' + vary_name + '_kde' + '.svg',
                        format = 'svg', 
                        transparent = True,
                        frameon = False)
                        
        if plot_format == 'png':
            plt.savefig('./figures/mlp/manifolds/mlp_manifold_' + model + '_vary_' + vary_name + '_kde' + '.png', 
                        bbox_inches = 'tight',
                        dpi = 300)
        
    if show:
        return plt.show()
    else:
        plt.close()
        return 'finished ...'
    
    
if __name__ == "__main__":
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--model",
                     type = str,
                     default = 'ddm')
    CLI.add_argument("--machine",
                     type = str,
                     default = 'machine')
    CLI.add_argument("--traindatanalytic",
                     type = int,
                     default = 0)
    CLI.add_argument("--ngraphs",
                     type = int,
                     default = 9)
    CLI.add_argument("--trainfileidx",
                     type = int,
                     default = 0)
    CLI.add_argument("--networkidx",
                     type = int,
                     default = -1)
    CLI.add_argument("--mlekdereps",
                     type = int,
                     default = 10)
    CLI.add_argument("--manifoldlayers",
                     type = int,
                     default = 50)
    CLI.add_argument("--modelidentifier",
                     type = str,
                     default = '')

    args = CLI.parse_args()
    print(args)
    
    model = args.model
    machine = args.machine
    ngraphs = args.ngraphs
    trainfileidx = args.trainfileidx
    networkidx = args.networkidx
    traindatanalytic = args.traindatanalytic
    mlekdereps = args.mlekdereps
    manifoldlayers = args.manifoldlayers
    
    if args.modelidentifier == 'None':
        modelidentifier = ''
    else:
        modelidentifier = args.modelidentifier
    
    # INSERT HERE
#     machine = 'home' # put custom if you want to keep your network_path
#     model = 'weibull_cdf'

    if machine == 'x7':
        stats = pickle.load(open("/media/data_cifs/afengler/git_repos/nn_likelihoods/kde_stats.pickle", "rb"))
        model_params = stats[model]
        output_folder = model_params['output_folder_x7']
        with open("model_paths_x7.yaml") as tmp_file:
            network_path = yaml.load(tmp_file)[model]
    
    if machine == 'ccv':
        stats = pickle.load(open("/users/afengler/git_repos/nn_likelihoods/kde_stats.pickle", "rb"))
        model_params = stats[model]
        ndt_idx = model_params['param_names'].index('ndt')
        output_folder = model_params['output_folder']
        with open("model_paths.yaml") as tmp_file:
            if networkidx == -1:
                network_path = yaml.load(tmp_file)[model]
                #network_id = network_path[list(re.finditer('/', network_path))[-2].end():]
            else:
                network_path = yaml.load(tmp_file)[model + '_batch'][networkidx]
                #network_id = network_path[list(re.finditer('/', network_path))[-2].end():]

    if machine == 'home':
        stats = pickle.load(open("/users/afengler/OneDrive/git_repos/nn_likelihoods/kde_stats.pickle", "rb"))
        model_params = stats[model]
        ndt_idx = model_params['param_names'].index('ndt')
        output_folder = model_params['output_folder']
        with open("model_paths_home.yaml") as tmp_file:
            print('NETWOK ID:', networkidx)
            if networkidx == -1:
                network_path = yaml.load(tmp_file)[model + modelidentifier]
                #network_id = network_path[list(re.finditer('/', network_path))[-2].end():]
            else:
                network_path = yaml.load(tmp_file)[model + '_batch'][networkidx]
                #network_id = network_path[list(re.finditer('/', network_path))[-2].end():]
                print('Why am I passing here ?????')

    # Load training data
    if machine == 'home':
        if traindatanalytic:
            training_file_folder = '/users/afengler/OneDrive/project_nn_likelihoods/data/analytic/' + model + '/training_data_binned_0_nbins_0_n_20000/'
        else:
            training_file_folder = '/users/afengler/OneDrive/project_nn_likelihoods/data/kde/' + model + '/training_data_binned_0_nbins_0_n_20000/'
    
    if machine == 'ccv':
        if traindatanalytic:
            training_file_folder = '/users/afengler/data/analytic/' + model + '/training_data_binned_0_nbins_0_n_20000/'
        else:
            training_file_folder = '/users/afengler/data/kde/' + model + '/training_data_binned_0_nbins_0_n_20000/'
            
    training_files = os.listdir(training_file_folder)
    train_dat = pickle.load(open(training_file_folder + training_files[trainfileidx], 'rb'))
    
    # TD HERE WE SHOULD ACTUALLY HAVE > NDT NOT 0
    train_dat = train_dat[train_dat[:, -3] > train_dat[:, ndt_idx], :]

    # Preprocess
    train_dat[:, -1] = np.maximum(train_dat[:, -1], np.log(1e-7))

    # Predict Likelihoods
    keras_model = keras.models.load_model(network_path + 'model_final.h5')
    prediction = keras_model.predict(train_dat[:, :-1], 
                                     batch_size = 100)
    prediction = prediction[:, 0]
    prediction = np.maximum(prediction, np.log(1e-7))
    prediction_error = (prediction - train_dat[:, -1])

    # Distribution of Prediction Errors
    plt.hist(prediction_error, 
             bins = np.arange(-0.2, 0.2, 0.01), 
             density = True,
             histtype = 'step',
             color = 'black',
             fill = 'black')

    plt.xlabel('MLP Prediction Error', 
               size = 20)

    plt.title('Prediction Error: ' + model.upper().replace('_', '-'), 
              size = 24)
    
    if traindatanalytic:
        plt.savefig('/Users/afengler/OneDrive/git_repos/nn_likelihoods/figures/mlp/prediction_errors/prediction_error_distribution_' + model + '_analytic' + '.png', 
                    dpi = 300, 
                    bbox_inches = 'tight')
    else:
        plt.savefig('/Users/afengler/OneDrive/git_repos/nn_likelihoods/figures/mlp/prediction_errors/prediction_error_distribution_' + model + '_kde' + '.png', 
                    dpi = 300, 
                    bbox_inches = 'tight')
    plt.close()

    # Graph showing number of cases where prediction error is larger than 1, with loglikelihood of prediction on the x axis
    cnts = {}
    for err in np.arange(0.1, 1, 0.1):
        cnts[int(err * 10)] = []
        for i in range(17):
            cnts[int(err * 10)].append(np.sum(np.asarray(prediction_error > err) * np.asarray(prediction > - i)) /  np.sum(prediction > - i))
        plt.scatter(-np.arange(0, 17, 1), cnts[int(err * 10)], color = 'black', alpha = err, label = str(round(err, 1)))

    plt.xlabel('log likelihood', size = 20)
    plt.ylabel('Proportion Error >= ', size = 20)
    plt.title('Prediction Error / ll: ' + model.upper().replace('_', '-'), size = 24)
    plt.ylim((0, 0.05))
    plt.legend(title = 'Error size', title_fontsize = 14, labelspacing = 0.1, fontsize = 10)
    
    if traindatanalytic:
        plt.savefig('/Users/afengler/OneDrive/git_repos/nn_likelihoods/figures/mlp/prediction_errors/prediction_error_vs_likelihood_' + model + '_analytic' + '.png', 
                    dpi = 300, 
                    bbox_inches = 'tight')
        
    else:
        plt.savefig('/Users/afengler/OneDrive/git_repos/nn_likelihoods/figures/mlp/prediction_errors/prediction_error_vs_likelihood_' + model + '_kde' + '.png', 
                    dpi = 300, 
                    bbox_inches = 'tight')
    plt.close()


    param_bounds = np.array(model_params['param_bounds_sampler'] + model_params['boundary_param_bounds_sampler'])
    n_params = param_bounds.shape[0]

    parameter_matrix = np.random.uniform(low = param_bounds[:, 0],
                                         high = param_bounds[:, 1],
                                         size = (ngraphs, n_params))


    if model == 'angle2':
        model2 = 'angle'
    if model == 'ddm_analytic':
        model2 = 'ddm'
    if model == 'weibull_cdf2':
        model2 = 'weibull_cdf'
    if model == 'full_ddm2':
        model2 = 'full_ddm'
    else:
        model2 = model

    print('Now Plotting KDE vs. MLP Likelihoods')
    kde_vs_mlp_likelihoods(ax_titles = [str(i) for i in range(1, ngraphs + 1, 1)],
                           parameter_matrix = parameter_matrix,
                           network_dir = network_path,
                           cols = 3,
                           model = model2,
                           n_samples = 20000,
                           nreps = mlekdereps,
                           save = True,
                           show = False,
                           machine = 'home',
                           method = 'mlp',
                           traindatanalytic = traindatanalytic)
    
    
    # MANIFOLD PLOTS
#     if model == 'ddm' or model == 'ddm_analytic':
#         vary_idx_vec = [0, 1, 2, 3]
#         start_params = [0, 1.5, 0.5, 1]
#         vary_range_vec = [[-2, 2], [0.3, 2], [0.2, 0.8], [0, 2]]
#         vary_name_vec = ['v', 'a', 'w', 'ndt']
    
#     if model == 'angle2' or model == 'angle':
#         vary_idx_vec = [0, 1, 2, 3, 4]
#         start_params = [0, 1.5, 0.5, 1, 0.2]
#         vary_range_vec = [[-2, 2], [0.3, 2], [0.2, 0.8], [0, 2], [0, 1]]
#         vary_name_vec = ['v', 'a', 'w', 'ndt', 'angle']

#     if model == 'ornstein':
#         vary_idx_vec = [0, 1, 2, 3, 4]
#         start_params = [0, 1.5, 0.5, 0.0, 1]
#         vary_range_vec = [[-2, 2], [0.3, 2], [0.2, 0.8], [-1, 1], [0, 2]]
#         vary_name_vec = ['v', 'a', 'w', 'ndt', 'Inhibition']

#     if model == 'full_ddm':
#         vary_idx_vec = [0, 1, 2, 3, 4, 5, 6]
#         start_params = [0.0, 1.5, 0.5, 1.0, 0.0, 0.0, 0.0]
#         vary_range_vec = [[-2, 2], [0.3, 2], [0.2, 0.8], [0.25, 2], [0.0, 0.2], [0.0, 1.0], [0.0, 0.25]]
#         vary_name_vec = ['v', 'a', 'w', 'ndt', 'w noise', 'v noise', 'ndt noise']
        
#     # add full_ddm2
    
#     if model == 'levy':
#         vary_idx_vec = [0, 1, 2, 3, 4]
#         start_params = [0.0, 1.5, 0.5, 1.5, 1.0]
#         vary_range_vec = [[-2, 2], [0.3, 2], [0.2, 0.8], [1.0 , 2.0], [0.0, 2.0]]
#         vary_name_vec = ['v', 'a', 'w', 'noise alpha', 'ndt']

#     if model == 'weibull_cdf':
#         vary_idx_vec = [0, 1, 2, 3, 4, 5]
#         start_params = [0, 1.5, 0.5, 1.0, 2.0, 2.0]
#         vary_range_vec = [[-2, 2], [0.3, 2], [0.2, 0.8], [0.0 , 2.0], [0.3, 5.0], [0.3, 7.0]]
#         vary_name_vec = ['v', 'a', 'w', 'ndt', 'alpha bound', 'beta boundary']
        
#     if model == 'ddm_sdv' or model == 'ddm_sdv_analytic':
#         vary_idx_vec = [0, 1, 2, 3, 4]
#         start_params = [0.0, 1.5, 0.5, 1.0, 0.5]
#         vary_range_vec = [[-2, 2], [0.3, 2], [0.2, 0.8], [0.25, 2],  [0.0, 2.0]]
#         vary_name_vec = ['v', 'a', 'w', 'ndt', 'v_noise']
        
    
#     print('Now plotting MLP Likelihood Manifolds')
#     for i in range(len(start_params)):
#         print(i)
#         mlp_manifold(params = start_params.copy(),
#                      vary_idx = vary_idx_vec[i],
#                      vary_range = vary_range_vec[i],
#                      vary_name = vary_name_vec[i],
#                      n_levels = manifoldlayers,
#                      network_dir = network_path,
#                      save = True,
#                      show = False,
#                      title = 'MLP Manifold',
#                      model = model,
#                      traindatanalytic = traindatanalytic)