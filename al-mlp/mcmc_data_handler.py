import os
import pickle
import numpy as np
import re
from string import ascii_letters
from datetime import datetime
import argparse
import yaml

def collect_datasets_diff_evo(in_files = [],
                              out_file = [],
                              n_post_samples_by_param = 15000,
                              sort_ = True,
                              save = True):
    """Function prepares raw mcmc data for plotting"""
    
    # Intialization
    in_files = sorted(in_files)
    tmp = pickle.load(open(in_files[0],'rb'))
    n_param_sets = len(in_files) * len(tmp[2])
    n_param_sets_file = len(tmp[2])
    n_chains = tmp[2][0][0].shape[0]
    #n_samples = tmp[2][0][0].shape[1]
    n_params = tmp[2][0][0].shape[2]
    n_data = tmp[1].shape[1]
    n_choices = tmp[1].shape[2]
    
    # Data containers 
    means = np.zeros((n_param_sets, n_params))
    maps = np.zeros((n_param_sets, n_params))
    orig_params = np.zeros((n_param_sets, n_params))
    orig_data = np.zeros((n_param_sets, n_data, n_choices))
    r_hat_last = np.zeros((n_param_sets))
    timings = np.zeros((n_param_sets))
    posterior_subsamples = np.zeros((n_param_sets, n_post_samples_by_param, n_params))
    posterior_subsamples_ll = np.zeros((n_param_sets, n_post_samples_by_param))

    file_cnt = 0
    for file_ in in_files:
        # Load datafile in
        tmp_data = pickle.load(open(file_, 'rb'))
        for i in range(n_param_sets_file):
            
            # Extract samples and log likelihood sequences
            tmp_samples = np.reshape(tmp_data[2][i][0][:, :, :], (-1, n_params))
            tmp_log_l = np.reshape(tmp_data[2][i][1][:, :], (-1))        
            tmp_timing = tmp_data[3][i]

            # Fill in return datastructures
            posterior_subsamples[(n_param_sets_file * file_cnt) + i, :, :] = tmp_samples[np.random.choice(tmp_samples.shape[0], size = n_post_samples_by_param), :]
            posterior_subsamples_ll[(n_param_sets_file * file_cnt) + i, :] = tmp_log_l[np.random.choice(tmp_log_l.shape[0], size = n_post_samples_by_param)]
            means[(n_param_sets_file * file_cnt) + i, :] = np.mean(tmp_samples, axis = 0)
            maps[(n_param_sets_file * file_cnt) + i, :] = tmp_samples[np.argmax(tmp_log_l), :]
            orig_params[(n_param_sets_file * file_cnt) + i, :] = tmp_data[0][i, :]
            orig_data[(n_param_sets_file * file_cnt) + i, :, :] = tmp_data[1][i, :, :]
            r_hat_last[(n_param_sets_file * file_cnt) + i] = tmp_data[2][i][2][-1]
            timings[(n_param_sets_file * file_cnt) + i] = tmp_timing

        print(file_cnt)
        file_cnt += 1
    
    out_dict = {'means': means, 
                'maps': maps, 
                'gt': orig_params, 
                'r_hats': r_hat_last, 
                'posterior_samples': posterior_subsamples, 
                'posterior_ll': posterior_subsamples_ll, 
                'data': orig_data,
                'timings': timings}
    
    if save == True:
        print('writing to file to ' + out_file)
        pickle.dump(out_dict, open(out_file, 'wb'), protocol = 2)
    
    return out_dict

def collect_datasets_slice(in_files = [],
                           out_file = [],
                           n_post_samples_by_param = 1500,
                           n_burnin = 500,
                           sort_ = True,
                           save = True):
    """Function prepares raw mcmc data for plotting"""
    
    # Intialization
    in_files = sorted(in_files)
    tmp = pickle.load(open(in_files[0],'rb'))
    n_param_sets = len(in_files) * len(tmp[2])
    n_param_sets_file = len(tmp[2])
    #n_chains = tmp[2][0][0].shape[0]
    #n_samples = tmp[2][0][0].shape[1]
    n_params = tmp[2][0][0].shape[1]
    n_data = tmp[1].shape[1]
    
    # this chould be different for cnn - mlp
    n_choices = np.unique(tmp[1][0, :, 1]).shape[0]
    
    # Data containers 
    means = np.zeros((n_param_sets, n_params))
    maps = np.zeros((n_param_sets, n_params))
    orig_params = np.zeros((n_param_sets, n_params))
    orig_data = np.zeros((n_param_sets, n_data, n_choices))
    r_hat_last = np.ones((n_param_sets)) # Just placeholder for consistency
    timings = np.zeros((n_param_sets))
    posterior_subsamples = np.zeros((n_param_sets, n_post_samples_by_param, n_params))
    posterior_subsamples_ll = np.zeros((n_param_sets, n_post_samples_by_param))

    file_cnt = 0
    for file_ in in_files:
        # Load datafile in
        tmp_data = pickle.load(open(file_, 'rb'))
        for i in range(n_param_sets_file):
            
            # Extract samples and log likelihood sequences
            tmp_samples = tmp_data[2][i][0] 
            tmp_log_l = tmp_data[2][i][1]
            #tmp_samples = np.reshape(tmp_data[2][i][0][:, :, :], (-1, n_params))
            #tmp_log_l = np.reshape(tmp_data[2][i][1][:, :], (-1))        
            tmp_timing = tmp_data[3][i]

            # Fill in return datastructures
            tmp_ids = np.random.choice(np.arange(n_burnin, tmp_samples.shape[0], 1), size = n_post_samples_by_param, replace = False)
            posterior_subsamples[(n_param_sets_file * file_cnt) + i, :, :] = tmp_samples[tmp_ids, :]
            posterior_subsamples_ll[(n_param_sets_file * file_cnt) + i, :] = tmp_log_l[tmp_ids]
            means[(n_param_sets_file * file_cnt) + i, :] = np.mean(tmp_samples[n_burnin:, :], axis = 0)
            maps[(n_param_sets_file * file_cnt) + i, :] = tmp_samples[np.argmax(tmp_log_l), :]
            orig_params[(n_param_sets_file * file_cnt) + i, :] = tmp_data[0][i, :]
            orig_data[(n_param_sets_file * file_cnt) + i, :, :] = tmp_data[1][i, :, :]
            # r_hat_last[(n_param_sets_file * file_cnt) + i] = tmp_data[2][i][2][-1]
            timings[(n_param_sets_file * file_cnt) + i] = tmp_timing

        print(file_cnt)
        file_cnt += 1
    
    out_dict = {'means': means, 
                'maps': maps, 
                'gt': orig_params, 
                'r_hats': r_hat_last, 
                'posterior_samples': posterior_subsamples, 
                'posterior_ll': posterior_subsamples_ll, 
                'data': orig_data,
                'timings': timings}
    
    if save == True:
        print('writing to file to ' + out_file)
        pickle.dump(out_dict, open(out_file, 'wb'), protocol = 2)
    
    return out_dict

if __name__ == "__main__":
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--machine",
                     type = str,
                     default = 'x7')
    CLI.add_argument("--method",
                     type = str,
                     default = 'ddm')
    CLI.add_argument("--nburnin",
                    type = int,
                    default = 0)
    CLI.add_argument("--ndata",
                     type = int,
                     default = 1024)
    CLI.add_argument("--nsubsample",
                     type = int,
                     default = 10000)
    CLI.add_argument("--nnbatchid",
                     type= int,
                     default = -1)
    CLI.add_argument("--analytic",
                     type = int,
                     default = 0)
    CLI.add_argument("--initmode",
                     type = str,
                     default = '')
    CLI.add_argument("--sampler",
                     type = str,
                     default = 'diff_evo')
    CLI.add_argument("--fileprefix",
                     type = str,
                     default = None)
    CLI.add_argument("--modelidentifier",
                     type = str,
                     default = 'None')
    
    args = CLI.parse_args()
    print(args)

    machine = args.machine
    method = args.method
    nburnin = args.nburnin    
    ndata = args.ndata
    nsubsample = args.nsubsample
    nnbatchid = args.nnbatchid
    analytic = args.analytic
    initmode = args.initmode
    sampler = args.sampler
    
    if args.modelidentifier == None or args.modelidentifier == 'None':
        modelidentifier = ''
    else:
        modelidentifier = args.modelidentifier
        
    
    if args.fileprefix == None or args.fileprefix == 'None':
        fileprefix = ''
    else:
        fileprefix = args.fileprefix

    if machine == 'home':
        method_comparison_folder = '/Users/afengler/OneDrive/project_nn_likelihoods/data/kde/' + method + '/method_comparison/'
    
    if machine == 'ccv':        
        if method == 'ddm_analytic':
            method_comparison_folder = '/users/afengler/data/analytic/' + 'ddm' + '/method_comparison/'
            network_id = ''
            network_path = ''
        elif method == 'ddm_sdv_analytic':
            method_comparison_folder = '/users/afengler/data/analytic/' + 'ddm_sdv' + '/method_comparison/'
            network_id = ''
            network_path = ''
        else:
            method_comparison_folder = '/users/afengler/data/kde/' + method + '/method_comparison/'
        
        if analytic:
            method_comparison_folder += 'analytic/'
            network_path = ''
            network_id = ''
        else:
            with open("/users/afengler/git_repos/nn_likelihoods/model_paths.yaml") as tmp_file:
                if nnbatchid == -1:
                    network_path = yaml.load(tmp_file)[method + modelidentifier]
                    network_id = network_path[list(re.finditer('/', network_path))[-2].end():]

                else:
                    network_path = yaml.load(tmp_file)[method + modelidentifier + '_batch'][nnbatchid]
                    network_id = network_path[list(re.finditer('/', network_path))[-2].end():]

    if machine == 'x7':
        method_comparison_folder = '/media/data_cifs/afengler/data/kde/' + model + '/method_comparison/'
        
        with open("/media/data_cifs/afengler/git_repos/nn_likelihoods/model_paths_x7.yaml") as tmp_file:
            if nnbatchid == -1:
                network_path = yaml.load(tmp_file)[method + modelidentifier]
                network_id = network_path[list(re.finditer('/', network_path))[-2].end():]
            else:
                network_path = yaml.load(tmp_file)[method + modelidentifier + '_batch'][nnbatchid]
                network_id = network_path[list(re.finditer('/', network_path))[-2].end():]
                
    print('Loading network from: ')
    print(network_path)
    
    if initmode == '':
        file_signature = fileprefix + 'post_samp_data_param_recov_unif_reps_1_n_' + str(ndata) + '_' #+ '_1_'
    else:
        file_signature = fileprefix + 'post_samp_data_param_recov_unif_reps_1_n_' + str(ndata) + '_init_' + initmode + '_' #'_1_'
        print('file_signature: ', file_signature)
    
    summary_file = method_comparison_folder + network_id + '/summary_' + file_signature[:-1] + '.pickle'
    file_signature_len = len(file_signature)
    
    print(method_comparison_folder + network_id + '/')
    files = os.listdir(method_comparison_folder + network_id + '/')
    #files_ = [method_comparison_folder + network_id + '/' + file_ for file_ in files if file_[:file_signature_len] == file_signature]
    files_ = [method_comparison_folder + network_id + '/' + file_ for file_ in files if file_signature in file_]
    
    print(files_)
    
    if sampler == 'diffevo':
        _ = collect_datasets_diff_evo(in_files = files_,
                                      out_file = summary_file,
                                      n_post_samples_by_param = nsubsample,
                                      sort_ = True,
                                      save = True)
    
    if sampler == 'slice':
        _ = collect_datasets_slice(in_files = files_,
                                   out_file = summary_file,
                                   n_post_samples_by_param = nsubsample,
                                   n_burnin = nburnin,
                                   sort_ = True,
                                   save = True)
    
            