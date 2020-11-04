import os
import pickle
import numpy as np
import re
from string import ascii_letters
from datetime import datetime
import argparse
import gzip


def collect_datasets_is(folder = [],
                        model = [],
                        ndata = [],
                        nsubsample = []):
    
    # Load in parameter recovery data
    if machine == 'ccv':
        if model == 'weibull' or model == 'weibull2':
                param_recov_files = os.listdir('/users/afengler/data/kde/' + 'weibull_cdf' + '/parameter_recovery_data_binned_1_nbins_512_n_' + str(ndata) + '/')
                param_recov_dat = pickle.load(open('/users/afengler/data/kde/' + 'weibull_cdf' + '/parameter_recovery_data_binned_1_nbins_512_n_' + str(ndata) + '/' + param_recov_files[0], 'rb'))

        else:
                param_recov_files = os.listdir('/users/afengler/data/kde/' + model + '/parameter_recovery_data_binned_1_nbins_512_n_' + str(ndata) + '/')
                param_recov_dat = pickle.load(open('/users/afengler/data/kde/' + model + '/parameter_recovery_data_binned_1_nbins_512_n_' + str(ndata) + '/' + param_recov_files[0], 'rb'))
    
    if machine == 'x7':       
        param_recov_files = os.listdir('/media/data_cifs/projects/prj_approx-bayes/projectABC/data/' + model + '/parameter_recovery_data_binned_1_nbins_512_n_' + str(ndata) + '/')
        param_recov_dat = pickle.load(open('/media/data_cifs/projects/prj_approx-bayes/projectABC/data/' + model + '/parameter_recovery_data_binned_1_nbins_512_n_' + str(ndata) + '/' + param_recov_files[0], 'rb'))
            
    
    n_data_substring = 'N_' + str(ndata)
    
    is_dict = {}
    is_dict['gt'] = []
    is_dict['posterior_samples'] = []
    is_dict['timings'] = []
    is_dict['perplexities'] = []
    is_dict['importance_weights'] = []
    is_dict['effective_sample_size'] = []
    is_dict['means'] = []
    is_dict['maps'] = []
    is_dict['data'] = []
    
    files_ = os.listdir(folder)
    cnt = 0
    for file_ in files_:
        if model + '_training_' in file_ and n_data_substring in file_ and 'summary' not in file_:
            print(cnt)
            print('Processing file: ', file_)
            cnt += 1
            # extract id
            st = file_.find('_idx_')
            fin = file_.find('_tdist')
            idx = int(file_[st + len('_idx_'):fin])
            
            tmp = pickle.load(gzip.open(folder + file_, 'rb'), encoding = 'latin1')
            
            sub_idx = np.random.choice(tmp['posterior_samples'].shape[0], nsubsample, replace = False) 
            is_dict['gt'].append(tmp['gt_params'])
            is_dict['posterior_samples'].append(tmp['posterior_samples'][sub_idx, :])
            is_dict['timings'].append(tmp['timeToConvergence'])
            is_dict['perplexities'].append(tmp['norm_perplexity'])
            is_dict['importance_weights'].append(tmp['final_w'][sub_idx])
            is_dict['effective_sample_size'].append(1 / np.sum(np.square(tmp['final_w'])))
            is_dict['means'].append(np.mean(tmp['posterior_samples'], axis = 0))
            is_dict['maps'].append(tmp['final_x'][np.argmax(tmp['log_likelihood']), :])
            
            # Add data
            is_dict['data'].append(param_recov_dat[1][0][idx, : , :])
        
            
            print('Processed file: ', file_)
        #print(model + '_training_' in file_)
    
    is_dict['gt'] = np.stack(is_dict['gt'])
    is_dict['posterior_samples'] = np.stack(is_dict['posterior_samples'])
    is_dict['timings'] = np.array(is_dict['timings'])
    is_dict['perplexities'] = np.array(is_dict['perplexities'])
    is_dict['importance_weights'] = np.stack(is_dict['importance_weights'])
    is_dict['means'] = np.stack(is_dict['means'])
    is_dict['maps'] = np.stack(is_dict['maps'])
    is_dict['data'] = np.stack(is_dict['data'])
    
    
    if machine == 'ccv':
        if model == 'weibull':
            print('writing to file: ', '/users/afengler/data/eLIFE_exps/summaries/IS_summary_' + 'weibull_cdf' + \
                             '_' + n_data_substring + '.pickle')

            pickle.dump(is_dict, open('/users/afengler/data/eLIFE_exps/summaries/IS_summary_' + 'weibull_cdf' + \
                             '_' + n_data_substring + '.pickle', 'wb'), protocol = 4)

        else:
            print('writing to file: ', '/users/afengler/data/eLIFE_exps/summaries/IS_summary_' + model + \
                             '_' + n_data_substring + '.pickle')

            pickle.dump(is_dict, open('/users/afengler/data/eLIFE_exps/summaries/IS_summary_' + model + \
                             '_' + n_data_substring + '.pickle', 'wb'), protocol = 4)
    if machine == 'x7':
        print('writing to file: ', '/media/data_cifs/projects/prj_approx-bayes/projectABC/' + isfolder + '/' + 'IS_summary_' + \
              model + '_' + n_data_substring + '.pickle')

        pickle.dump(is_dict, open( '/media/data_cifs/projects/prj_approx-bayes/projectABC/' + isfolder + '/' + 'IS_summary_' + \
              model + '_' + n_data_substring + '.pickle', 'wb'), protocol = 4)
    
    return is_dict

if __name__ == "__main__":
    # Currently available models = ['weibull', 'race_model_6', 'ornstein', 'full_ddm', 'ddm_seq2', 'ddm_par2', 'ddm_mic2']

    CLI = argparse.ArgumentParser()
    CLI.add_argument("--machine",
                     type = str,
                     default = 'x7')
    CLI.add_argument("--method",
                     type = str,
                     default = 'ddm')
    CLI.add_argument("--ndata",
                     type = int,
                     default = 1024)
    CLI.add_argument("--nsubsample",
                     type = int,
                     default = 10000)
    CLI.add_argument("--isfolder",
                     type = str,
                     default = 'eLIFE_exps')
    
    args = CLI.parse_args()
    print(args)

    machine = args.machine
    method = args.method 
    ndata = args.ndata
    nsubsample = args.nsubsample
    isfolder = args.isfolder
    
    if machine == 'home':
        is_sample_folder = '/Users/afengler/OneDrive/project_nn_likelihoods/data/' + isfolder + '/'
        if method == 'weibull_cdf' or method == 'weibull_cdf2':
            method = 'weibull'
    
    if machine == 'ccv':  
        is_sample_folder = '/users/afengler/data/' + isfolder + '/'
        if method == 'weibull_cdf' or method == 'weibull_cdf2':
            method = 'weibull'
    
    if machine == 'x7':
        is_sample_folder = '/media/data_cifs/projects/prj_approx-bayes/projectABC/' + isfolder + '/'

    print(is_sample_folder)
    print('Started processing model: ', method, ' with ndata: ', ndata)
    collect_datasets_is(folder = is_sample_folder,
                        model = method,
                        ndata = ndata,
                        nsubsample = nsubsample)