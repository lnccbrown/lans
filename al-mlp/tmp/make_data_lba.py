import numpy as np
import scipy as scp
import pandas as pd
from datetime import datetime
import glob
import os
import uuid
import ddm_data_simulation as ddm_data_simulator
import scipy.integrate as integrate
import lba 


# Generate training / test data for DDM
# We want training data for
# v ~ U(-3,3)
# a ~ U[0.1, 3]
# w ~ U(0,1)
# rt ~ random.sample({-1, 1}) * GAMMA(scale = 1, shape = 2)

# Function that generate 'features' (ML parlance, here features are (v, a, w, rt, c))
# ----

def gen_lba_features_kde_imit(v_range = [1, 2],
                              A_range = [0.1, 3],
                              b_range = [0, 1],
                              s_range = [0.1, 0.2],
                              n_choices = 2,
                              n_samples = 20000,
                              n_by_param = 1000,
                              mixture_p = [0.8, 0.1, 0.1],
                              print_detailed_cnt = False):

    # Initial computations for dimension consistency
    n_params = int(np.ceil(n_samples / n_by_param))
    n_lba = int(n_by_param * mixture_p[0])
    n_unif_up = int(n_by_param * mixture_p[1])
    n_unif_down = int(n_by_param * mixture_p[2])
    n_samples_by_param = n_lba + n_unif_up + n_unif_down

    # Initialize dataframe -----------------
    data_columns = []
    
    # add v columns
    for i in range(n_choices):
        data_columns.append('v_' + str(i))
        
    data_columns = data_columns + ['A', 'b', 's', 'rt', 'choice']
    len_data_columns = len(data_columns)
    data = pd.DataFrame(np.zeros((n_params * (n_samples_by_param), len(data_columns))), columns = data_columns)
    # --------------------------------------
    
    row_cnt = 0
    for i in np.arange(0, n_params, 1):

        v_tmp = []
        for j in range(n_choices):
            v_tmp.append(np.random.uniform(low = v_range[0], high = v_range[1]))

        # Get random sample from specified parameter space
        A_tmp = np.random.uniform(low = A_range[0], high = A_range[1])
        b_tmp = np.random.uniform(low = b_range[0], high = b_range[1])
        s_tmp = np.random.uniform(low = s_range[0], high = s_range[1])

        # Store parameter values in appropriate rows in dataframe
        data.loc[row_cnt:(row_cnt + n_samples_by_param), data_columns[:-2]] = v_tmp + [A_tmp, b_tmp, s_tmp]

        # Mixture Component 1: Model Simulations -------------------
        rt_tmp, choice_tmp, _ = lba.rlba(v = v_tmp,
                                         A = A_tmp,
                                         b = b_tmp,
                                         s = s_tmp,
                                         n_samples = n_lba)
        
        data.iloc[row_cnt:(row_cnt + n_lba), len_data_columns - 2] = rt_tmp.ravel()
        data.iloc[row_cnt:(row_cnt + n_lba), len_data_columns - 1] = choice_tmp.ravel()
        # ----------------------------------------------------------

        # Mixture Component 2: Lower Uniform -----------------------
        choice_tmp = np.random.choice([0, 1], size = n_unif_down)
        rt_tmp = np.random.uniform(low = -1, high = 0.0001, size = n_unif_down)

        data.iloc[(row_cnt + n_lba):(row_cnt + n_lba + n_unif_down), len_data_columns - 2] = rt_tmp
        data.iloc[(row_cnt + n_lba):(row_cnt + n_lba + n_unif_down), len_data_columns - 1] = choice_tmp
        # ---------------------------------------------------------

        # Mixture Component 3: Upper Uniform ----------------------
        choice_tmp = np.random.choice([0, 1], size = n_unif_up)
        rt_tmp = np.random.uniform(low = 0.0001, high = 100, size = n_unif_up)

        data.iloc[(row_cnt + n_lba + n_unif_down):(row_cnt + n_samples_by_param), len_data_columns - 2] = rt_tmp
        data.iloc[(row_cnt + n_lba + n_unif_down):(row_cnt + n_samples_by_param), len_data_columns - 1] = choice_tmp
        # ---------------------------------------------------------

        row_cnt += n_samples_by_param
        print(i, ' parameters sampled')

    return data

# Function that generates 'Labels' (ML parlance, here 'label' refers to a navarro-fuss likelihood computed for datapoint of the form (v,a,w,rt,c))
# ----
def gen_lba_labels(data = [1,1,0,1], eps = 1e-16):
    labels = np.zeros((data.shape[0],1))
    
    # extract v_s from data frame
    v_keys = []
    for key in data.keys():
        if key[0] == 'v':
            v_keys.append(key)
    
    # Generate labels
    for i in np.arange(0, labels.shape[0], 1):
        if data.loc[i, 'rt'] <= 0:
            labels[i] = np.log(eps)
        else:
            labels[i] = lba.dlba(rt = data.loc[i, 'rt'], 
                                 choice = data.loc[i, 'choice'],
                                 v = np.array(data.loc[i, v_keys], dtype = np.float32),
                                 A = data.loc[i, 'A'],
                                 b = data.loc[i, 'b'],
                                 s = data.loc[i, 's'],
                                 return_log = True,
                                 eps = eps)

        if (i % 1000) == 0:
            print('label ' + str(i) + ' generated')
    return labels
# ----

# Functions to generate full datasets
# ----
def make_data_rt_choice(v_range = [1, 2],
                        A_range = [0, 1],
                        b_range = [1.5, 3],
                        s_range = [0.1, 0.2],
                        n_choices = 2,
                        n_samples = 20000,
                        eps = 1e-16,
                        target_folder = '',
                        write_to_file = True,
                        print_detailed_cnt = True,
                        mixture_p = [0.8, 0.1, 0.1],
                        n_by_param = 1000):

    # Make target folder if it doesn't exist:
    if not os.path.isdir(target_folder):
        os.mkdir(target_folder)

    data_features = gen_lba_features_kde_imit(v_range = v_range,
                                              A_range = A_range,
                                              b_range = b_range,
                                              s_range = s_range,
                                              n_choices = n_choices,
                                              n_samples = n_samples,
                                              n_by_param = n_by_param,
                                              print_detailed_cnt = print_detailed_cnt,
                                              mixture_p = mixture_p)

    # Make labels
    data_labels = pd.DataFrame(gen_lba_labels(data = data_features, eps = eps),
                               columns = ['log_likelihood'])


    # Column concat features and labels
    data = pd.concat([data_features, data_labels], axis = 1)

    # Write to file
    if write_to_file == True:
       data.to_pickle(target_folder + '/data_' + uuid.uuid1().hex + '.pickle',
                      protocol = 4)

    return data

# Functions that make and load training and test sets from the basic datafiles generated above --------
def make_train_test_split(source_folder = '',
                          target_folder = '',
                          p_train = 0.8):

    # Deal with target folder
    if target_folder == '':
        target_folder = source_folder
    else:
        if not os.path.isdir(target_folder):
            os.mkdir(target_folder)


    # Get files in specified folder
    print('get files in folder')
    files_ = os.listdir(folder)

    # Check if folder currently contains a train-test split
    print('check if we have a train and test sets already')
    for file_ in files_:
        if file[:7] == 'train_f':
            return 'looks like a train test split exists in folder: Please remove before running this function'

    # If no train-test split in folder: collect 'data_*' files
    print('folder clean so proceeding...')
    data_files = []
    for file_ in files_:
        if file_[:5] == 'data_':
            data_files.append(file_)

    # Read in and concatenate files
    print('read, concatenate and shuffle data')
    data = pd.concat([pd.read_pickle(folder + file_) for file_ in data_files_])

    # Shuffle data
    np.random.shuffle(data.values)
    data.reset_index(drop = True, inplace = True)

    # Get meta-data from dataframe
    n_cols = len(list(data.keys()))

    # Get train and test ids
    print('get training and test indices')
    train_id = np.random.choice(a = [True, False],
                                size = data.shape[0],
                                replace = True,
                                p = [p_train, 1 - p_train])

    test_id = np.invert(train_id)

    # Write to file
    print('writing to file...')
    data.iloc[train_id, :(len(n_cols) - 1)].to_pickle(folder + 'train_features.pickle',
                                                          protocol = 4)

    data.iloc[test_id, :(len(n_cols) - 1)].to_pickle(folder + 'test_features.pickle',
                                                         protocol = 4)

    data.iloc[train_id, (len(n_cols) - 1)].to_pickle(folder + 'train_labels.pickle',
                                                         protocol = 4)

    data.iloc[test_id, (len(n_cols) - 1)].to_pickle(folder + 'test_labels.pickle',
                                                        protocol = 4)

    return 'success'

def load_data(folder = '',
              return_log = False, # function expects log likelihood, so if log = False we take exponent of loaded labels 
              prelog_cutoff = 1e-29 # 'none' or value
              ):

    # Load training data from file
    train_features = pd.read_pickle(folder + '/train_features.pickle')
    train_labels = np.transpose(np.asmatrix(pd.read_pickle(folder + '/train_labels.pickle')))

    # Load test data from file
    test_features = pd.read_pickle(folder + '/test_features.pickle')
    test_labels = np.transpose(np.asmatrix(pd.read_pickle(folder + '/test_labels.pickle')))

    # Preprocess labels
    # 1. get rid of labels smaller than cutoff
    if prelog_cutoff != 'none':
        train_labels[train_labels < np.log(prelog_cutoff)] = np.log(prelog_cutoff)
        test_labels[test_labels < np.log(prelog_cutoff)] = np.log(prelog_cutoff)

    # 2. Take exp
    if return_log == False:
        train_labels = np.exp(train_labels)
        test_labels = np.exp(test_labels)

    return train_features, train_labels, test_features, test_labels
# -------------------------------------------------------------------------------------------