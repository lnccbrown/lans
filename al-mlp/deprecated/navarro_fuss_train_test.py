# Load my functions
import make_data_wfpt as mdw
# Load basic utilities
import scipy as scp
import numpy as np
import pandas as pd
import psutil
import pickle
import os

if __name__ == "__main__":

    # PICK
    target_folder = '/users/afengler/data/navarro_fuss/train_test_data_kde_imit/'

    mdw.make_data_rt_choice(v_range = [-2.0, 2.0],
                            a_range = [1, 3],
                            w_range = [0.3, 0.7],
                            rt_params = [1,2],
                            n_samples = 1000000,
                            eps = 10**(-29),
                            target_folder = target_folder,
                            write_to_file = True,
                            print_detailed_cnt = False,
                            method = 'kde_imit',
                            mixture_p = [0.8, 0.1, 0.1])
