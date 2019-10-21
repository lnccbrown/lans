import os
import numpy as np

class Config(object):

    def __init__(self):
        # Directory setup
        self.base_dir = '/media/data_cifs/lakshmi/projectABC/'
        self.data_dir = 'data'
        self.dataset = 'dataset_parallel_*'

        # Data configuration
        # Let's say we go with NxHxWxC configuration as of now
        self.param_dims = [None, 1, 4, 1]
        self.output_hist_dims = [None, 1, 500, 2]
        self.results_dir = '/media/data_cifs/lakshmi/projectABC/results/'
        self.model_output = '/media/data_cifs/lakshmi/projectABC/models/cnn-v0'

        # Model hyperparameters
        self.epochs = 100
        self.train_batch = 64
        self.val_batch= 8
        self.test_batch = 1
        self.model_name = 'mlp_cnn'
