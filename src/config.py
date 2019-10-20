import os
import numpy as np

class Config(object):

    def __init__(self):
        # Directory setup
        self.base_dir = '/media/data_cifs/lakshmi/projectABC/'

        # Data configuration
        self.param_dims = [None,1,4]

        self.results_dir = '/media/data_cifs/lakshmi/zebrafish/results/'
        self.model_output = ''
        self.model_input = ''
        self.train_summaries = ''

        # Model hyperparameters
        self.epochs = 100
        #self.image_orig_size = [1080, 1920, 3]
        #self.image_target_size = [416, 416, 3]
        #self.image_orig_size = [1080, 1920, 1]
        self.image_orig_size = [480, 640, 1]
        self.image_target_size = [416, 416, 1]

        self.label_shape = [13,13,3]
        self.resize_ims = True
        self.train_batch = 64
        self.val_batch= 8
        self.test_batch = 1

        self.model_output = '/media/data_cifs/lakshmi/zebrafish/darkAndLight_Bootstrapped/'
        self.model_name = 'cnn_box'
        self.num_classes = 2
