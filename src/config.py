import os
import numpy as np

class Config(object):

    def __init__(self):
        # Directory setup
        self.base_dir = '/media/data_cifs/lakshmi/projectABC/'
        self.data_dir = 'data'
        self.dataset = 'angle_ndt*'
	self.tfrecord_dir = 'tfrecords'
	self.summary_dir = 'summaries'
	self.train_tfrecords = self.dataset[:-1]+'_train.tfrecords'
	self.val_tfrecords = self.dataset[:-1]+'_val.tfrecords'
	self.test_tfrecords = self.dataset[:-1]+'_test.tfrecords'

        # Data configuration
        # Let's say we go with NxHxWxC configuration as of now
	self.model_name = 'mlp_cnn_angle'
        self.param_dims = [None, 1, 5, 1]
	self.test_param_dims = [1,1,5,1]
        self.output_hist_dims = [None, 1, 256, 2]
        self.results_dir = '/media/data_cifs/lakshmi/projectABC/results/'
        self.model_output = os.path.join(self.base_dir,
					'models',
					self.model_name)
	
	self.data_prop = {'train':0.9, 'val':0.05, 'test':0.05}

        # Model hyperparameters
        self.epochs = 100
        self.train_batch = 64
        self.val_batch= 1
        self.test_batch = 64
	# how often should the training stats be printed?
	self.print_iters = 250
	# how often do you want to validate?
	self.val_iters = 1000
