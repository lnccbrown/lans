import os
import numpy as np
import glob

class Config(object):

    def __init__(self):
        # Directory setup
        self.base_dir = '/media/data_cifs/lakshmi/projectABC/'
        self.data_dir = 'data'
	self.tfrecord_dir = 'tfrecords'
	self.summary_dir = 'summaries'

	# select dataset
	self.ddm_initialize()
	self.inference_dataset = glob.glob('../data/ddm/parameter_recovery/*')

	self.train_tfrecords = self.dataset_dir+'_train.tfrecords'
	self.val_tfrecords = self.dataset_dir+'_val.tfrecords'
	self.test_tfrecords = self.dataset_dir+'_test.tfrecords'

	"""
	Specify whether or not to treat gaussian errors as isotropic
	This only matters for the "reverse" model
	"""
	self.full_cov_matrix = False

        # Data configuration
        self.results_dir = '/media/data_cifs/lakshmi/projectABC/results/'
	self.model_suffix = 'full_cov' if self.full_cov_matrix else 'isotropic'
        self.model_output = os.path.join(self.base_dir,
					'models',
					self.model_name+'_'+self.model_suffix)
	
	self.data_prop = {'train':0.9, 'val':0.05, 'test':0.05}

	self.min_param_values = np.array([x[0] for x in self.bounds])
	self.param_range = np.array([x[1] - x[0] for x in self.bounds])

        # Model hyperparameters
        self.epochs = 15
        self.train_batch = 128
        self.val_batch = 64
        self.test_batch = 128
	# how often should the training stats be printed?
	self.print_iters = 250
	# how often do you want to validate?
	self.val_iters = 1000

    def angle_initialize(self):
	self.dataset_dir = 'angle_ndt'
	self.dataset = 'angle_ndt*'
	self.model_name = 'angle'
	self.param_dims = [None, 1, 5, 1]
	self.test_param_dims = [1, 1, 5, 1]
	self.output_hist_dims = [None, 1, 256, 2]
	self.bounds = [(-1.5, 1.5), (0.6, 1.5), (0.3, 0.7), (0.0, 1.0), (0, (np.pi / 2 - .2))]

    def ddm_initialize(self):
	self.dataset_dir = 'ddm_ndt'
	self.dataset = 'ddm_ndt*'
	self.model_name = 'ddm'
	self.param_dims = [None, 1, 4, 1]
	self.test_param_dims = [1, 1, 4, 1]
	self.output_hist_dims = [None, 1, 256, 2]
	self.bounds = [(-2.0, 2.0), (0.5, 1.5), (0.3, 0.7), (0.0, 1.0)]

    def weibull_initialize(self):
	self.model_name = 'weibull'
	self.dataset_dir = 'weibull_cdf_ndt'
	self.dataset = 'weibull_cdf_ndt*'
	self.param_dims = [None, 1, 6, 1]
	self.test_param_dims = [1, 1, 6, 1]
	self.output_hist_dims = [None, 1, 256, 2]
	self.bounds = [(-1.5, 1.5), (0.6, 1.5), (0.3, 0.7), (0.0, 1.0), (0.5, 5.0), (0.5, 7.0)]

    def full_ddm_initialize(self):
	self.dataset_dir = 'full_ddm'
        self.dataset = 'full_ddm*'
	self.model_name = 'fullddm'
	self.param_dims = [None, 1, 7, 1]
	self.test_param_dims = [1, 1, 7, 1]
	self.output_hist_dims = [None, 1, 256, 2]

    def ornstein_initialize(self):
	self.model_name = 'ornstein'
	self.dataset_dir = 'ornstein'
	self.dataset = 'ornstein_base*'
	self.param_dims = [None, 1, 5, 1]
	self.test_param_dims = [1, 1, 5, 1]
	self.output_hist_dims = [None, 1, 256, 2]
	self.bounds = [(-1.5, 1.5), (0.5, 1.5), (0.3, 0.7), (-1.0, 1.0), (0.0, 1.0)]

    def race_model_3_initialize(self):
	self.model_name = 'race_model_3'
	self.dataset_dir = 'race_model_3'
	self.dataset = 'race_model_base*'
	self.param_dims = [None, 1, 8, 1]
	self.test_param_dims = [1, 1, 8, 1]
	self.output_hist_dims = [None, 1, 256, 3]

    def lca_3_initialize(self):
	self.model_name = 'lca_3'
	self.dataset_dir = 'lca_3'
	self.dataset = 'lca_base*'
	self.param_dims = [None, 1, 10, 1]
	self.test_param_dims = [1, 1, 10, 1]
	self.output_hist_dims = [None, 1, 256, 3]

    def race_model_4_initialize(self):
	self.model_name = 'race_model_4'
	self.dataset_dir = 'race_model_4'
	self.dataset = 'race_model_base*'
	self.param_dims = [None, 1, 10, 1]
	self.test_param_dims = [1, 1, 10, 1]
	self.output_hist_dims = [None, 1, 256, 4]

    def lca_4_initialize(self):
	self.model_name = 'lca_4'
	self.dataset_dir = 'lca_4'
	self.dataset = 'lca_base*'
	self.param_dims = [None, 1, 12, 1]
	self.test_param_dims = [1, 1, 12, 1]
	self.output_hist_dims = [None, 1, 256, 4]

    def race_model_5_initialize(self):
	self.model_name = 'race_model_5'
	self.dataset_dir = 'race_model_5'
	self.dataset = 'race_model_base*'
	self.param_dims = [None, 1, 12, 1]
	self.test_param_dims = [1, 1, 12, 1]
	self.output_hist_dims = [None, 1, 256, 5]

    def lca_5_initialize(self):
	self.model_name = 'lca_5'
	self.dataset_dir = 'lca_5'
	self.dataset = 'lca_base*'
	self.param_dims = [None, 1, 14, 1]
	self.test_param_dims = [1, 1, 14, 1]
	self.output_hist_dims = [None, 1, 256, 5]

    def race_model_6_initialize(self):
	self.model_name = 'race_model_6'
	self.dataset_dir = 'race_model_6'
	self.dataset = 'race_model_base*'
	self.param_dims = [None, 1, 14, 1]
	self.test_param_dims = [1, 1, 14, 1]
	self.output_hist_dims = [None, 1, 256, 6]

    def lca_6_initialize(self):
	self.model_name = 'lca_6'
	self.dataset_dir = 'lca_6'
	self.dataset = 'lca_base*'
	self.param_dims = [None, 1, 16, 1]
	self.test_param_dims = [1, 1, 16, 1]
	self.output_hist_dims = [None, 1, 256, 6]

