# Loading necessary packages
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import scipy as scp
import shutil
import csv
import os
from datetime import datetime
import itertools
import dnnregressor_model_and_input_fn as dnn_model_input
import dnnregressor_predictor as dnn_pred
import make_data_wfpt as mdw
import scipy.stats as scps

class choice_probabilities_analytic_mh():

    model_directory = os.getcwd() + '/keras_models'
    model_prefix = '/dnnregressor'

    def __init__(self):
        self.model_time = '09_03_18_17_28_21'
        self.model_signature = '_choice_probabilities_analytic_'
        self.model_num = 0
        self.model_checkpoint = 'final'
        self.data_sim_params = {'v': 1, 'a': 1, 'w': 0.5, 'n_samples': 1000}
        self.data_sim = {}
        self.priors = {'v': scps.norm(loc = 0, scale = 1),
                       'a': scps.gamma(a = 0.5, scale = 2)}
        self.mcmc_params = {'n_samples': 10000,
                            'cov_init': np.array([[1, 0], [0, 1]])
                            }
        self.chain_stats = {'acc_cnt': 0}

    def make_model_path(self):
        self.model_path = self.model_directory + \
                          self.model_prefix + \
                          self.model_signature + \
                          self.model_time + \
                          '/model_' + str(self.model_num) 
    def make_checkpoint_path(self):
        self.checkpoint_path = self.model_directory + \
                                self.model_prefix + \
                                self.model_signature + \
                                self.model_time + \
                                '/ckpt_' + str(self.model_num) + '_' + self.model_checkpoint
    
    def make_data_set(self):
        dataset, _, __ = mdw.make_data_rt_choice(v_range = [self.data_sim_params['v'], self.data_sim_params['v']],
                                                 a_range = [self.data_sim_params['a'], self.data_sim_params['a']],
                                                 w_range = [self.data_sim_params['w'], self.data_sim_params['w']],
                                                 n_samples = self.data_sim_params['n_samples'],
                                                 write_to_file = False,
                                                 mixture_p = [1, 0, 0],
                                                 method = 'sim')

        self.data_sim = {'data': dataset,
                         'n_choice_lower': np.sum(dataset['choice'][dataset['choice'] > 0]),
                         'n_samples': self.data_sim_params['n_samples']}

#     def get_hyper_params(self):
#         hyper_params = pd.read_csv(self.model_directory + '/dnnregressor_mse' + self.model_signature + self.model_time + '/dnn_training_results_mse' + self.model_signature + self.model_time + '.csv',
#                                    converters = {'hidden_units':eval, 'activations':eval})
#         hyper_params = hyper_params.to_dict(orient = 'list')
#         for key in hyper_params.keys():
#             hyper_params[key] = hyper_params[key][0]
#         self.model_params = hyper_params

#     def make_predictor(self):
#         features = {'v': [],
#                     'a': [],
#                     'w': []}

#         feature_columns = dnn_model_input.make_feature_columns_numeric(features = features)
#         self.model_params['feature_columns'] = feature_columns

#         my_predictor = dnn_pred.get_dnnreg_predictor(model_directory = self.model_directory + '/dnnregressor_mse' + self.model_signature + self.model_time + '/',
#                                                     params = self.model_params)
#         self.predictor = my_predictor

#     def get_prediction_dnn(self,
#                            params = {'v': [1],
#                                      'a': [1],
#                                      'w': [0.5]}):

#         prediction = dnn_pred.get_predictions(regressor = self.predictor,
#                                               features = params,
#                                               checkpoint = self.model_directory + self.model_prefix + self.model_signature + self.model_time + '/' + self.model_checkpoint)
#         return prediction
    
    def get_dnn_keras(self):
        
        # Load model and specified checkpoint
        model = keras.models.load_model(self.model_path)
        model.load_weights(self.checkpoint_path)
        self.model = model
    
#     def get_prediction_dnn_keras(self,
#                                  params = np.zeros((1, 3):
#         return self.predictor.predict(params, batch_size = 1)
       
                                                  
    def get_log_posterior(self,
                          params = {'v': [1.0],
                                    'a': [1.0],
                                    'w': [0.5]},
                          method = 'dnn'):

        log_target = 0

        # Prior choice_probabilities
        log_target += self.priors['v'].logpdf(params['v'][0])
        
        #print('v prior log pdf:', log_target)
        
        log_target += self.priors['a'].logpdf(params['a'][0]) # logpdf function can handle case where we propose value outside of support --> -inf
        
        #print('v and a prior log pdf:', log_target)

        # Get choice probability from neural network
#         if method == 'dnn':
#             p_lower_barrier_model = np.log(self.get_prediction_dnn(params = params))
#             print('log p lower barrier model:', p_lower_barrier_model)

        if method == 'wfpt':
            p_lower_barrier_model = np.log(mdw.choice_probabilities(v = params['v'][0],
                                                                    a = params['a'][0],
                                                                    w = params['w'][0]))
                                                  
        if method == 'dnn':
            params = np.array(list(params.values())).T
            p_lower_barrier_model = np.log(self.model.predict(params))

        #print('n samples:', self.data_sim['n_samples'])
        #print('n choice lower:', self.data_sim['n_choice_lower'])
        
        # Probability of observing choices given model
        log_target += scps.binom.logpmf(k = self.data_sim['n_choice_lower'],
                                        n = self.data_sim['n_samples'],
                                        p = np.exp(p_lower_barrier_model)
                                        )
        #print('binom logpmf:', log_target)

        return log_target

    def metropolis_hastings(self,
                            method = 'dnn', # Method is essentially the model from which we take the log choice probabilities, ('dnn' or 'wfpt')
                            print_steps = True,
                            write_to_file = False):

        # Initialize covariance matrix of proposal distribution
        cov = self.mcmc_params['cov_init']
        mean = np.array([0, 0]) # Note: I hope to make this adaptive
        mvn = scps.multivariate_normal(mean = mean,
                                       cov = cov) # Note: I hope to make this adaptive

        # Initialize parameters at prior means
        params = {'v': [self.priors['v'].mean()],
                  'a': [self.priors['a'].mean()],
                  'w': [0.5]}

        # Initialize parameter chain with zeros
        chain = pd.DataFrame(np.zeros((self.mcmc_params['n_samples'], 3)),
                             columns = ['v', 'a', 'log_posterior'])

        # Get log posterior
        log_posterior = self.get_log_posterior(params = params,
                                               method = method)

        # Run Metropolis Hastings Algorithm
        acc_cnt = 0
        for i in range(0, self.mcmc_params['n_samples'], 1):

            # Propose new parameters
            tmp = mvn.rvs(size = 1)
            params['v'][0] = params['v'][0] + tmp[0]
            params['a'][0] = params['a'][0] + tmp[1]

            # Get log log posterior
            log_posterior_proposal = self.get_log_posterior(params = params,
                                                            method = method)

            # Compute acceptance probability
            alpha = np.exp(log_posterior_proposal - log_posterior)

            # Accept / Reject and store corresponding values in chain
            if scps.uniform.rvs(loc = 0, scale = 1, size = 1) <= alpha:
                chain.loc[i] = [params['v'][0],
                                params['a'][0],
                                log_posterior_proposal]

                log_posterior = log_posterior_proposal

                acc_cnt += 1

            else:
                params['v'][0] = params['v'][0] - tmp[0]
                params['a'][0] = params['a'][0] - tmp[1]
                chain.loc[i]= [params['v'][0],
                               params['a'][0],
                               log_posterior]

            if print_steps:
                print(i)

        self.chain = chain
        self.chain_stats['acc_cnt'] = acc_cnt

        if write_to_file == True:
            chain_len = str(chain.shape[0])
            chain_method = method
            chain_time = datetime.now().strftime('%m_%d_%y_%H_%M_%S')
            self.chain.to_csv(self.model_path + '_mh_' + chain_method + '_' + chain_len + '_' + chain_time)
        return chain, acc_cnt
    
    
    def metropolis_hastings_custom(self,
                                   method = 'dnn',
                                   print_steps = True,
                                   write_to_file = False,
                                   variance_scale_param = 0.2,
                                   variance_epsilon = 0.05):
        
        x = self.data_sim['n_choice_lower'] / self.data_sim['n_samples']
 
        def get_t(x, v_star):
            return np.arccos(np.abs(np.log((1 - x) / x)) / (np.power(v_star, 2)) / \
                   np.linalg.norm([np.log((1 - x) / x) / np.power(v_star, 2), 1])) * (2 / np.pi)

        def corr_from_t(t, sign = (-1)):
            #print('t', t)
            if t < 0.5:
                return sign * (-2 * t)
            else:
                return sign * ((-1) + (2 * (t - 0.5)))
            
        def variances(t, k = variance_scale_param, eps = variance_epsilon):
            ss_a = k * (1 - t) + eps
            ss_v = k * t + eps
            return ss_a, ss_v
        
        def make_cov(ss_a, ss_v, correlation_from_t):
            cov = np.zeros((2,2))
            cov[0, 0] = ss_v
            cov[1, 1] = ss_a
            cov[0, 1] = cov[1, 0] = np.sqrt(ss_v) * np.sqrt(ss_a) * correlation_from_t
            #print('cov', cov)
            return cov
        
        def get_cov_from_state(state = [], x = x):
            t = get_t(x = x, v_star = state[0])
            correlation_from_t = corr_from_t(t, sign = np.sign(np.log((1 - x) / x)))
            ss_a, ss_v = variances(t = t)
            return make_cov(ss_a = ss_a, ss_v = ss_v, correlation_from_t = correlation_from_t)
            
        
        def q_ratio(state_proposed = [], state_departed = [], x = x):
            cov_proposed = get_cov_from_state(state = state_proposed, x = x)
            #print('cov_proposed', cov_proposed)
            cov_departed = get_cov_from_state(state = state_departed, x = x)
            #print('cov_departed', cov_departed)
            return scps.multivariate_normal.logpdf(state_departed, mean = state_proposed, cov = cov_proposed) - \
                   scps.multivariate_normal.logpdf(state_proposed, mean = state_departed, cov = cov_departed)
        
        # Initialize parameters at prior means
        params = {'v': [self.priors['v'].mean() + 0.1],
                  'a': [self.priors['a'].mean()],
                  'w': [0.5]}

        # Initialize parameter chain with zeros
        chain = pd.DataFrame(np.zeros((self.mcmc_params['n_samples'], 3)),
                             columns = ['v', 'a', 'log_posterior'])

        # Get log posterior for initial state
        log_posterior = self.get_log_posterior(params = params,
                                               method = method)
        
        # Run Metropolis Hastings Algorithm
        acc_cnt = 0
        for i in range(0, self.mcmc_params['n_samples'], 1):

            # Propose new state
            old_state = [params['v'][0], params['a'][0]]
            #print('old_state', old_state)
            cov = get_cov_from_state(state = old_state, x = x)
            tmp = scps.multivariate_normal.rvs(mean = [0,0], cov = cov)
            #print('perturbation', tmp)                                      
            params['v'][0] = old_state[0] + tmp[0]
            params['a'][0] = old_state[1] + tmp[1]
            #print('new_state', [params['v'][0], params['a'][0]])
            # Get log log posterior
            log_posterior_proposal = self.get_log_posterior(params = params,
                                                            method = method)

            # Compute acceptance probability
            tmp_log_q_ratio = q_ratio(state_proposed = [params['v'][0], params['a'][0]], 
                                                                           state_departed = old_state, 
                                                                           x = x)
            log_alpha = (log_posterior_proposal - log_posterior) + tmp_log_q_ratio
            alpha = np.exp(log_alpha)                                                                                 
            #print('q_ratio', tmp_log_q_ratio)
            #print('alpha', alpha)
            
            # Accept / Reject and store corresponding values in chain
            if scps.uniform.rvs(loc = 0, scale = 1, size = 1) <= alpha:
                chain.loc[i] = [params['v'][0],
                                params['a'][0],
                                log_posterior_proposal]

                log_posterior = log_posterior_proposal

                acc_cnt += 1

            else:
                params['v'][0] = old_state[0]
                params['a'][0] = old_state[1]
                chain.loc[i]= [params['v'][0],
                               params['a'][0],
                               log_posterior]

            if print_steps:
                print(i)
            elif i % 1000 == 0:
                print(i)
                
            
        self.chain = chain
        self.chain_stats['acc_cnt'] = acc_cnt

        if write_to_file == True:
            chain_len = str(chain.shape[0])
            chain_method = method
            chain_time = datetime.now().strftime('%m_%d_%y_%H_%M_%S')
            self.chain.to_csv(self.model_path + '_mhc_' + chain_method + '_' + chain_len + '_' + chain_time)
#                               + '_mh_' + chain_method + '_' + chain_len '_' + chain_time)
        return chain, acc_cnt
