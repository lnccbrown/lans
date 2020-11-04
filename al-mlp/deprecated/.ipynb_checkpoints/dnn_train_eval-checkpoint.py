# Loading necessary packages
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import scipy as scp
import time
from datetime import datetime
import csv
import os
import yaml

class dnn_trainer():

    # Getting working directory in case it is useful later (definition of model_dir)
    cwd = os.getcwd()

    def __init__(self, yaml_config_file_path = 'file_path'):
        
        # Read configuration file
        with open(yaml_config_file_path, 'r') as stream: 
            config_data = yaml.unsafe_load(stream) 
        
        self.model_params = config_data['model_params']
        self.train_params = config_data['train_params']
        self.data_params = config_data['data_params']
        self.data_params['timestamp'] = datetime.now().strftime('%m_%d_%y_%H_%M_%S')

        self.model  = []
        self.data = {}

   # Define function to generate keras model
    def keras_model_generate(
                             self,
                             save_model = True
                             ):
        
        # Refresh some data-structures now that data is here
        self.data_params['training_data_size'] = self.data['train_features'].shape[0]
        self.model_params['input_shape'] = self.data['train_features'].shape[1]
        self.model_params['output_shape'] = self.data['test_labels'].shape[1]

        # This returns a tensor
        inputs = keras.layers.Input(shape = (self.model_params['input_shape'], ))

        # Model hidden
        op = keras.layers.Dense(
                                self.model_params['hidden_layers'][0],
                                activation = self.model_params['hidden_activations'][0],
                                kernel_regularizer = keras.regularizers.l1_l2(l1 = self.model_params['l1_kernel'][0],
                                                                              l2 = self.model_params['l2_kernel'][0])
                               )(inputs)

        for cnt in range(1, len(self.model_params['hidden_layers']), 1):
            op = keras.layers.Dense(self.model_params['hidden_layers'][cnt],
                                    activation = self.model_params['hidden_activations'][cnt],
                                    kernel_regularizer = keras.regularizers.l1_l2(l1 = self.model_params['l1_kernel'][cnt],
                                                                              l2 = self.model_params['l2_kernel'][cnt]))(op)

        # Model output
        outputs = keras.layers.Dense(self.model_params['output_shape'],
                                     activation = self.model_params['output_activation'])(op)

        # Make model
        model = keras.models.Model(
                                   inputs = inputs,
                                   outputs = outputs
                                   )

        model.compile(
                      optimizer = self.model_params['optimizer'],
                      loss = self.model_params['loss'],
                      metrics = self.model_params['metrics']
                      )

        # Store model class
        self.model = model

        # Saving Model
        if save_model == True:
            if self.train_params['model_cnt'] == 0:
                os.mkdir(
                         self.data_params['model_directory'] + '/' + \
                         self.data_params['model_name'] + \
                         self.data_params['data_type_signature'] + \
                         self.data_params['timestamp']
                         )

            tmp_model_cnt = str(self.train_params['model_cnt'])
            model.save(
                       filepath = self.data_params['model_directory'] + '/' + \
                       self.data_params['model_name'] + \
                       self.data_params['data_type_signature'] + \
                       self.data_params['timestamp'] + '/model_' + tmp_model_cnt
                       )

    # Define Training function
    def run_training(
                     self,
                     warm_start = False,
                     save_history = True
                     ):

        # get currecnt data and time for file naming purposes
        model_cnt_tmp = str(self.train_params['model_cnt'])

        # get set of callback functions to consider dependent on the train parameters specified
        callback_funs = []
        for tmp in self.train_params['callback_funs']:
            if tmp == 'ReduceLROnPlateau':
                callback_funs.append(keras.callbacks.ReduceLROnPlateau(
                                                                       monitor = self.train_params['callback_monitor'],
                                                                       factor = self.train_params['red_coef_learning_rate'],
                                                                       patience = self.train_params['plateau_patience'],
                                                                       min_lr = self.train_params['min_learning_rate'],
                                                                       verbose = 1
                                                                       )
                                    )

            if tmp == 'EarlyStopping':
                callback_funs.append(keras.callbacks.EarlyStopping(
                                                                   min_delta = self.train_params['min_delta'],
                                                                   monitor = self.train_params['callback_monitor'],
                                                                   patience = self.train_params['early_stopping_patience'],
                                                                   verbose = 1
                                                                   )
                                     )

            if tmp == 'ModelCheckpoint':
                callback_funs.append(keras.callbacks.ModelCheckpoint(
                                                                     filepath = self.data_params['model_directory'] + '/' +\
                                                                                self.data_params['model_name'] + \
                                                                                self.data_params['data_type_signature'] + \
                                                                                self.data_params['timestamp'] + '/' + \
                                                                                self.data_params['checkpoint'] + \
                                                                                '_' + model_cnt_tmp + '_{epoch:02d}',
                                                                     save_best_only = self.train_params['ckpt_save_best_only'],
                                                                     save_weights_only = self.train_params['ckpt_save_weights_only'],
                                                                     period = self.train_params['ckpt_period'],
                                                                     verbose = 1
                                                                     )
                                    )

        # Fit Model
        if warm_start == True: # in case we want to continue training
            self.model.load_weights(
                                    filepath = self.data_params['model_directory'] + \
                                    '/' + self.data_params['model_name'] + \
                                    self.data_params['data_type_signature'] + \
                                    self.data_params['timestamp'] + '/' + \
                                    self.data_params['checkpoint']
                                    )

        else: # cold start
            train_history = self.model.fit(
                                           x = self.data['train_features'],
                                           y = self.data['train_labels'],
                                           batch_size = self.train_params['batch_size'],
                                           epochs = self.train_params['max_train_epochs'],
                                           validation_data = (self.data['test_features'], self.data['test_labels']),
                                           callbacks = callback_funs
                                           )

        train_history = pd.DataFrame(train_history.history)

        # Save Weights
        self.model.save_weights(
                                filepath = self.data_params['model_directory'] + \
                                '/' + self.data_params['model_name'] + \
                                self.data_params['data_type_signature'] + \
                                self.data_params['timestamp'] + \
                                '/' + self.data_params['checkpoint'] + '_' + model_cnt_tmp + \
                                '_final'
                                )

        # Save History
        train_history.to_csv(
                             self.data_params['model_directory'] + \
                             '/' + self.data_params['model_name'] + \
                             self.data_params['data_type_signature'] + \
                             self.data_params['timestamp'] + \
                             '/' + 'history_' + model_cnt_tmp + '.csv'
                             )

        # Save model stats
        if self.train_params['model_cnt'] == 0:
            self.file_model_stats(make = True)
        else:
            self.file_model_stats(make = False)

        # Increment model count
        self.train_params['model_cnt'] += 1

        return train_history

    def file_model_stats(
                         self,
                         make = True
                         ):

        param_dict = {}
        param_dict.update(self.model_params)
        param_dict.update(self.train_params)
        param_dict.update(self.data_params)

        headers = []
        values = list(param_dict.values())

        # Make csv file that collects model_specification
        if make == True:

            for str in param_dict.keys():
                headers.append(str)

            with open(
                      self.data_params['model_directory'] + '/' + \
                      self.data_params['model_name'] + \
                      self.data_params['data_type_signature'] + \
                      self.data_params['timestamp'] + \
                      '/model_specifications' + '.csv',
                      'w'
                      ) as f:
                writer = csv.writer(f)
                writer.writerow(headers)

        # Append csv file
        with open(
                  self.data_params['model_directory'] + '/' + \
                  self.data_params['model_name'] + \
                  self.data_params['data_type_signature'] + \
                  self.data_params['timestamp'] + \
                  '/model_specifications' + '.csv',
                  'a'
                  ) as f:

            writer = csv.writer(f)
            writer.writerow(values)
            
            
            
# NOT USED:         
#         self.model_params = {
#                             'input_shape': 3,
#                             'output_shape': 1,
#                             'output_activation': 'sigmoid',
#                             'hidden_layers': [20, 20, 20, 20],
#                             'hidden_activations': ['relu', 'relu', 'relu', 'relu'],
#                             'l1_activation': [0.0, 0.0, 0.0, 0.0],
#                             'l2_activation': [0.0, 0.0, 0.0, 0.0],
#                             'l1_kernel': [0.0, 0.0, 0.0, 0.0],
#                             'l2_kernel': [0.0, 0.0, 0.0, 0.0],
#                             'optimizer': 'Nadam',
#                             'loss': 'mse',
#                             'metrics': ['mse']
#                             }

#         self.train_params = {
#                             'callback_funs': ['ReduceLROnPlateau', 'EarlyStopping', 'ModelCheckpoint'],
#                             'plateau_patience': 10,
#                             'min_delta': 1e-4, # Minimum improvement in evaluation metric that counts as learning
#                             'early_stopping_patience': 15,
#                             'callback_monitor': 'loss',
#                             'min_learning_rate': 1e-7,
#                             'red_coef_learning_rate': 0.1,
#                             'ckpt_period': 10,
#                             'ckpt_save_best_only': True,
#                             'ckpt_save_weights_only': True,
#                             'max_train_epochs': 2000,
#                             'batch_size': 10000,
#                             'warm_start': False,
#                             'checkpoint': 'ckpt',
#                             'model_cnt': 0  # This is important for saving result files coherently, a global cnt of models run thus far
#                             }

#         self.data_params = {
#                            'data_type': 'choice_probabilities',
#                            'model_directory': self.cwd + '/keras_models',
#                            'checkpoint': 'ckpt',
#                            'model_name': 'dnnregressor',
#                            'data_type_signature': '_choice_probabilities_analytic_',
#                            'timestamp': self.timestamp,
#                            'training_data_size': 2500000
#                            }
