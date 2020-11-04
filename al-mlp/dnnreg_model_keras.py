import tensorflow as tf
from tensorflow import keras
import os
import pandas as pd



# Function asks for a dictionary as input with the following keys (and associated datatypes)
# params = {'input_shape': 3,
#           'output_shape': 1,
#           'output_activation': 'sigmoid',
#           'hidden_layers': [20, 20, 20],
#           'hidden_activations': ['relu', 'relu', 'relu'],
#           'l1_activation': [0.0, 0.0, 0.0],
#           'l2_activation': [0.0, 0.0, 0.0],
#           'l1_kernel': [0.0, 0.0, 0.0, 0.0],
#           'l2_kernel': [0.0, 0.0, 0.0, 0.0],
#           'optimizer': 'Nadam',
#           'loss': 'mse',
#           'metrics': ['mse'],
#           'batch_size': 100,
#           'max_epoch': 1000,
#           'eval_after_n_epochs': 10,
#           'data_type': 'choice_probabilities',
#           'model_directory': '',
#           'training_data_size': 'online'
#           }


def keras_model_generate(params = {}):

    # This returns a tensor
    inputs = keras.layers.Input(shape = (params['input_shape'], ))

    # Model hidden
    op = keras.layers.Dense(params['hidden_layers'][0],
                            activation = params['hidden_activations'][0],
                            kernel_regularizer = keras.regularizers.l1_l2(l1 = params['l1_kernel'][0],
                                                                          l2 = params['l2_kernel'][0]),
                            activity_regularizer = keras.regularizers.l1_l2(l1 = params['l1_activation'][0],
                                                                            l2 = params['l2_activation'][0])
                           )(inputs)

    for cnt in range(1, len(params['hidden_layers']), 1):
        op = keras.layers.Dense(params['hidden_layers'][cnt],
                                activation = params['hidden_activations'][cnt],
                                kernel_regularizer = keras.regularizers.l1_l2(l1 = params['l1_kernel'][cnt],
                                                                          l2 = params['l2_kernel'][cnt]),
                                activity_regularizer = keras.regularizers.l1_l2(l1 = params['l1_activation'][cnt],
                                                                            l2 = params['l2_activation'][cnt]))(op)

    # Model output
    outputs = keras.layers.Dense(params['output_shape'], params['output_activation'])(op)

    # Make model
    model = keras.models.Model(inputs = inputs, outputs = outputs)
    model.compile(
                  optimizer = model_params['optimizer'],
                  loss = model_params['loss'],
                  metrics = model_params['metrics']
                  )

    return model
