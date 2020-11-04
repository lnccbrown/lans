# Load packages
import tensorflow as tf
import pandas as pd
from tensorflow import keras
import numpy as np
import pandas as pd
import os
import scipy as scp
import scipy.stats as scps
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
import time
import scipy.optimize as scp_opt

import samplers.diagnostics as diag_

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

if __name__ == "__main__":

    # Make model
    model = keras.Sequential()
    model.add(keras.layers.Dense(units = 256, 
                                 activation = 'tanh', 
                                 input_dim = 6))
    model.add(keras.layers.Reshape([16, 16, 1]))
    model.add(keras.layers.Conv2D(filters = 32, 
                                  kernel_size = (9, 9), 
                                  strides = 1, 
                                  padding = 'valid',
                                  activation= 'tanh',
                                  data_format = 'channels_last'
                                  ))
    model.add(keras.layers.Conv2D(filters = 64,
                                  kernel_size = (6, 6),
                                  strides = 2,
                                  padding = 'valid',
                                  activation='tanh',
                                  data_format = 'channels_last'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units = 1,
                                 activation = None))
    model.compile(loss = tf.losses.mse,
                  optimizer = "adam")

    X = np.random.uniform(low = -1.0, high = 1.0, size = (10000, 6))
    mult_vec = np.random.normal(size = (6, 1))
    y = np.matmul(X[:, :], mult_vec)

    X = np.float32(X)
    y = np.float32(y)

    bounds = [(-1.0, 1.0) for i in  range(6)]

    now = time.time()
    out = model.fit(X, y, 
                    validation_split = 0.1, 
                    epochs = 25, 
                    batch_size = 1000, 
                    shuffle = True, 
                    verbose = 2)

    def loss_fun(x):
        return np.square(model.predict(np.reshape(x, (1, 6))) - y[0])[0, 0]

    scp_opt.differential_evolution(loss_fun,
                                   bounds = bounds,
                                   popsize = 10, 
                                   maxiter = 10,
                                   disp = True,
                                   polish = False)