# Load packages
#import tensorflow as tf
#from tensorflow import keras
import cython
import numpy as np
import pickle  

# Activations
def relu(x):
    return np.maximum(0, x)

def linear(x):
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(- x))

# def tanh(x):
#     return (2 / (1 + np.exp(- 2 * x))) - 1

# Function to extract network architecture 

class mlp_target():
    def __init__(self, weights, biases, activations, n_datapoints):
        self.n_layers = len(weights)
        self.weights = weights
        self.biases = biases
        self.activations = activations
        self.activation_fns = {"relu":relu, "linear":linear, 'sigmoid':sigmoid, "tanh":np.tanh}
        self.n_datapoints = n_datapoints
        self.get_intermediate_dims()

    def get_intermediate_dims(self):
        self.intermediate_dims = []
        for i in range(self.n_layers - 1):
            self.intermediate_dims.append(np.zeros((self.n_datapoints, self.weights[i].shape[1])))

# Function to perform forward pass given architecture
# TD optimize this function as much as possible (need lakshmis advice for this....)
    def predict(self, x):
        
        self.intermediate_dims[0][:, :] = self.activation_fns[self.activations[0]](
            np.dot(x, self.weights[0]) + self.biases[0])
        
        i = 1
        while i < (self.n_layers - 1):
            self.intermediate_dims[i][:, :] = self.activation_fns[self.activations[i]](
                np.dot(self.intermediate_dims[i - 1], self.weights[i]) + self.biases[i])
            i += 1
            #print('x shape', x.shape)
        return self.activation_fns[self.activations[i]](np.dot(self.intermediate_dims[i - 1], self.weights[i]) + self.biases[i])


# TD NOT YET FASTEST POSSIBLE .... THERE IS ROOM HERE