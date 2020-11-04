# Generate some data
import numpy as np

def make_data():
    x_0 = np.arange(-np.pi*2, np.pi*2, 0.001)
    x_1 = np.random.choice([0,1], len(x_0))
    y = np.zeros((len(x_1),1))
    y[x_1 == 0] = np.asmatrix(np.sin(x_0[x_1 == 0]) + np.random.normal(loc = 0, scale = 0.1, size = np.sum(x_1 == 0))).T
    y[x_1 == 1] = np.asmatrix(np.cos(x_0[x_1 == 1]) + np.random.normal(loc = 0, scale = 0.1, size = np.sum(x_1 == 1))).T

    features = np.stack((x_0, x_1), axis = 1)
    labels = y
    return features , labels

# Making training and test sets,
def train_test_split(features, labels, p = 0.8, write_to_file = False):
    filter_ = np.random.choice([0,1], p = [p, 1 - p], size = features.shape[0])
    train_features = features[filter_ == 0,:]
    train_labels = labels[filter_ == 0,:]
    test_features = features[filter_ == 1,:]
    test_labels = labels[filter_ == 1,:]

    train_features_dict = dict()
    test_features_dict = dict()

    for i in range(0, features.shape[1], 1):
        train_features_dict['x_' + str(i)] = train_features[:, i]
        test_features_dict['x_' + str(i)] = test_features[:, i]

    return train_features_dict, train_labels, test_features_dict, test_labels
