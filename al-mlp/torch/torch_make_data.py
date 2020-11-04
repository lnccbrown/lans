import numpy as np
import pandas as pd
import pickle
import os
from itertools import product
from cddm_data_simulation import ddm_simulate
from cdwiener import array_fptd

def generate_input_grid(param_grid_size = 100, rt_grid_size=100):
    grid_v = np.random.uniform(-1, 1, param_grid_size)
    grid_a = np.random.uniform(0, 1.5, param_grid_size)
    grid_z = np.random.uniform(0, 1, param_grid_size)
    grid_rt = np.random.uniform(0, 5, rt_grid_size)
    grid_choice = [-1, 1]
    return np.array(list(product(grid_v, grid_a, grid_z, grid_rt, grid_choice)))

def generate_random_grid(size, v_bound = [-1, 1], a_bound = [.3, 3], w_bound = [.3, .7], t_params = [.75, 1.5]):
    v = np.random.uniform(low = v_bound[0], high = v_bound[1], size = size)
    a = np.random.uniform(low = a_bound[0], high = a_bound[1], size = size)
    w = np.random.uniform(low = w_bound[0], high = w_bound[1], size = size)
    t = np.random.gamma(t_params[0], t_params[1], size=size)
    choice = np.random.choice([-1, 1], size)

    data = pd.DataFrame({"v": v, "a": a, "w": w, "rt": t, "choice": choice})
    return data

def make_data(folder):
    data = generate_random_grid(10000000)
    train_id = np.random.choice(10000000)
    with open(folder + "train_data" + str(train_id,) + ".pickle", "wb") as f:
        pickle.dump(data, f)

def combine_data(folder):
    full_data = []
    files = os.listdir(folder)
    if "train_features.pickle" in files:
        print("full dataset already created!")
        return
    for dataset in files:
        with open(folder + dataset, "rb") as f:
            data = pickle.load(f)
            full_data.append(data)
    train_features = pd.concat(full_data)
    with open(folder + "train_features.pickle", "wb" ) as f:
        pickle.dump(train_features, f, protocol = 4)

def generate_labels(folder):
    with open(folder + "train_features.pickle", "rb") as f:
        dataset = pickle.load(f)
    dataset = np.array(dataset)
    labels = array_fptd(dataset[:, 3] * dataset[:, 4], dataset[:, 0], dataset[:, 1], dataset[:, 2], 1e-29)
    with open(folder + "train_labels.pickle", "wb") as f:
        pickle.dump(labels, f)

combine_data("/users/afengler/data/navarro_fuss/train_test_data_sim/")