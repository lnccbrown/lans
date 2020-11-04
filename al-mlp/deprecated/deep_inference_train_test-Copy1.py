import numpy as np
import pandas as pd
import cddm_data_simulation as cd
import boundary_functions as bf


n_sims = 200000
n_samples_per_sim = 2000

method = "full"

if method == "ddm":
    dgp = cd.ddm_flexbound_simulate
    boundary = bf.constant
#     custom_objects = {"huber_loss": tf.losses.huber_loss}
#     fcn_path = "/users/afengler/data/tony/kde/ddm/keras_models/\
# deep_inference08_12_19_11_15_06/model.h5"
#    fcn_custom_objects = {"heteroscedastic_loss": tf.losses.huber_loss}
    param_names = ["v", "a", "w"]
    boundary_params = []
    param_bounds = np.array([[-2, .6, .3], [2, 1.5, .7]])
    boundary_param_bounds = []
elif method == "linear_collapse":
    dgp = cd.ddm_flexbound_simulate
    boundary = bf.linear_collapse
    param_names = ["v", "a", "w"]
    boundary_param_names = ["node", "theta"]
    param_bounds = np.array([[-2, .6, .3], [2, 1.5, .7]])
    boundary_param_bounds = np.array([[1, 0], [2, 1.37]])
elif method == "ornstein":
    dgp = cd.ornstein_uhlenbeck
    boundary = bf.constant
    param_names = ["v", "a", "w", "g"]
    boundary_params = []
    boundary_param_bounds = []
    param_bounds = np.array([[-2, .6, .3, -1], [2, 1.5, .7, 1]])
elif method == "full":
    output_folder = "/users/afengler/data/tony/kde/full_ddm/train_test_data_fcn/"
    dgp = cd.full_ddm
    boundary = bf.constant
    param_names = ["v", "a", "w", "dw", "sdv"]
    boundary_params = []
    boundary_param_bounds = []
    param_bounds = np.array([[-2, .6, .3, 0, 0], [2, 1.5, .7, .1, .5]])

labels = np.zeros((n_sims, len(param_names)))
features = np.zeros((n_sims, n_samples_per_sim, 2))

labels = np.random.uniform(param_bounds[0], param_bounds[1], size=(n_sims, len(param_names)))

for i in range(n_sims):
    if i % 10000 == 0:
        print(i)
    param_dict_tmp = dict(zip(param_names, labels[i]))
    rts, choices, _ = dgp(**param_dict_tmp, n_samples=n_samples_per_sim, boundary_fun=boundary, delta_t=.01)
    features[i] = np.concatenate([rts, choices], axis=1)

pickle.dump(features, open(output_folder + "train_features.pickle", "wb"), protocol=4)
pickle.dump(labels, open(output_folder + "train_labels.pickle", "wb"), protocol=4)
