import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
#from cdwiener import array_fptd
import os
import pandas as pd
import time
import psutil
import argparse
from datetime import datetime
import pickle
import yaml
import keras_to_numpy as ktnp

#from kde_training_utilities import kde_load_data
from kde_training_utilities import kde_load_data_new
#from kde_training_utilities import kde_make_train_test_split

if __name__ == "__main__":
    
    # Interface ----
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--machine",
                     type = str,
                     default = 'ccv')
    CLI.add_argument('--method',
                     type = str,
                     default = 'ddm')
    CLI.add_argument('--datafolder',
                     type = str,
                     default = 'base_simulations')
    CLI.add_argument('--nfiles',
                     type = int,
                     default = 100)
    CLI.add_argument('--maxidfiles',
                     type = int,
                     default = 100)
    CLI.add_argument('--nbydataset',
                     type = int,
                     default = 10000000)
    CLI.add_argument('--warmstart',
                     type = int,
                     default = 0)
    
    args = CLI.parse_args()
    print(args)

    # CHOOSE ---------
    method = args.method
    # method = "weibull_cdf" # ddm, linear_collapse, ornstein, full, lba
    warm_start = args.warmstart
    n_training_datasets_to_load = args.nfiles
    maxidfiles = args.maxidfiles
    machine = args.machine
    data_folder = args.datafolder
    # ----------------

    # INITIALIZATIONS ----------------------------------------------------------------
    if machine == 'x7':
        stats = pickle.load(open("/media/data_cifs/afengler/git_repos/nn_likelihoods/kde_stats.pickle", "rb"))[method]
        dnn_params = yaml.load(open("/meia/data_cifs/afengler/git_repos/nn_likelihoods/hyperparameters.yaml"))
    else:
        stats = pickle.load(open("/users/afengler/git_repos/nn_likelihoods/kde_stats.pickle", "rb"))[method]
        dnn_params = yaml.load(open("/users/afengler/git_repos/nn_likelihoods/hyperparameters.yaml"))


    if machine == 'x7':
        #data_folder = stats["data_folder_x7"]
        model_path = stats["model_folder_x7"]
    else:
        #data_folder = stats["data_folder"]
        model_path = stats["model_folder"]

    if not warm_start:
        model_path += dnn_params["model_type"] + "_{}_".format(method) + datetime.now().strftime('%m_%d_%y_%H_%M_%S') + "/"

    print('if it does not exist, make model path')

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Copy hyperparameter setup into model path
    if machine == 'x7':
        os.system("cp {} {}".format("/media/data_cifs/afengler/git_repos/nn_likelihoods/hyperparameters.yaml", model_path))
    else:
        os.system("cp {} {}".format("/users/afengler/git_repos/nn_likelihoods/hyperparameters.yaml", model_path))

    # set up gpu to use
    if machine == 'x7':
        os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = dnn_params['gpu_x7'] 

    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())

    # Load the training data
    print('loading data.... ')

    # X, y, X_val, y_val = kde_load_data(folder = data_folder, 
    #                                    return_log = True, # Dont take log if you want to train on actual likelihoods
    #                                    prelog_cutoff = 1e-7 # cut out data with likelihood lower than 1e-7
    #                                   )
    
    X, y = kde_load_data_new(path = data_folder,
                             file_id_list = list(1 + np.random.choice(maxidfiles, replace = False, size = n_training_datasets_to_load)),
                             # file_id_list = [i for i in range(1, n_training_datasets_to_load + 1, 1)],
                             return_log = True,
                             prelog_cutoff_low = 1e-7,
                             prelog_cutoff_high = 100)


    # --------------------------------------------------------------------------------

    # MAKE MODEL ---------------------------------------------------------------------
    print('Setting up keras model')

    if not warm_start:
        input_shape = X.shape[1]
        model = keras.Sequential()

        for i in range(len(dnn_params['hidden_layers'])):
            if i == 0:
                model.add(keras.layers.Dense(units = dnn_params["hidden_layers"][i], 
                                             activation = dnn_params["hidden_activations"][i], 
                                             input_dim = input_shape))
            else:
                model.add(keras.layers.Dense(units = dnn_params["hidden_layers"][i],
                                             activation = dnn_params["hidden_activations"][i]))

        # Write model specification to yaml file        
        spec = model.to_yaml()
        open(model_path + "model_spec.yaml", "w").write(spec)


        print('STRUCTURE OF GENERATED MODEL: ....')
        print(model.summary())

        if machine == 'x7':

            if dnn_params['loss'] == 'huber':
                model.compile(loss = tf.losses.huber_loss, 
                              optimizer = "adam", 
                              metrics = ["mse"])
        
        if machine == 'ccv':

            if dnn_params['loss'] == 'huber':
                model.compile(loss = tf.keras.losses.Huber(),
                              optimizer = "adam",
                              metrics = ["mse"])

        if dnn_params['loss'] == 'mse':
            model.compile(loss = 'mse', 
                          optimizer = "adam", 
                          metrics = ["mse"])
    if warm_start:
        # Returns a compiled model identical to the previous one
        if machine == 'x7':
            model_paths = yaml.load(open("/media/data_cifs/afengler/git_repos/nn_likelihoods/model_paths_x7.yaml"))
        if machine == 'ccv':
            model_paths = yaml.load(open("/users/afengler/git_repos/nn_likelihoods/model_paths.yaml"))

        model_path = model_paths[method +  '_' + str(n_training_datasets_to_load)]
        model = load_model(model_path + 'model_final.h5', custom_objects = {"huber_loss": tf.losses.huber_loss})

    # ---------------------------------------------------------------------------

    # FIT MODEL -----------------------------------------------------------------
    print('Starting to fit model.....')

    # Define callbacks
    ckpt_filename = model_path + "model_ckpt.h5"

    checkpoint = keras.callbacks.ModelCheckpoint(ckpt_filename, 
                                                 monitor = 'val_loss', 
                                                 verbose = 1, 
                                                 save_best_only = False)

    earlystopping = keras.callbacks.EarlyStopping(monitor = 'val_loss', 
                                                  min_delta = 0, 
                                                  verbose = 1, 
                                                  patience = 2)

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', 
                                                  factor = 0.1,
                                                  patience = 1, 
                                                  verbose = 1,
                                                  min_delta = 0.0001,
                                                  min_lr = 0.0000001)

    history = model.fit(X, y,  
                        validation_split = 0.01,
                        epochs = dnn_params["n_epochs"],
                        batch_size = dnn_params["batch_size"], 
                        shuffle = True,
                        callbacks = [checkpoint, reduce_lr, earlystopping], 
                        verbose = 2,
                        #validation_data = (X_val, y_val)
                       )
    # ---------------------------------------------------------------------------

    # SAVING --------------------------------------------------------------------
    print('Saving model and relevant data...')
    # Log of training output
    pd.DataFrame(history.history).to_csv(model_path + "training_history.csv")

    # Save Model
    model.save(model_path + "model_final.h5")

    # Extract model architecture as numpy arrays and save in model path
    __, ___, ____, = ktnp.extract_architecture(model, save = True, save_path = model_path)

    # Update model paths in model_path.yaml
    if machine == 'x7':
        if not warm_start:
            model_paths = yaml.load(open("model_paths_x7.yaml"))
            model_paths[method + '_' + str(n_training_datasets_to_load)] = model_path
            yaml.dump(model_paths, open("model_paths_x7.yaml", "w"))
    if machine == 'ccv':
        if not warm_start:
            model_paths = yaml.load(open("model_paths.yaml"))
            model_paths[method + '_' + str(n_training_datasets_to_load)] = model_path
            yaml.dump(model_paths, open("model_paths.yaml", "w"))
    # ----------------------------------------------------------------------------