import numpy as np
import scipy as scp
import os
import pandas as pd
import matplotlib as plt

# Local imports
import ddm_data_simulation
import make_data_wfpt as mdw
import dnnregressor_train_eval_keras as dnnk

class ddm_mle_estimator:
    def __init__(self):
        self.ddm_sim_params = dict({'v': 1,
                                    'a': 1,
                                    'w': 0.5,
                                    's': 1,
                                    'delta_t': 0.001,
                                    'max_t': 20,
                                    'sample_size': 1000}
                                    )



        self.parameter_bounds = dict({'v': [-2, 2],
                                      'a': [0.3, 4],
                                      'w': [0.001, 0.999]
                                      }
                                      )

        # Genetic algorithm related
        self.gen_alg_params = dict({'population_size': 40,
                                    'mutation_probability': 0.1,
                                    'print_steps': True,
                                    'steps': 100}
                                    )

        self.gen_alg_population = []
        self.gen_alg_population_record = np.zeros((self.gen_alg_params['steps'], self.gen_alg_params['population_size'], 3))
        self.gen_alg_fitness = []
        self.gen_alg_fitness_record = np.zeros((self.gen_alg_params['steps'], self.gen_alg_params['population_size']))

        # Grid search related
        self.grid_search_parameters = dict({'precision_v': 0.2,
                                            'precision_a': 0.2,
                                            'precision_w': 0.1})


        # Data and miscellaneous
        self.data = []

        # self.model_directory = 'none'    # directory for model that can predict likelihoods (to be passed as parameter to a function that restores the model from tensorflow for example)
        # self.hyper_parameter_file = ''

        self.model_data = {'model_path': '',
                           'ckpt_path': ''
                           }

        # Meta
        self.meta_parameters = dict({'model': 'navarro',
                                     'datatype': 'choice_probabilities',
                                     'optimizer': 'genetic'})

    def get_keras_model(self):
        model = keras.models.load_model(self.model_data['model_path'])
        model.load_weights(self.model_data['ckpt_path'])
        self.model = model
        return model

    def make_data(self):
        rts, choices = ddm_data_simulation.ddm_simulate_rts(v = self.ddm_sim_params['v'],
                                                            a = self.ddm_sim_params['a'] / 2, # we divide by 2 => correct conversion when simulating with euler narayujan
                                                            w = self.ddm_sim_params['w'],
                                                            s = self.ddm_sim_params['s'],
                                                            delta_t = self.ddm_sim_params['delta_t'],
                                                            max_t = self.ddm_sim_params['max_t'],
                                                            n_samples = self.ddm_sim_params['sample_size']
                                                            )

       self.p_lower_barrier = np.sum(choices > 0) / len(choices)
       self.data = np.multiply(rts, choices)

    # functions related to genetic algorithm setup
    def make_pop_init(self):
        population = pd.DataFrame(np.zeros((self.gen_alg_params['population_size'], 3)), columns = ['v', 'a', 'w'])
        population.loc[:, list(self.ddm_sim_params.keys())[0]] = np.random.uniform(low = self.parameter_bounds['v'][0], high = self.parameter_bounds['v'][1], size = self.gen_alg_params['population_size'])
        population.loc[:, list(self.ddm_sim_params.keys())[1]] = np.random.uniform(low = self.parameter_bounds['a'][0], high = self.parameter_bounds['a'][1], size = self.gen_alg_params['population_size'])
        population.loc[:, list(self.ddm_sim_params.keys())[2]] = np.random.uniform(low = self.parameter_bounds['w'][0], high = self.parameter_bounds['w'][1], size = self.gen_alg_params['population_size'])
        self.gen_alg_population = population.copy()

    def make_selection_probabilities(self, n_keep = 0):
        probabilities = np.zeros((n_keep,))
        denom = sum(np.arange(1,n_keep + 1, 1))
        for n in range(1, n_keep + 1, 1):
            num = n_keep - n + 1
            probabilities[n-1] = num / denom
        return probabilities

    def population_fitness_rt_choice_nf(self):
        fitness = np.zeros((self.gen_alg_params['population_size'], ))
        for i in range(0, self.gen_alg_params['population_size'], 1):
            fitness[i] = self.loglik_choice_rt_nf(eps = 1e-29,
                                               v = self.gen_alg_population.loc[i, 'v'],
                                               a = self.gen_alg_population.loc[i, 'a'],
                                               w = self.gen_alg_population.loc[i, 'w']
                                               )
        self.gen_alg_fitness = fitness

    def population_fitness_rt_choice_dnn(self):
        fitness = np.zeros((self.gen_alg_params['population_size'], ))

        for i in range(0, self.gen_alg_params['population_size'], 1):
            data  = pd.DataFrame(np.zeros((len(self.data), 5)), columns = ['v','a', 'w', 'rt', 'nf_likelihood'])
            data.loc[:, 'v'] = self.gen_alg_population.loc[i, 'v']
            data.loc[:, 'a'] = self.gen_alg_population.loc[i, 'a']
            data.loc[:, 'w'] = self.gen_alg_population.loc[i, 'w']
            data.loc[:, 'rt'] = self.data.flatten()
            data.loc[:, 'nf_likelihood'] = 0.0

            features, labels, __, ___ = mdw.train_test_split_rt_choice(data = data,
                                                                       p_train = 1.0,
                                                                       write_to_file = False,
                                                                       from_file = False,
                                                                       backend = 'keras'
                                                                       )


            #self.features = features
            #self.labels = labels
            fitness[i] = np.sum(np.log(1e-29 + np.abs(self.model.predict(features))))
            # fitness[i] = np.sum(np.log(1e-29 + dnn_predictor.get_predictions(regressor = self.dnn_predictor,
            #                                                          features = features,
            #                                                          labels = labels)
            #                                                          )
            #                                                          )
            self.gen_alg_fitness = fitness

    def population_fitness_choice_p_nf(self):

        fitness = np.zeros((self.gen_alg_params['population_size'], ))

        for i in range(0, self.gen_alg_params['population_size'], 1):
            fitness[i] = self.loglik_choice_probability(v = self.gen_alg_population.loc[i, 'v'],
                                                        a = self.gen_alg_population.loc[i, 'a'],
                                                        w = self.gen_alg_population.loc[i, 'w']
                                                        )

        self.gen_alg_fitness = fitness

    def make_next_generation(self):
        pop_size = self.gen_alg_params['population_size']
        assert ((pop_size % 2) % 2) == 0, 'Population size divided by two should be an even integer'
        n_keep = pop_size // 2

        # Get indices of sorted fitness values for current generation
        # Allows us to shave of the top # (n_keep)
        fit_sort_id = list(reversed(np.argsort(self.gen_alg_fitness)))

        # Create pool to start from for next generation (best n_keep values)
        pop_pool_new_gen = self.gen_alg_population.loc[fit_sort_id[:n_keep], :].values.copy()
        selection_probabilities = self.make_selection_probabilities(n_keep = n_keep)

        # Get base for next gen
        indices = np.random.choice(np.arange(0, n_keep, 1), p = selection_probabilities, size = n_keep)
        print('indices: ', indices)
        #print('pop_pool_new_gen_going_in: ', pop_pool_new_gen)
        pop_pool_new_gen = pop_pool_new_gen[indices, :]
        #print('pop_pool_new_gen filtered: ', pop_pool_new_gen[indices, :])

        # Now creating full next generation
        new_gen = np.zeros((pop_size, len(self.parameter_bounds)))

        # Fill in upper half of new_gen array with new offspring
        cnt = 0
        while cnt < ((pop_size // 2) - 1):
            new_gen[cnt, :] = pop_pool_new_gen[cnt, :]
            cnt += 1
            new_gen[cnt, :] = pop_pool_new_gen[cnt, :]

            # randomly select indice
            indice = np.random.choice(np.arange(0, len(self.parameter_bounds), 1))
            weighting_beta = np.random.uniform(low = 0, high = 1)
            new_gen[(cnt - 1), indice] = ((1 - weighting_beta) * pop_pool_new_gen[cnt - 1, indice]) + (weighting_beta * pop_pool_new_gen[cnt, indice])
            new_gen[cnt, indice] = ((1 - weighting_beta) * pop_pool_new_gen[cnt, indice]) + (weighting_beta * pop_pool_new_gen[cnt - 1, indice])
            cnt += 1

        # Fill in lower half with old generations top values
        new_gen[cnt:, :] = pop_pool_new_gen
        #print('new gen: ', new_gen)
        # Include random mutations

        # Generate mutation locations
        mutations = np.reshape(np.random.choice([1,0], p = [self.gen_alg_params['mutation_probability'], 1 - self.gen_alg_params['mutation_probability']],
                               size = pop_size * len(self.parameter_bounds)),
                               newshape = (pop_size, len(self.parameter_bounds)))

        # Make iterable
        mutation_locations = zip(*np.where(mutations == 1))

        # Apply mutations
        my_keys = list(self.parameter_bounds.keys())
        for (i,j) in mutation_locations:
            new_gen[i, j] = np.random.uniform(low = self.parameter_bounds[my_keys[j]][0],
                                              high = self.parameter_bounds[my_keys[j]][1]
                                              )

        #print('new_gen_just_before_making_it_dataframe: ', new_gen)
        #print('new_gen_data_frame : ', pd.DataFrame(new_gen, columns = ['v', 'a', 'w']))
        self.gen_alg_population = pd.DataFrame(new_gen, columns = ['v', 'a', 'w'])

        if self.meta_parameters['model'] == 'navarro':
            self.population_fitness_rt_choice_nf()

        if self.meta_parameters['model'] == 'dnn':
            self.population_fitness_rt_choice_dnn()

    def make_fitness_stats(self):
        return np.mean(self.gen_alg_fitness), np.max(self.gen_alg_fitness)

    def run_gen_alg(self):
        self.make_pop_init()
        if self.meta_parameters['model'] == 'navarro':
            self.population_fitness_rt_choice_nf()

        #self.gen_alg_fitness_max = np.zeros((self.gen_alg_steps, ))
        #self.gen_alg_fitness_mean = np.zeros((self.gen_alg_steps, ))
        if self.meta_parameters['model'] == 'dnn':
            self.population_fitness_rt_choice_dnn()

        if self.meta_parameters['model'] == 'choice_probabilities':
            self.population_fitness_choice_p_nf()

        for step in range(0, self.gen_alg_params['steps'], 1):
            self.make_next_generation()
            print('cur_population: ', self.gen_alg_population)
            print('cur_fitness: ', self.gen_alg_fitness)
            #self.gen_alg_fitness_mean[step], self.gen_alg_fitness_max[step] = self.make_fitness_stats()

            # Record current population and fitness
            self.gen_alg_fitness_record[step] = self.gen_alg_fitness
            self.gen_alg_population_record[step] = self.gen_alg_population
            if self.gen_alg_params['print_steps']:
                print(step)

    def plot_gen_alg(self):
        plt.plot(arange(0, self.gen_alg_steps, 1), self.gen_alg_fitness_mean, 'g.')
        plt.plot(arange(0, self.gen_alg_steps, 1), self.gen_alg_fitness_mean, 'r.')
        plt.show()

    # compute log likelihood one set of data
    def loglik_choice_rt_nf(self,
                         eps = 1e-29,
                         v = 0,
                         a = 1,
                         w = 0.5):

        tmp = 0
        for t in self.data:
            tmp += np.log(mdw.fptd(t, v, a, w, eps))
        return tmp

    def d_kl_choice_probabiliy_navarro(self,
                                       eps = 1e-29,
                                       v = 0,
                                       a = 1,
                                       w = 0.5):

        p_navarro = mdw.choice_probabilities(v = v,
                                              a = a,
                                              w = w)
        p_data = self.p_lower_barrier

        d_kl = p_navarro * np.log(p_data) + (1 - p_navarro) * np.log(1 - p_data)

        return d_kl

    def loglik_choice_probability_nf(self,
                                      eps = 1e-29,
                                      v = 0,
                                      a = 1,
                                      w = 0.5):

        log_p_navarro = np.log(mdw.choice_probabilities(v = v,
                                                         a = a,
                                                         w = w)
                                                         )

        tmp = 0
        for c in np.sign(self.data):
            if c > 0:
                tmp += log_p_navarro
            else:
                tmp += log_p_navarro

        return tmp

    # Methods related to grid search
    def grid_search_make_grid(self):
        v_vals = np.arange(self.parameter_bounds['v'][0], self.parameter_bounds['v'][1], self.grid_search_parameters['precision_v'][0])
        a_vals = np.arange(self.parameter_bounds['a'][0], self.parameter_bounds['a'][1], self.grid_search_parameters['precision_a'][0])
        w_vals = np.arange(self.parameter_bounds['w'][0], self.parameter_bounds['w'][1], self.grid_search_parameters['precision_w'][0])

        grid = pd.DataFrame(np.zeros(len(v_vals)*len(a_vals)*len(w_vals), 3),
                            columns = ['v', 'a', 'w'])

        cnt = 0
        for v_tmp in v_vals:
            for a_tmp in a_vals:
                for w_tmp in w_vals:
                    grid.loc[cnt] = [v_tmp, a_tmp, w_tmp]
                    cnt += 1

        self.grid_search_grid = grid


    def run_grid_search(self):
        self.grid_search_make_grid()

        self.grid_search_results = self.grid_search_grid.copy()
        self.grid_search_results['loglik'] = 0
        self.grid_search_results['d_kl'] = 0

        for i in range(0, len(self.grid_search_grid['v']), 1):

            if self.meta_parameters['datatype'] == 'choice_rt':
                if self.meta_parameters['model'] == 'navarro':
                    self.grid_search_results.loc[i, 'loglik'] = self.loglik_choice_rt_nf(v = self.grid_search_grid.loc[i, 'v'],
                                                                                      a = self.grid_search_grid.loc[i, 'a'],
                                                                                      w = self.grid_search_grid.loc[i, 'w'])
                if self.meta_parameters['model'] == 'dnn':
                    #self.grid_search_results.loc[i, 'loglik'] =

            if self.meta_parameters['datatype'] == 'choice_probabilities':
                self.grid_search_results.loc[i, 'loglik'] = self.loglik_choice_probability_nf(v = self.grid_search_grid.loc[i, 'v'],
                                                                                                   a = self.grid_search_grid.loc[i, 'a'],
                                                                                                   w = self.grid_search_grid.loc[i, 'w'])

                self.grid_search_results['d_kl'] = self.d_kl_choice_probability_navarro(v = self.grid_search_grid.loc[i, 'v'],
                                                                                        a = self.grid_search_grid.loc[i, 'a'],
                                                                                        w = self.grid_search_grid.loc[i, 'w'])



# ---------------------------------------------------------------------------------------------------------------------------------------------------

# # Methods related to importing tensorflow model for likelihood
# def tf_estimator_hyperparameters(self):
#     hyper_params= pd.read_csv(self.model_directory + '/' + hyper_parameter_file,
#                               converters = {'hidden_units': eval,
#                                             'activations': eval}
#                              )
#     model_params = hyper_params.to_dict(orient = 'list')
#     for key in model_params.keys():
#         model_params[key] = model_params[key][0]
#
#     return model_params


# def initialize_dnn_predictor(self):
#     if self.model_directory == 'none':
#         print('please specify directory of your dnn_regressor model files!')
#     if self.model_directory != 'none':
#         # Get hyperparameters to feed into dnn_regressor model_input function
#         self.dnn_model_params = self.tf_estimator_hyperparameters()
#
#         # Make feature columns and add to
#         features = dict({'v': [],
#                         'a': [],
#                         'w': [],
#                         'rt': [],
#                         'choice': []}
#                         )
#         feature_columns = dnn_model_input.make_feature_columns_numeric(features = features)
#         self.dnn_model_params['feature_columns'] = feature_columns
#         self.dnn_predictor = dnn_predictor.get_dnnreg_predictor(model_directory = self.model_directory, params = self.dnn_model_params)
