#import ddm_data_simulation as ddm_sim
import scipy as scp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import multiprocessing as mp
from  multiprocessing import Process
from  multiprocessing import Pool
import psutil
import pickle
import os
import time
import sys
import argparse

import kde_training_utilities as kde_util
import kde_class as kde

if __name__ == "__main__":
    # Interfact ----
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--machine",
                     type = str,
                     default = 'ccv')
    CLI.add_argument('--method',
                     type = str,
                     default = 'ddm')
    CLI.add_argument('--simfolder',
                     type = str,
                     default = 'base_simulations')
    CLI.add_argument('--fileprefix',
                     type = str,
                     default = 'ddm_base_simulations')
    CLI.add_argument('--fileid',
                     type = str,
                     default = 'TEST')
    CLI.add_argument('--outfolder',
                     type = str,
                     default = 'train_test')
    CLI.add_argument('--nbyparam',
                     type = int,
                     default = 1000)
    CLI.add_argument('--mixture',
                     nargs = '*',
                     type = float,
                     default = [0.8, 0.1, 0.1])
    CLI.add_argument('--nproc',
                    type = int,
                    default = 8)
    CLI.add_argument('--analytic',
                     type = int,
                     default = 0)
    
    args = CLI.parse_args()
    print(args)
    
    # Specify base simulation folder ------
    if args.machine == 'x7':
        method_params = pickle.load(open("/media/data_cifs/afengler/git_repos/nn_likelihoods/kde_stats.pickle",
                                         "rb"))[args.method]
        method_folder = method_params['method_folder_x7']

    if args.machine == 'ccv':
        method_params = pickle.load(open("/users/afengler/git_repos/nn_likelihoods/kde_stats.pickle", 
                                         "rb"))[args.method]
        method_folder = method_params['method_folder']
    
    if args.machine == 'home':
        method_parms = pickle.load(open("/Users/afengler/OneDrive/git_repos/nn_likelihoods/kde_stats.pickle",
                                       "rb"))[args.method]
        method_folder = method_params['method_folder_home']
    
    # Speficy names of process parameters
    process_params = method_params['param_names'] + method_params['boundary_param_names']
    
    # Make output folder if it doesn't exist
    if not os.path.isdir(method_folder + args.outfolder + '/'):
        os.mkdir(method_folder + args.outfolder + '/')
    
# STANDARD VERSION ----------------------------------------------------------------------------------------
    
    # Main function 
    start_time = time.time()
    kde_util.kde_from_simulations_fast_parallel(base_simulation_folder = method_folder + args.simfolder,
                                                file_name_prefix = args.fileprefix,
                                                file_id = args.fileid,
                                                target_folder = method_folder + args.outfolder,
                                                n_by_param = args.nbyparam,
                                                mixture_p = args.mixture,
                                                process_params = process_params,
                                                print_info = False,
                                                n_processes= args.nproc,
                                                analytic = args.analytic)
    
    end_time = time.time()
    exec_time = end_time - start_time
    print('Time elapsed: ', exec_time)

#-----------------------------------------------------------------------------------------------------------

# UNUSED --------------------------
    # LBA
#     process_params = ['v_0', 'v_1', 'A', 'b', 's', 'ndt']
#     files_ = pickle.load(open(base_simulation_folder + 'keep_files.pickle', 'rb'))

#     # DDM NDT
#     process_params = ['v', 'a', 'w', 'ndt']
#     files_ = pickle.load(open(base_simulation_folder + 'keep_files.pickle', 'rb'))

    # DDM ANGLE NDT
    #process_params = ['v', 'a', 'w', 'ndt', 'theta']
    #files_ = pickle.load(open(base_simulation_folder + 'keep_files.pickle', 'rb'))

   # print(mp.get_all_start_methods())
    
    
# # ALTERNATIVE VERSION

# # We should be able to parallelize this !
    
#     # Parallel
#     if args.nproc == 'all':
#         n_cpus = psutil.cpu_count(logical = False)
#     else:
#         n_cpus = args.nproc

#     print('Number of cpus: ')
#     print(n_cpus)
    
#     file_ = pickle.load(open( method_folder + args.simfolder + '/' + args.fileprefix + '_' + str(args.fileid) + '.pickle', 'rb' ) )
    
#     stat_ = pickle.load(open( method_folder + args.simfolder + '/simulator_statistics' + '_' + str(args.fileid) + '.pickle', 'rb' ) )
   
#     # Initializations
#     n_kde = int(args.nbyparam * args.mixture[0])
#     n_unif_down = int(args.nbyparam * args.mixture[1])
#     n_unif_up = int(args.nbyparam * args.mixture[2])
#     n_kde = n_kde + (args.nbyparam - n_kde - n_unif_up - n_unif_down) # correct n_kde if sum != args.nbyparam
    
#     # Add possible choices to file_[2] which is the meta data for the simulator (expected when loaded the kde class)
    
#     # TODO: THIS INFORMATION SHOULD BE INCLUDED AS META-DATA INTO THE BASE SIMULATOIN FILES
#     file_[2]['possible_choices'] = np.unique([-1, 1])
#     #file_[2]['possible_choices'] = np.unique(file_[1][0, :, 1])
#     file_[2]['possible_choices'].sort()

#     # CONTINUE HERE   
#     # Preparation loop --------------------------------------------------------------------
#     #s_id_kde = np.sum(stat_['keep_file']) * (n_unif_down + n_unif_up)
#     cnt = 0
#     starmap_iterator = ()
#     tmp_sim_data_ok = 0
#     results = []
#     for i in range(file_[1].shape[0]):
#         if stat_['keep_file'][i]:
            
#             # Don't remember what this part is doing....
#             if tmp_sim_data_ok:
#                 pass
#             else:
#                 tmp_sim_data = file_[1][i]
#                 tmp_sim_data_ok = 1
                
#             lb = cnt * (n_unif_down + n_unif_up + n_kde)

#             # Allocate to starmap tuple for mixture component 3
#             if args.analytic:
#                 starmap_iterator += ((file_[1][i, :, :].copy(), file_[0][i, :].copy(), file_[2].copy(), n_kde, n_unif_up, n_unif_down, cnt), )
#             else:
#                 starmap_iterator += ((file_[1][i, :, :], file_[2], n_kde, n_unif_up, n_unif_down, cnt), ) 
#                 #starmap_iterator += ((n_kde, n_unif_up, n_unif_down, cnt), )
#             # alternative
#             # tmp = i
#             # starmap_iterator += ((tmp), )
            
#             cnt += 1
#             if (cnt % 100 == 0) or (i == file_[1].shape[0] - 1):
#                 with Pool(processes = n_cpus, maxtasksperchild = 200) as pool:
#                     results.append(np.array(pool.starmap(kde_util.make_kde_data, starmap_iterator)).reshape((-1, 3)))   #.reshape((-1, 3))
#                     #result = pool.starmap(make_kde_data, starmap_iterator)
#                 starmap_iterator = ()
#                 print(i, 'arguments generated')
     
#     # Make dataframe to save
#     # Initialize dataframe
    
#     my_columns = process_params + ['rt', 'choice', 'log_l']
#     data = pd.DataFrame(np.zeros((np.sum(stat_['keep_file']) * args.nbyparam, len(my_columns))),
#                         columns = my_columns)    
    
#     #data.values[: , -3:] = result.reshape((-1, 3))
    
#     data.values[:, -3:] = np.concatenate(results)
#     # Filling in training data frame ---------------------------------------------------
#     cnt = 0
#     tmp_sim_data_ok = 0
#     for i in range(file_[1].shape[0]):
#         if stat_['keep_file'][i]:
            
#             # Don't remember what this part is doing....
#             if tmp_sim_data_ok:
#                 pass
#             else:
#                 tmp_sim_data = file_[1][i]
#                 tmp_sim_data_ok = 1
                
#             lb = cnt * (n_unif_down + n_unif_up + n_kde)

#             # Make empty dataframe of appropriate size
#             p_cnt = 0
            
#             for param in process_params:
#                 data.iloc[(lb):(lb + n_unif_down + n_unif_up + n_kde), my_columns.index(param)] = file_[0][i, p_cnt]
#                 p_cnt += 1
                
#             cnt += 1
#     # ----------------------------------------------------------------------------------

#     # Store data
#     print('writing data to file: ', method_folder + args.outfolder + '/data_' + str(args.fileid) + '.pickle')
#     pickle.dump(data.values, open(method_folder + args.outfolder + '/data_' + str(args.fileid) + '.pickle', 'wb'), protocol = 4)
    
#     # Write metafile if it doesn't exist already
#     # Hack for now: Just copy one of the base simulations files over
    
#     if os.path.isfile(method_folder + args.outfolder + '/meta_data.pickle'):
#         pass
#     else:
#         pickle.dump(tmp_sim_data, open(method_folder + args.outfolder + '/meta_data.pickle', 'wb') )

#     #return 0 #data
                                       