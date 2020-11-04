# My own code
import kde_training_utilities as kde_utils
import glob

if __name__ == "__main__":
    # CHOOSE FOLDER
    machine = 'x7'
    if machine == 'ccv':
        # CCV 
        my_folder = '/users/afengler/data/kde/weibull_cdf/train_test_data_ndt_20000/'
    if machine == 'x7':
        # X7
        my_folder = '/media/data_cifs/afengler/data/kde/weibull_cdf/train_test_data_ndt_20000/'
    
    print('Folder used:', my_folder)
    
    # List of data files to process
    file_list = glob.glob(my_folder + '/data_*')
    n_files_out = 20
    kde_utils.kde_make_train_data(path = my_folder,
                                  n_files_out = n_files_out,
                                  file_in_list = file_list)
    
    file_id_list = [i for i in range(n_files_out)]
    kde_utils.kde_load_data_new(path = my_folder,
                                file_id_list = file_id_list,
                                prelog_cutoff = 1e-7)
     
#     kde_utils.kde_make_train_test_split(folder = my_folder, 
#                                         p_train = 0.99)
                                        
#         def kde_make_train_test_split(path = '',
#                               p_train = 0.8,
#                               n_files_out = 10,
#                               file_in_list = 'all'):