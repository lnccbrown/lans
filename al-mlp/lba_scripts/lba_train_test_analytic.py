# My own code
import make_data_lba as mdlba

if __name__ == "__main__":
    # CHOOSE FOLDER
    # CCV 
    my_folder = '/users/afengler/data/lba_analytic/train_test_data_kde_imit/'
    # X7
    #my_folder = '/media/data_cifs/afengler/data/lba_analytic/train_test_data_kde_imit/'
    
    my_dat = mdlba.make_data_rt_choice(v_range = [1, 2],
                                       A_range = [0, 1],
                                       b_range = [1.5, 3],
                                       s_range = [0.1, 0.2],
                                       n_choices = 2,
                                       n_samples = 1000000,
                                       eps = 1e-16,
                                       target_folder = my_folder,
                                       write_to_file = True,
                                       print_detailed_cnt = True,
                                       mixture_p = [0.8, 0.1, 0.1],
                                       n_by_param = 1000)