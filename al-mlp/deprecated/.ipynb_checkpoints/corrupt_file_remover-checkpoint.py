import pickle
import os

folder_ = '/users/afengler/data/kde/weibull/base_simulations/'
file_list = os.listdir(folder_)

corrupt_files_cnt = 0
for file_ in file_list:
    if os.path.getsize(folder_ + file_) == 0:
        os.remove(folder_ + file_)
        corrupt_files_cnt += 1

print(corrupt_files_cnt)
