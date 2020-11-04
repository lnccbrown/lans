import ddm_data_simulation as ddm_sim
import numpy as np
import pandas as pd
import multiprocessing as mp
import pickle
import os

import kde_training_utilities as kde_util
import kde_class as kde

part = 0
print("part {} of 10".format(part))

if __name__ == "__main__":
	# PICK
	base_simulation_folder = '/users/afengler/data/kde/ddm/base_simulations_20000/'
	target_folder = '/users/afengler/data/tony/kde/ddm/train_test_data_fcn/'

	params = ["v", "a", "w"]
	data_cols = ["rt", "choice"]
	n_sim_per_kde = 8000 # take only the first half for memory reasons
	files = os.listdir(base_simulation_folder)
	n_sim = len(files) - 2
	total_rows = n_sim_per_kde * n_sim

	if not os.path.isdir(target_folder):
		os.mkdir(target_folder)

	ix = slice((part * total_rows // 10), (part + 1) * total_rows // 10)
	print(ix)
	out = np.zeros((total_rows // 10, len(params) + len(data_cols)))
	out = pd.DataFrame(out, columns=params + data_cols)
	y = np.zeros((total_rows // 10, 1))

	for i, file in enumerate(files[ix]):
		if i % 1000 == 0:
			print(i)
		data = np.array(pickle.load(open(base_simulation_folder + file, "rb")))
		tmp_rt = data[0][:n_sim_per_kde]
		tmp_choice = data[1][:n_sim_per_kde]
		ix = np.arange(i * n_sim_per_kde, (i+1) * n_sim_per_kde)
		out.loc[ix, "rt"] = tmp_rt.ravel()
		out.loc[ix, "choice"] = tmp_choice.ravel()
		for param in params:
			out.loc[ix, param] = data[2][param]

		tmp_kde = kde.logkde(data)
		y[ix] = tmp_kde.kde_eval((tmp_rt, tmp_choice))

	with open(target_folder + "train_features{}.pickle".format(part)) as f:
		pickle.dump(out, f)

	with open(target_folder + "train_labels{}.pickle".format(part)) as f:
		pickle.dump(y, f)
