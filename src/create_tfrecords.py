import os, tqdm, pickle, glob
import numpy as np
import matplotlib.pyplot as plt
from tf_data_handler import write_tfrecord
import config

def main(
        data_folder = '',
        experiment = '',
        tfrecord_dir = '',
        config=None
):
    data_prop = config.data_prop
    all_features, all_labels = [], []

    # get the listing of files
    data_files = glob.glob(os.path.join(
				data_folder,
				experiment))

    # collect all the samples
    for data_file in tqdm.tqdm(data_files):
	data = pickle.load(open(data_file,'rb'))
	all_features.extend(data[1].astype(np.float32))
	all_labels.extend(data[0].astype(np.float32))

    print('Finished reading data. Making splits')

    total_items = len(all_features)
    arr = np.arange(total_items)
    #np.random.shuffle(arr)

    all_features = np.asarray(all_features)[arr]
    all_labels = np.asarray(all_labels)[arr]
    #assert (np.sum(data_prop.values()) != 1.), 'Train vs Test split specified incorrectly'

    train_idx_lim = int(data_prop['train'] * total_items)
    val_idx_lim = int((data_prop['train'] + data_prop['val']) * total_items)

    # write the tf records as specified
    if config.train_tfrecords != None:
        write_tfrecord(
            tfrecord_dir,
            config.train_tfrecords,
            all_features[:train_idx_lim],
            all_labels[:train_idx_lim])

    if config.val_tfrecords != None:
        write_tfrecord(
            tfrecord_dir,
            config.val_tfrecords,
            all_features[train_idx_lim:val_idx_lim],
            all_labels[train_idx_lim:val_idx_lim])

    if config.test_tfrecords != None:
        write_tfrecord(
            tfrecord_dir,
            config.test_tfrecords,
            all_features[val_idx_lim:],
            all_labels[val_idx_lim:])


if __name__ == '__main__':
    config = config.Config()
    main(
        data_folder= os.path.join(config.base_dir,config.data_dir, config.dataset_dir),
        experiment=config.dataset,
        tfrecord_dir= os.path.join(config.base_dir, config.tfrecord_dir),
        config=config
    )
