import pickle, os, glob, tqdm

folders = glob.glob('data/*')
for fold in folders:
    files = glob.glob(fold+'/*')
    for f in tqdm.tqdm(files):
        X = pickle.load(open(f,'rb'))
        pickle.dump(X,open(f,'wb'),protocol=2)
