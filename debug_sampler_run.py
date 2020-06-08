import pickle, glob, gzip, tqdm
import matplotlib.pyplot as plt

#import ipdb; ipdb.set_trace()
files = glob.glob('/media/data_cifs/lakshmi/projectABC/results/eLIFE_exps/*.pklz')
model = 'levy'

runtimes = []
plt.figure()
for f in tqdm.tqdm(files):
    if model not in f:
        continue
    fid = gzip.open(f,'rb')
    X = pickle.load(fid)
    runtimes.append(X['timeToConvergence'])
    if '1024' in f:
        plt.plot(X['norm_perplexity'][1:],color='r')
    else:
        plt.plot(X['norm_perplexity'][1:],color='b')
    fid.close()
plt.show(block=False)

plt.figure()
plt.hist(runtimes)
plt.show()

