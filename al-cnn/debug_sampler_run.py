import pickle, glob, gzip, tqdm
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
import matplotlib as mpl
import multiprocessing as MP

'''
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
'''

def collect_runtime(f):
    fid = gzip.open(f, 'rb')
    X = pickle.load(fid)
    fid.close()
    return X['timeToConvergence']


pool = MP.Pool(8)

plt.figure()
models = ['ddm', 'levy', 'ornstein', 'ddm_sdv', 'full_ddm2', 'angle', 'race_model_3', 'race_model_4', 'weibull', 'lca_3', 'lca_4']
n_models = len(models)
datasets = ['1024', '4096']
count = 0.
cmap =  mpl.cm.get_cmap('Paired')
for model in models:
    for dataset in datasets:
        files = glob.glob('/media/data_cifs/lakshmi/projectABC/results/time_benchmark_eLIFE_exps/IS_model_{}_training_data_binned_1_nbins_512_n_100000_N_{}_idx_*'.format(model,dataset))
        runtimes = []
        for _ in tqdm.tqdm(pool.imap(collect_runtime,files), total=len(files)):
            runtimes.append(_)
            pass
        sns.distplot(runtimes, fit=norm, kde=False, hist=True, label='{}_{}'.format(model,dataset), hist_kws={'color':cmap(count/(n_models*2)), 'alpha':0.5}, fit_kws={'color':cmap(count/(n_models*2)), 'alpha':1.})
        count += 1.

pool.close()

plt.xlabel('Time (in seconds)')
plt.legend()
plt.show()        
