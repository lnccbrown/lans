import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['font.size'] = 24
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14

'''
X = pickle.load(open('/media/data_cifs/lakshmi/projectABC/results/cogsci/IS_model_full_ddm_training_data_binned_1_nbins_256_n_100000_N_1024_idx_10.pickle','rb'))
print(X['norm_perplexity'])
df = pd.DataFrame(X['posterior_samples'][:10000,:], columns=['v', 'a', 'w', 'ndt', 'dw', 'sdv', 'dndt'])
ax = pd.scatter_matrix(df, figsize=(6,6), alpha=0.01, diagonal='kde', density_kwds={'color':'black'}, c='gray')
for i in range(ax.shape[0]):
    for j in range(ax.shape[1]):
        ax[i,j].tick_params(axis='both', labelsize=14)

plt.subplots_adjust(hspace = 0.15, wspace = 0.15) #0.75 
#plt.tight_layout()
plt.savefig('cov_full_ddm.png',dpi=100)
plt.close()

X = pickle.load(open('/media/data_cifs/lakshmi/projectABC/results/cogsci/IS_model_ornstein_training_data_binned_1_nbins_256_n_100000_N_1024_idx_10.pickle','rb'))
print(X['norm_perplexity'])
df = pd.DataFrame(X['posterior_samples'][:10000,:], columns=['v', 'a', 'w', 'g', 'ndt'])
ax = pd.scatter_matrix(df, figsize=(6,6), alpha=0.01, diagonal='kde', density_kwds={'color':'black'}, c='gray')
for i in range(ax.shape[0]):
    for j in range(ax.shape[1]):
        ax[i,j].tick_params(axis='both', labelsize=14)

plt.subplots_adjust(hspace = 0.15, wspace = 0.15) #0.75 
#
plt.savefig('cov_ornstein.png', dpi=100)
plt.close()


X = pickle.load(open('/media/data_cifs/lakshmi/projectABC/results/cogsci/IS_model_weibull_training_data_binned_1_nbins_256_n_100000_N_1024_idx_0.pickle','rb'))
print(X['norm_perplexity'])
df = pd.DataFrame(X['posterior_samples'][:10000,:], columns=['v', 'a', 'w','ndt','alpha','beta'])
ax = pd.scatter_matrix(df, figsize=(6,6) , alpha=0.01, diagonal='kde', density_kwds={'color':'black'}, c='gray')
for i in range(ax.shape[0]):
    for j in range(ax.shape[1]):
        ax[i,j].yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
        ax[i,j].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
        ax[i,j].tick_params(axis='both', labelsize=14)
        ax[i,j].tick_params(axis='x', rotation=0)
        #ax[i,j].tick_params(axis='x', which='minor', bottom=False)
        if i == ax.shape[0] - 1:
            Z = ax[i,j].get_xticks()
            a,b = np.mean(Z), np.std(Z)
            ax[i,j].set_xticks([a-b/2, a+b/2])
        if j == 0:
            Z = ax[i,j].get_yticks()
            a,b = np.mean(Z), np.std(Z)
            ax[i,j].set_yticks([a-b/2, a+b/2])


#import ipdb; ipdb.set_trace()
plt.subplots_adjust(hspace = 0.15, wspace = 0.15) #0.75 
#plt.align_ylabels(ax[:,0])
plt.gcf().align_labels(ax)
#
plt.savefig('cov_weibull.png', dpi=100)
plt.close()
'''

X = pickle.load(open('/media/data_cifs/lakshmi/projectABC/results/cogsci/IS_model_race_model_4_training_data_binned_1_nbins_256_n_100000_N_1024_idx_0.pickle','rb'))
print(X['norm_perplexity'])
df = pd.DataFrame(X['posterior_samples'][:10000,:], columns=['v1','v2','v3', 'v4' , 'a', 'w1','w2', 'w3', 'w4','ndt'])
ax = pd.scatter_matrix(df, figsize=(6,6), alpha=0.01, diagonal='kde', density_kwds={'color':'black'}, c='gray')
for i in range(ax.shape[0]):
    for j in range(ax.shape[1]):
        ax[i,j].tick_params(axis='both', labelsize=12)
        ax[i,j].yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
        ax[i,j].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
        ax[i,j].tick_params(axis='x', rotation=90)
        if i == ax.shape[0] - 1:
            Z = ax[i,j].get_xticks()
            a,b = np.mean(Z), np.std(Z)
            ax[i,j].set_xticks([a-b/2, a+b/2])
        if j == 0:
            Z = ax[i,j].get_yticks()
            a,b = np.mean(Z), np.std(Z)
            ax[i,j].set_yticks([a-b/2, a+b/2])


plt.subplots_adjust(hspace = 0.25, wspace = 0.25) #0.75 
plt.gcf().align_labels(ax)
#plt.tight_layout()
#
plt.savefig('cov_race4.png', dpi=100)
plt.close()

'''
model = 'race_model_6'
idx = 43

#X = pickle.load(open('/media/data_cifs/lakshmi/projectABC/results/cogsci/IS_model_ddm_mic2_training_data_binned_1_nbins_512_n_100000_N_1024_idx_171.pickle','rb'))
X = pickle.load(open('/media/data_cifs/lakshmi/projectABC/results/benchmark_exps/IS_model_{}_training_data_binned_1_nbins_256_n_100000_N_1024_idx_{}_tdistribution.pickle'.format(model,idx),'rb'))

print(X['norm_perplexity'])
df = pd.DataFrame(X['posterior_samples'][:10000,:]) #, columns=['v','a','w', 'ndt','theta'])
pd.scatter_matrix(df, figsize=(6,6), alpha=0.01, diagonal='kde', density_kwds={'color':'black'}, c='gray')
#plt.savefig('cov_ddm_mic2.png',dpi=100)
#plt.close()
plt.show()
'''

'''
X = pickle.load(open('/media/data_cifs/lakshmi/projectABC/results/results_chong_sample_0_model_ddm_seq2_training_data_binned_1_nbins_512_n_100000.pickle','rb'))

print(X['norm_perplexity'])
df = pd.DataFrame(X['posterior_samples'][:10000,:]) #, columns=['v','a','w', 'ndt','theta'])
pd.scatter_matrix(df, figsize=(6,6), alpha=0.01, diagonal='kde', density_kwds={'color':'black'}, c='gray')
plt.show()

'''

'''
#X = pickle.load(open('/media/data_cifs/lakshmi/projectABC/results/benchmark_exps/IS_model_weibull_training_data_binned_1_nbins_256_n_100000_N_1024_idx_0_tdist_reparam.pickle','rb'))
#print(X['norm_perplexity'])
#df = pd.DataFrame(X['posterior_samples'][:10000,:]) #, columns=['v','a','w', 'ndt','theta'])
#pd.scatter_matrix(df, figsize=(6,6), alpha=0.01, diagonal='kde', density_kwds={'color':'black'}, c='gray')
#plt.show(block=False)

#print(X['gt_params'])
X = pickle.load(open('/media/data_cifs/lakshmi/projectABC/results/results_chong_sample_3_model_ddm_mic2_training_data_binned_1_nbins_512_n_100000.pickle','rb'))
print(X['norm_perplexity'])
df = pd.DataFrame(X['posterior_samples'][:10000,:]) #, columns=['v','a','w', 'ndt','theta'])
pd.scatter_matrix(df, figsize=(6,6), alpha=0.01, diagonal='kde', density_kwds={'color':'black'}, c='gray')
plt.show()
'''
