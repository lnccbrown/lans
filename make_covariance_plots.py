import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['font.size'] = 18

'''
X = pickle.load(open('/media/data_cifs/lakshmi/projectABC/results/cogsci/IS_model_full_ddm_training_data_binned_1_nbins_256_n_100000_N_1024_idx_10.pickle','rb'))
print(X['norm_perplexity'])
df = pd.DataFrame(X['posterior_samples'][:10000,:], columns=['v', 'a', 'w', 'ndt', 'dw', 'sdv', 'dndt'])
pd.scatter_matrix(df, figsize=(6,6), alpha=0.01, diagonal='kde', density_kwds={'color':'black'}, c='gray')
plt.savefig('cov_full_ddm.png',dpi=100)
plt.close()

X = pickle.load(open('/media/data_cifs/lakshmi/projectABC/results/cogsci/IS_model_ornstein_training_data_binned_1_nbins_256_n_100000_N_1024_idx_10.pickle','rb'))
print(X['norm_perplexity'])
df = pd.DataFrame(X['posterior_samples'][:10000,:], columns=['v', 'a', 'w', 'g', 'ndt'])
pd.scatter_matrix(df, figsize=(6,6), alpha=0.01, diagonal='kde', density_kwds={'color':'black'}, c='gray')
plt.savefig('cov_ornstein.png', dpi=100)
plt.close()

X = pickle.load(open('/media/data_cifs/lakshmi/projectABC/results/cogsci/IS_model_weibull_training_data_binned_1_nbins_256_n_100000_N_1024_idx_0.pickle','rb'))
print(X['norm_perplexity'])
df = pd.DataFrame(X['posterior_samples'][:10000,:], columns=['v', 'a', 'w','ndt','alpha','beta'])
pd.scatter_matrix(df, figsize=(6,6) , alpha=0.01, diagonal='kde', density_kwds={'color':'black'}, c='gray')
plt.savefig('cov_weibull.png', dpi=100)
plt.close()

X = pickle.load(open('/media/data_cifs/lakshmi/projectABC/results/cogsci/IS_model_race_model_4_training_data_binned_1_nbins_256_n_100000_N_1024_idx_0.pickle','rb'))
print(X['norm_perplexity'])
df = pd.DataFrame(X['posterior_samples'][:10000,:], columns=['v1','v2','v3', 'v4' , 'a', 'w1','w2', 'w3', 'w4','ndt'])
pd.scatter_matrix(df, figsize=(6,6), alpha=0.01, diagonal='kde', density_kwds={'color':'black'}, c='gray')
plt.savefig('cov_race4.png', dpi=100)
plt.show()
plt.close()
'''

#X = pickle.load(open('/media/data_cifs/lakshmi/projectABC/results/cogsci/IS_model_ddm_mic2_training_data_binned_1_nbins_512_n_100000_N_1024_idx_171.pickle','rb'))
X = pickle.load(open('/media/data_cifs/lakshmi/projectABC/results/cogsci/IS_model_angle_training_data_binned_1_nbins_256_n_100000_N_512_idx_57.pickle','rb'))

print(X['norm_perplexity'])
df = pd.DataFrame(X['posterior_samples'][:10000,:], columns=['v','a','w', 'ndt','theta'])
pd.scatter_matrix(df, figsize=(6,6), alpha=0.01, diagonal='kde', density_kwds={'color':'black'}, c='gray')
#plt.savefig('cov_ddm_mic2.png',dpi=100)
#plt.close()
plt.show(block=False)

#X = pickle.load(open('/media/data_cifs/lakshmi/projectABC/results/cogsci/IS_model_ddm_mic2_training_data_binned_1_nbins_512_n_100000_N_1024_idx_171.pickle','rb'))
X = pickle.load(open('/media/data_cifs/lakshmi/projectABC/results/cogsci/IS_model_angle_training_data_binned_1_nbins_256_n_100000_N_512_idx_57_a.pickle','rb'))

print(X['norm_perplexity'])
df = pd.DataFrame(X['posterior_samples'][:10000,:], columns=['v','a','w', 'ndt','theta'])
pd.scatter_matrix(df, figsize=(6,6), alpha=0.01, diagonal='kde', density_kwds={'color':'black'}, c='gray')
#plt.savefig('cov_ddm_mic2.png',dpi=100)
#plt.close()
plt.show()

