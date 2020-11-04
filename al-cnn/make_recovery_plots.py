import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import tqdm
from scipy import stats
import matplotlib

matplotlib.rcParams['font.family'] = "Times New Roman"
matplotlib.rcParams['xtick.labelsize'] = 34
matplotlib.rcParams['ytick.labelsize'] = 34
#matplotlib.rcParams['axes.linewidth'] = 0.2

def make_rec_plots(model_name='fullddm', params=None, gridshape=None):

    #my_files = glob.glob('/media/data_cifs/lakshmi/projectABC/results/cogsci/IS_model_%s_training_data_binned_1_nbins_256_n_100000_N_1024_idx_*'%(model_name))
    my_files = glob.glob('/media/data_cifs/lakshmi/projectABC/results/cogsci/IS_model_%s_training_data_binned_1_nbins_512_n_100000_N_1024_idx_*'%(model_name))

    #####
    # Do this only to pick out the last few
    M = ['_'.join(x.split('_')[:-1]) + '_%06d'%int(x.split('_')[-1].split('.')[0]) + '.pickle' for x in my_files]
    M.sort()
    my_files = ['_'.join(x.split('_')[:-1]) + '_%d'%int(x.split('_')[-1].split('.')[0]) + '.pickle' for x in M]
    my_files = my_files #[-200:]
    #####
    
    '''
    time_to_convergence = []
    true_params, rec_params = [], []
    eff_samples = []
    
    for f in tqdm.tqdm(my_files):
        X = pickle.load(open(f,'rb'))
        if X['norm_perplexity'] < 0.01:
            continue
        time_to_convergence.append(X['timeToConvergence'])
        effective_sample_size = 1 / np.sum(np.square(X['final_w']))
        eff_samples.append(effective_sample_size)

        rec = np.mean(X['posterior_samples'],axis=0)
        tr = X['gt_params']
        true_params.append(tr)
        rec_params.append(rec)

    A = np.asarray(true_params)
    B = np.asarray(rec_params)
    C = np.asarray(time_to_convergence)
    D = np.asarray(eff_samples)

    print('Model: {}, Avg. TTC: {}, std: {}, min: {}, max: {}'.format(model_name, np.mean(C),np.std(C),C.min(), C.max()))
    print('Model: {}, Avg. ESS: {}, std: {}, min: {}, max: {}'.format(model_name, np.mean(D),np.std(D),D.min(), D.max()))

    np.save('true_{}.npy'.format(model_name), A)
    np.save('rec_{}.npy'.format(model_name) , B)
    '''

    A = np.load('true_{}.npy'.format(model_name))
    B = np.load('rec_{}.npy'.format(model_name))

    #params = ['v', 'a', 'w', 'ndt', 'dw', 'sdv', 'dndt']
    plt.figure(figsize=(6,6))
    axis_font = 10 #18,14,10
    title_font = 14 # 24,18,14 

    #plt.figure()
    for k in range(A.shape[1]):
        slope, intercept, r_value, p_value, std_err = stats.linregress(A[:,k], B[:,k])
        plt.subplot(gridshape[0],gridshape[1],k+1)
        plt.scatter(A[:,k], B[:,k], 100, alpha=0.5, linewidths=0, marker='.', c='gray')
        amin, amax = A[:,k].min(), A[:,k].max()
        plt.plot(np.array([amin,amax]), intercept + slope*np.array([amin, amax]), 'r', linewidth=2, alpha=0.75)
        
        q1, q3 = np.quantile(A[:,k], 0.25), np.quantile(A[:,k],0.75)
        lab = [q1, q3]
        locs, labels = plt.xticks(lab, ['%0.2f'%x for x in lab])
        plt.setp(labels, fontsize=axis_font, fontweight='bold') #18 #10
 
        q1, q3 = np.quantile(B[:,k], 0.25), np.quantile(B[:,k],0.75)
        lab = [q1, q3]
        locs, labels = plt.yticks(lab, ['%0.2f'%x for x in lab])
        plt.setp(labels, fontsize=14, fontweight='bold') #, rotation=45) #18 #10
        plt.tick_params(axis='y', pad=-5) 
        if k >= A.shape[1]-gridshape[1]:
            plt.xlabel('True', fontname='Times New Roman', fontweight='bold', fontsize=axis_font)
        if k%gridshape[1] == 0:
            plt.ylabel('Recovered', fontname='Times New Roman', fontweight='bold', fontsize=axis_font)
        plt.title('%s ($R^2$ = %.2f)'%(params[k],r_value), fontname='Times New Roman',fontweight='bold', fontsize=title_font) #24 #14
    
    plt.subplots_adjust(hspace = 0.75, wspace = 0.35) #0.75 
    #plt.tight_layout()
    plt.savefig('param_rec_{}.png'.format(model_name),bbox_inches='tight')
    plt.show() 
    #plt.close()
    

if __name__ == '__main__':
    #import ipdb; ipdb.set_trace()
    #make_rec_plots(model_name='ddm_par2',params=['v_h', 'v_l1', 'v_l2', 'a', 'w_h', 'w_l1', 'w_l2', 'ndt'], gridshape=[3,3])
    #make_rec_plots(model_name='ornstein',params=['v', 'a', 'w', 'g', 'ndt'], gridshape=[3,2])
    #make_rec_plots(model_name='weibull',params=['v', 'a', 'w','ndt','alpha','beta'], gridshape=[3,2])
    make_rec_plots(model_name='race_model_4',params=['v1','v2','v3', 'v4' , 'a', 'w1','w2', 'w3', 'w4','ndt'], gridshape=[3,4])
    #make_rec_plots(model_name='full_ddm', params=['v', 'a', 'w','ndt','dw', 'sdv', 'dndt'], gridshape=[3,3])
    
'''
X = pickle.load(open('/media/data_cifs/lakshmi/projectABC/results/cogsci/IS_model_ornstein_training_data_binned_1_nbins_256_n_100000_N_1024_idx_10.pickle','rb'))
print(X['norm_perplexity'])
df = pd.DataFrame(X['posterior_samples'], columns=['v', 'a', 'w', 'g', 'ndt'])
pd.scatter_matrix(df, figsize=(6,6), alpha=0.25)
plt.savefig('cov_ornstein.png', dpi=100)
plt.close()

X = pickle.load(open('/media/data_cifs/lakshmi/projectABC/results/cogsci/IS_model_weibull_training_data_binned_1_nbins_256_n_100000_N_1024_idx_0.pickle','rb'))
print(X['norm_perplexity'])
df = pd.DataFrame(X['posterior_samples'], columns=['v', 'a', 'w','ndt','alpha','beta'])
pd.scatter_matrix(df, figsize=(6,6), alpha=0.25)
plt.savefig('cov_weibull.png', dpi=100)
plt.close()

X = pickle.load(open('/media/data_cifs/lakshmi/projectABC/results/cogsci/IS_model_race_model_4_training_data_binned_1_nbins_256_n_100000_N_1024_idx_0.pickle','rb'))
print(X['norm_perplexity'])
df = pd.DataFrame(X['posterior_samples'], columns=['v1','v2','v3', 'v4' , 'a', 'w1','w2', 'w3', 'w4','ndt'])
pd.scatter_matrix(df, figsize=(6,6), alpha=0.25)
plt.savefig('cov_race4.png', dpi=100)
plt.show()
plt.close()
'''
