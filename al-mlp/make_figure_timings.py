import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    machine = 'home'
    if machine == 'home':
        pdt = pd.DataFrame(pickle.load(open('/users/afengler/OneDrive/project_nn_likelihoods/data/timings/timings.pickle', 'rb')))
    if machine == 'ccv':
        pdt = pd.DataFrame(pickle.load(open('./timings/timings.pickle', 'rb')))
    if machine == 'x7':
        pass
    
    pdt_group_mean = pdt.groupby('nsamples').mean()

    p1 = plt.scatter(np.log2(pdt_group_mean.index), 1000 * pdt_group_mean['numpy_timings'], 
                     label = 'Numpy')
    p2 = plt.scatter(np.log2(pdt_group_mean.index), 1000 * pdt_group_mean['keras_no_batch_timings'], 
                     label = 'Keras batch 1')
    p3 = plt.scatter(np.log2(pdt_group_mean.index), 1000 * pdt_group_mean['keras_var_batch_timings'], 
                     label = 'Keras batch 1000')
    p4 = plt.scatter(np.log2(pdt_group_mean.index), 1000 * pdt_group_mean['keras_fix_batch_timings'], 
                     label = 'Keras batch all')
    p5 = plt.scatter(np.log2(pdt_group_mean.index), 1000 * pdt_group_mean['keras_var_batch_no_pred_timings'],
                     label = 'Keras batch all no pred')
    p6 = plt.scatter(np.log2(pdt_group_mean.index), 1000 * pdt_group_mean['navarro_timings'], 
                     label = 'Navarro Fuss')
    
    plt.xlabel('log2 sample size',
               size = 20)
    
    plt.ylabel('ms', 
               size = 20)
    
    plt.title('Time / Forward Pass', 
              size = 24)
    
    plt.legend(loc = 'lower left', 
               bbox_to_anchor = (0, 0.65),
               fancybox = True, 
               shadow = True, 
               ncol = 1)

    plt.savefig('./figures/timings_mlp.png', 
                dpi = 150,
                bbox_inches = 'tight')
    plt.show()