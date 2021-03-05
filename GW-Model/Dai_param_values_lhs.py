# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 14:05:30 2020

@author: Jing
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from lhsdrv import lhs

# Parameter bounds and distributions
bounds = {'a'  : [2.0, 0.4],
          'b'  : [0.2, 0.5],
          'hk' : [2.9, 0.5],
          'hk1': [2.6, 0.3],
          'hk2': [3.2, 0.3],
          'f1' : [3.5, 0.75],
          'f2' : [2.5, 0.3],
          'r'  : [0.3, 0.05]}

dists = {'a'  : 'NORMAL',
         'b'  : 'UNIFORM',
         'hk' : 'LOGNORMAL-N',
         'hk1': 'LOGNORMAL-N',
         'hk2': 'LOGNORMAL-N',
         'f1' : 'NORMAL',
         'f2' : 'NORMAL',
         'r'  : 'NORMAL'}

# Sampling from the stratum using LHSdrv
N = 5000
problem = {'nvars': 8,
           'names': ['a', 'b', 'hk', 'hk1', 'hk2', 'f1', 'f2', 'r'],
           'bounds': [bounds['a'], bounds['b'], bounds['hk'], bounds['hk1'], bounds['hk2'], bounds['f1'], bounds['f2'], bounds['r']],
           'dists': [dists['a'], dists['b'], dists['hk'], dists['hk1'], dists['hk2'], dists['f1'], dists['f2'], dists['r']]
           }
param_values = lhs(problem, N, seed=933090940)

# Plot the pdfs
plt.figure(figsize=(18, 6))
for i in range(problem['nvars']):
    plt.subplot(2, 4, i + 1)
    plt.hist(param_values[:, i], bins=50, label='mean=%.4f' %np.mean(param_values[:, i]))
    plt.xlabel(problem['names'][i])
    plt.ylabel('PDF')
    plt.legend()

plt.show()

# Save the values
np.save('param_values_' + str(N) + '.npy', param_values)