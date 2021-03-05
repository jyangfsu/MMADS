# -*- coding: utf-8 -*-
"""
Created on Fri May 15 20:58:57 2020

@author: Jing
"""
import numpy as np
import numba as nb
import pandas as pd
from scipy import stats
from SALib.sample import saltelli

from latin251 import lhs
from sobol_G_fun import evaluate

# Set random seed
np.random.seed(2**30)

# Consider three processes each of which with alternative process models
delta = np.array([0, 0, 0])
alpha = np.array([[1,  2], [1, 2], [1, 2]])
a = np.array([[1.5, 1.2], [4.2, 1.8], [6.5, 2.3]])

# Model weitht
N = 100
Ma, Mb, Mc = 2, 2, 2
PMA = [0.5, 0.5]
PMB = [0.5, 0.5]
PMC = [0.5, 0.5]

# Sampling from the stratum using LHS251
problem = {'nvars': 3,
           'names': ['x1', 'x2', 'x3'],
           'bounds': [[0, 1], [0, 1], [0, 1]],
           'dists' :['UNIFORM', 'UNIFORM', 'UNIFORM']
           }

# Settings used for binning
nbins_per_dim = 5                        # Numebr of bins along each paramter dimension              
max_points_per_bin = 10                  # Estimated max number of points in each bin
max_bins_total = nbins_per_dim**2        # Estimated max number of bins  
bin_results = pd.DataFrame({k: v for k, v in zip(problem['names'], [np.tile(np.linspace(0, 1, nbins_per_dim + 1), (problem['nvars'], 1))[i, :] for i in range(problem['nvars'])])})

# Generate parameters using SALib.sample.saltelli
# param_values = saltelli.sample(problem, N, calc_second_order=False, seed=2**30)[::5, :]

# Generate parameters using LHS
param_values = lhs(problem, N, seed=933090936)

# Compute the model outputs
def cmpt_Y():
    Y = np.zeros((Ma, Mb, Mc, N))
    for i in range(Ma):
        for k in range (Mb):
            for m in range(Mc):
                Y[i, k, m, :] = evaluate(param_values, delta, np.array([alpha[0, i], alpha[1, k], alpha[2, m]]), np.array([a[0, i], a[1, k], a[2, m]]))
    return Y
Y = cmpt_Y()

# Print the binning information
def Bin_info(ret):
    print('    Number of bins generated = %d. \n    Max number of points in bin = %d. \n    Number of bins with points less than two = %d.' \
          %(nbins_per_dim**2, np.max(ret.statistic), nbins_per_dim**2 - np.count_nonzero(ret.statistic>=2)))
    
    binnumber = ret.binnumber
    if len(np.unique(binnumber)) != nbins_per_dim**2:
        print('    Warnning: bin(s) with no points exist(s)! Number of empty bins = %d' \
              %(nbins_per_dim**2 - len(np.unique(binnumber))))
    else:
        print('    No empty bins!')
    
    return

# Sensitivity meansures for process C using binning
def Bin_tmp_Y_C():
    print('Computing for process A...')
    tmp_Y = np.full((Ma, Mb, max_bins_total, Mc, max_points_per_bin), np.nan)
    
    for i in range(Ma):
        for k in range(Mb):
            for m in range(Mc):
                if (i==0 and k==0):
                     bins = [bin_results['x1'].values, bin_results['x2'].values]
                     ret = stats.binned_statistic_2d(param_values[:, 0], param_values[:, 1], 
                                                     Y[i, k, m, :], 
                                                     statistic='count', 
                                                     bins=bins, 
                                                     expand_binnumbers=False)
    
                     print('i = %d, k = %d, m = %d.' %(i, k, m))
                     Bin_info(ret)
                     binnumber = ret.binnumber
                     for inumber, number in enumerate(np.unique(binnumber)):
                         tmp_Y[i, k, inumber, m, :len(binnumber[binnumber==number])] = Y[i, k, m, :][binnumber==number]
                    
                elif (i==0 and k==1):
                     bins = [bin_results['x1'].values, bin_results['x2'].values]
                     ret = stats.binned_statistic_2d(param_values[:, 0], param_values[:, 1], 
                                                     Y[i, k, m, :], 
                                                     statistic='count', 
                                                     bins=bins, 
                                                     expand_binnumbers=False)
    
                     print('i = %d, k = %d, m = %d.' %(i, k, m))
                     Bin_info(ret)
                     binnumber = ret.binnumber
                     for inumber, number in enumerate(np.unique(binnumber)):
                         tmp_Y[i, k, inumber, m, :len(binnumber[binnumber==number])] = Y[i, k, m, :][binnumber==number]
                
                elif (i==1 and k==0):
                     bins = [bin_results['x1'].values, bin_results['x2'].values]
                     ret = stats.binned_statistic_2d(param_values[:, 0], param_values[:, 1], 
                                                     Y[i, k, m, :], 
                                                     statistic='count', 
                                                     bins=bins, 
                                                     expand_binnumbers=False)
    
                     print('i = %d, k = %d, m = %d.' %(i, k, m))
                     Bin_info(ret)
                     binnumber = ret.binnumber
                     for inumber, number in enumerate(np.unique(binnumber)):
                         tmp_Y[i, k, inumber, m, :len(binnumber[binnumber==number])] = Y[i, k, m, :][binnumber==number]
                
                elif (i==1 and k==1):
                     bins = [bin_results['x1'].values, bin_results['x2'].values]
                     ret = stats.binned_statistic_2d(param_values[:, 0], param_values[:, 1], 
                                                     Y[i, k, m, :], 
                                                     statistic='count', 
                                                     bins=bins, 
                                                     expand_binnumbers=False)
    
                     print('i = %d, k = %d, m = %d.' %(i, k, m))
                     Bin_info(ret)
                     binnumber = ret.binnumber
                     for inumber, number in enumerate(np.unique(binnumber)):
                         tmp_Y[i, k, inumber, m, :len(binnumber[binnumber==number])] = Y[i, k, m, :][binnumber==number]

    return tmp_Y

tmp_Y = Bin_tmp_Y_C()

@nb.njit()
def Bin_mean_var_C():    
    d_Y = np.full((Ma, Mb, max_bins_total, Mc, Mc, max_points_per_bin, max_points_per_bin), np.nan)
    for i in range(Ma):
        for k in range(Mb):
            for inumber in range(max_bins_total):
                for mst in range(Mc):
                    for mnd in range(Mc):
                        for nst in range(max_points_per_bin):
                            for nnd in range(max_points_per_bin):
                                d_Y[i, k, inumber, mst, mnd, nst, nnd] = abs(tmp_Y[i, k, inumber, mst, nst] - tmp_Y[i, k, inumber, mnd, nnd])
                                

    E_d = np.nanmean(d_Y)
    Var_d = np.nanvar(d_Y)
    
    return E_d, Var_d
                     
E_C, Var_C = Bin_mean_var_C()

print('E_C = %.4f. Var_C = %.4f' %(E_C, Var_C))
