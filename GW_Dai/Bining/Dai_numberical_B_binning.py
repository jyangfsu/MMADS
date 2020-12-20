# -*- coding: utf-8 -*-
"""
Created on Fri May 15 20:58:57 2020

@author: Jing
"""
import numpy as np
import numba as nb
from scipy import stats
from SALib.sample import saltelli
from lhsdrv import lhs

# Set random seed
# np.random.seed(2**30)

# Model information
N = 100                        # Number of samples generated for each  parameter
Ma = 2                         # Number of alterantive models for recharge process
Mb = 2                         # Number of alterantive models for geloogy process
Mc = 2                         # Number of alterantive models for snow melt process
PMA = np.array([0.5, 0.5])     # Process model weight for recharge process
PMB = np.array([0.5, 0.5])     # Process model weight for geology process
PMC = np.array([0.5, 0.5])     # Process model weight for snowmelt process

# parameters used in binning
nbins_per_dim = 3                        # Numebr of bins along each paramter dimension              
max_points_per_bin = 20                  # Estimated max number of points in each bin
max_bins_total = nbins_per_dim**4        # Estimated max number of bins  

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


# Sampling from the stratum using LHSDRV
problem = {'nvars': 8,
           'names': ['a', 'b', 'hk', 'hk1', 'hk2', 'f1', 'f2', 'r'],
           'bounds': [bounds['a'], bounds['b'], bounds['hk'], bounds['hk1'], bounds['hk2'], bounds['f1'], bounds['f2'], bounds['r']],
           'dists': [dists['a'], dists['b'], dists['hk'], dists['hk1'], dists['hk2'], dists['f1'], dists['f2'], dists['r']]
           }
param_values = lhs(problem, N, seed=933090936)


def bin_edges(bins, bounds, dists):
    """Rescale samples in 0-to-1 range to the distributions

    Arguments
    ---------
    params : numpy.ndarray
        numpy array of dimensions num_params-by-N,
        where N is the number of samples
    dists : list
        list of distributions, one for each parameter
            unif: uniform with lower and upper bounds
            triang: triangular with width (scale) and location of peak
                    location of peak is in percentage of width
                    lower bound assumed to be zero
            norm: normal distribution with mean and standard deviation
            lognorm: lognormal with ln-space mean and standard deviation
    """
    
    params = np.linspace(0, 1, bins + 1)
    b = np.array(bounds)

    # initializing matrix for converted values
    conv_params = np.zeros_like(params)
    b1 = b[0]
    b2 = b[1]

    if dists == 'UNIFORM':
        if b1 >= b2:
            raise ValueError('''Uniform distribution: lower bound
                must be less than upper bound''')
        else:
            conv_params[:] = params[:] * (b2 - b1) + b1

    elif dists == 'NORMAL':
        if b2 <= 0:
            raise ValueError('''Normal distribution: stdev must be > 0''')
        else:
            conv_params[:] = stats.norm.ppf(
                params[:], loc=b1, scale=b2)

    # lognormal distribution (ln-space, not base-10)
    # paramters are ln-space mean and standard deviation
    elif dists == 'LOGNORMAL-N':
        # checking for valid parameters
        if b2 <= 0:
            raise ValueError(
                '''Lognormal distribution: stdev must be > 0''')
        else:
            conv_params[:] = np.exp(
                stats.norm.ppf(params[:], loc=b1, scale=b2))

    else:
        valid_dists = ['UNIFORM', 'NORMAL', 'LOGNORMAL']
        raise ValueError('Distributions: choose one of %s' %
                         ', '.join(valid_dists))
    # Replace the -inf and +inf values (need to be improved)
    conv_params[conv_params == -np.inf] = -999999
    conv_params[conv_params == np.inf] = 999999
    
    return conv_params



# bin edages of reaction rate
bin_a = bin_edges(nbins_per_dim, bounds['a'], dists['a'])
bin_b = bin_edges(nbins_per_dim, bounds['b'], dists['b'])
bin_hk = bin_edges(nbins_per_dim, bounds['hk'], dists['hk'])
bin_hk1 = bin_edges(nbins_per_dim, bounds['hk1'], dists['hk1'])
bin_hk2 = bin_edges(nbins_per_dim, bounds['hk2'], dists['hk2'])
bin_f1 = bin_edges(nbins_per_dim, bounds['f1'], dists['f1'])
bin_f2 = bin_edges(nbins_per_dim, bounds['f2'], dists['f2'])
bin_r = bin_edges(nbins_per_dim, bounds['r'], dists['r'])


# pre-determin the binnumber to  accelerate    
binnumbers = np.zeros((Ma, Mc, N))
for i in range(Ma):
    for m in range(Mc):
        print('i = %d, m=%d.' %(i, m))
        if (i==0 and m==0):
            statistic, _, _, binnumbers[i, m, :] = stats.binned_statistic_2d(param_values[:, 0], param_values[:, 5], 
                                                                             np.zeros(N), 
                                                                             'count', 
                                                                             bins=[bin_a, bin_f1],
                                                                             expand_binnumbers=False) 
            
            if len(np.unique(binnumbers[i, m, :])) != nbins_per_dim**2:
                print('    Warnning: Number of empty bins = %d' \
                      %(nbins_per_dim**2 - len(np.unique(binnumbers[i, m, :]))))
            else:
                print('    No empty bins!')
            if np.count_nonzero(statistic>=2) != nbins_per_dim**2:
                print('    Warnning: Number of bins with points less than two = %d.' \
                      %(nbins_per_dim**2 - np.count_nonzero(statistic>=2)))
            else:
                print('    All bins are full!')
                
        if (i==0 and m==1):
           statistic, _, binnumbers[i, m, :] = stats.binned_statistic_dd(param_values[:, [0, 6, 7]], 
                                                                            np.zeros(100),
                                                                            'count', 
                                                                            bins=[bin_a, bin_f2, bin_r],
                                                                            expand_binnumbers=False)
                
           if len(np.unique(binnumbers[i, m, :])) != nbins_per_dim**3:
               print('    Warnning: Number of empty bins = %d' \
                     %(nbins_per_dim**3 - len(np.unique(binnumbers[i, m, :]))))
           else:
               print('    No empty bins!')
           if np.count_nonzero(statistic>=2) != nbins_per_dim**3:
               print('    Warnning: Number of bins with points less than two = %d.' \
                     %(nbins_per_dim**3 - np.count_nonzero(statistic>=3)))
           else:
               print('    All bins are full!')
               
        if (i==1 and m==0):
            statistic, _, _, binnumbers[i, m, :] = stats.binned_statistic_2d(param_values[:, 1], param_values[:, 5], 
                                                                             np.zeros(N), 
                                                                             'count', 
                                                                             bins=[bin_b, bin_f1],
                                                                             expand_binnumbers=False) 
            
            if len(np.unique(binnumbers[i, m, :])) != nbins_per_dim**2:
                print('    Warnning: Number of empty bins = %d' \
                      %(nbins_per_dim**2 - len(np.unique(binnumbers[i, m, :]))))
            else:
                print('    No empty bins!')
            if np.count_nonzero(statistic>=2) != nbins_per_dim**2:
                print('    Warnning: Number of bins with points less than two = %d.' \
                      %(nbins_per_dim**2 - np.count_nonzero(statistic>=2)))
            else:
                print('    All bins are full!')
               
        if (i==1 and m==1):
           statistic, _, binnumbers[i, m, :] = stats.binned_statistic_dd(param_values[:, [1, 6, 7]], 
                                                                            np.zeros(100),
                                                                            'count', 
                                                                            bins=[bin_b, bin_f2, bin_r],
                                                                            expand_binnumbers=False)
                
           if len(np.unique(binnumbers[i, m, :])) != nbins_per_dim**3:
               print('    Warnning: Number of empty bins = %d' \
                     %(nbins_per_dim**3 - len(np.unique(binnumbers[i, m, :]))))
           else:
               print('    No empty bins!')
           if np.count_nonzero(statistic>=2) != nbins_per_dim**3:
               print('    Warnning: Number of bins with points less than two = %d.' \
                     %(nbins_per_dim**3 - np.count_nonzero(statistic>=3)))
           else:
               print('    All bins are full!')
              
                
# determine the binnumber which the points belong to
@nb.njit
def Bin_Process_B(Y):
    bin_Y = np.full((Ma, Mc, max_bins_total, Mb, max_points_per_bin), np.nan)
    for i in range(Ma):
        for m in range(Mc):
            for k in range(Mb):
                binnumber = binnumbers[i, m, :]
                for inumber, number in enumerate(np.unique(binnumber)):
                        bin_Y[i, m, inumber, k, :binnumber[binnumber==number].shape[0]] = Y[i, k, m, :][binnumber==number]
                    
    return bin_Y

@nb.njit
def MMDS_mean_var_B(bin_Y):
    D = np.full((Ma, Mc, max_bins_total, Mb, max_points_per_bin, Mb, max_points_per_bin), np.nan)
    for k in range(Ma):
        for m in range(Mc):
            for inumber in range(max_bins_total):
                for k1 in range(Mb):
                    for l1 in range(max_points_per_bin):
                        for k2 in range(Mb):
                            for l2 in range(max_points_per_bin):
                                D[i, m, inumber, k1, l1, k2, l2] = abs(bin_Y[i, m, inumber, k1, l1] - bin_Y[k, m, inumber, k2, l2])
    
    E_d = np.nanmean(D)
    V_d = np.nanvar(D)
                        
    return E_d, V_d

# Load the binning model output
print('Loading output: ' + 'binning_Y_' + str(N) + '.npy')
Y = np.load('binning_Y_' + str(N) + '.npy')
bin_Y = Bin_Process_B(Y)
E_B, V_B = MMDS_mean_var_B(bin_Y)

print('Mean A =', E_B, 'Var A =', V_B)












































