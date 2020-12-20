# -*- coding: utf-8 -*-
"""
Created on Tue May 19 16:19:35 2020

@author: Jing
"""
import os 
import numpy as np
from scipy import stats

def mc(problem, nobs, seed=None):
    """
    Generate a latin-hypercube design using the LHS251.exe
    
    Returns a NumPy matrix containing the model inputs generated by Latin
    hypercube sampling.  The resulting matrix contains nobs rows and D columns,
    where D is the number of parameters.
    
    Parameters
    ----------
    problem : dict
        The descritpion of the problem. 
    nobs : int
        The number of factors to generate samples for     
  
    Optional
    --------
    seed : int
        The random seed for reproducing the results
    
    Returns
    -------
    H : 2d-array
        An nobs-by-D design matrix
     
    """
    if seed:
        np.random.seed(seed)
    
    nvars = problem['nvars']
    H = np.random.rand(nobs, nvars)
    H = _scale_samples(H, problem['bounds'], problem['dists'])
        
    return H

def _scale_samples(params, bounds, dists):
    """Rescale samples in 0-to-1 range to the distributions

    Arguments
    ---------
    problem : dict
        problem definition including bounds
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
    b = np.array(bounds)

    # initializing matrix for converted values
    conv_params = np.zeros_like(params)

    # loop over the parameters
    for i in range(conv_params.shape[1]):
        # setting first and second arguments for distributions
        b1 = b[i][0]
        b2 = b[i][1]

        if dists[i] == 'TRIANG':
            # checking for correct parameters
            if b1 <= 0 or b2 <= 0 or b2 >= 1:
                raise ValueError('''Triangular distribution: Scale must be
                    greater than zero; peak on interval [0,1]''')
            else:
                conv_params[:, i] = stats.triang.ppf(
                    params[:, i], c=b2, scale=b1, loc=0)

        elif dists[i] == 'UNIFORM':
            if b1 >= b2:
                raise ValueError('''Uniform distribution: lower bound
                    must be less than upper bound''')
            else:
                conv_params[:, i] = params[:, i] * (b2 - b1) + b1

        elif dists[i] == 'NORMAL':
            if b2 <= 0:
                raise ValueError('''Normal distribution: stdev must be > 0''')
            else:
                conv_params[:, i] = stats.norm.ppf(
                    params[:, i], loc=b1, scale=b2)

        # lognormal distribution (ln-space, not base-10)
        # paramters are ln-space mean and standard deviation
        elif dists[i] == 'LOGNORMAL':
            # checking for valid parameters
            if b2 <= 0:
                raise ValueError(
                    '''Lognormal distribution: stdev must be > 0''')
            else:
                conv_params[:, i] = np.exp(
                    stats.norm.ppf(params[:, i], loc=b1, scale=b2))
                
        # truncnormal distribution (ln-space, not base-10)
        # paramters are ln-space mean and standard deviation truncated at 0
        elif dists[i] == 'TRUNCNORMAL':
            # checking for valid parameters
            if b2 <= 0:
                raise ValueError(
                    '''Truncnorm distribution: stdev must be > 0''')
            else:
                conv_params[:, i] = stats.truncnorm.ppf(
                    params[:, i], (0 - b1) / b2, stats.norm.ppf(0.9999,  \
                          loc=b1, scale=b2), loc=b1, scale=b2)
                
        else:
            valid_dists = ['UNIFORM', 'TRIANG', 'NORMAL', 'LOGNORMAL', 'TRUNCNORMAL']
            raise ValueError('Distributions: choose one of %s' %
                             ', '.join(valid_dists))
            
    return conv_params

if __name__ == '__main__': 
    mc_inp = {'nvars': 4,
              'names':['PerLN' ,'PorNO', 'Mult', 'Length'],
              'bounds': [[0, 1], [0, 1], [0, 1], [0, 1]],
              'dists': ['UNIFORM', 'UNIFORM', 'UNIFORM', 'UNIFORM']
              }
    
    H = mc(mc_inp, nobs=500)