# -*- coding: utf-8 -*-
"""
Created on Fri May 15 14:56:20 2020

@author: Jing
"""
import numpy as np


def evaluate(values, delta, alpha, a):
    """Sobol G*-function.

    .. [1] Saltelli, A., Annoni, P., Azzini, I., Campolongo, F., Ratto, M., 
           Tarantola, S., 2010. Variance based sensitivity analysis of model 
           output. Design and estimator for the total sensitivity index. 
           Computer Physics Communications 181, 259–270. 
           https://doi.org/10.1016/j.cpc.2009.09.018

    Parameters
    ----------
    values : numpy.ndarray
        input variables
    delta : numpy.ndarray
        parameter values
    alpha : numpy.ndarray
        parameter values
    a : numpy.ndarray
        parameter values

    Returns
    -------
    Y : Result of G*-function
    """
    # Check the dimension of the input
    if (values.shape[1] != delta.shape[0]):
        raise ValueError("The dimension of inputs is not consistent")
    elif (values.shape[1] != alpha.shape[0]):
        raise ValueError("The dimension of inputs is not consistent")
    elif (values.shape[1] != a.shape[0]):
        raise ValueError("The dimension of inputs is not consistent")
    else:
        pass
    
    if type(values) != np.ndarray:
        raise TypeError("The argument `values` must be a numpy ndarray")

    ltz = values < 0
    gto = values > 1

    if ltz.any() == True:
        raise ValueError("Sobol G function called with values less than zero")
    elif gto.any() == True:
        raise ValueError("Sobol G function called with values greater than one")
        
    gi = ((1 + alpha) * np.power(np.abs(2 * (values + delta - np.modf(values + delta)[1]) - 1), alpha) + a) / (1 + a)

    return np.prod(gi, axis=1)
