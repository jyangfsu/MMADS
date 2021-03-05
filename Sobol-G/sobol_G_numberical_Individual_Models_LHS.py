# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 21:22:21 2021

@author: Jing
"""
import numpy as np
import numba as nb
import matplotlib.pyplot as plt

from latin251 import lhs
from sobol_G_fun import evaluate

# Consider three processes each of which has two alternative process models
Ma, Mb, Mc = 2, 2, 2
delta = np.array([0, 0, 0])
alpha = np.array([[1,  2], [1, 2], [1, 2]])
a = np.array([[1.5, 1.2], [4.2, 1.8], [6.5, 2.3]])

# Number of parameter values for each parameter
N = 500

# Sampling from the stratum using LHS. The LHS code of Iuzzolino (2003) was used.
problem = {'nvars': 3,
           'names': ['x1', 'y1', 'z1'],
           'bounds': [[0, 1], [0, 1], [0, 1]],
           'dists' :['UNIFORM', 'UNIFORM', 'UNIFORM']
           }
param_values = lhs(problem, N, seed=933090934)
    
# Perform the analysis
for ima in range(Ma):
    for imb in range(Mb):
        for imc in range(Mc):
            print('Current system model: ima=%d, imb=%d, imc=%d' %(ima, imb, imc))
            
            Y1 = np.zeros((N, N, N))
            for i in range(N):
                for j in range(N):
                    for k in range(N):
                        values = np.array([param_values[i][0], param_values[j][1], param_values[k][2]]).reshape(1, 3)
                        Y1[i, j, k] = evaluate(values, delta, np.array([alpha[0, ima], alpha[1, imb], alpha[2, imc]]), np.array([a[0, ima], a[1, imb], a[2, imc]]))
            
            # Save the model outputs 
            # np.save('D:\Y1_500.npy', Y1)
            
            # Calculate the mean and variance of the differences for g1*
            diffA = np.zeros(N * N * N * N)
            it = 0
            for j in range(N):
                for k in range(N):
                    for i1 in range(N):
                        for i2 in range(N):
                            diffA[it] = Y1[i1, j, k] - Y1[i2, j, k]
                            it = it + 1
            print('    Process g1*')                
            print('        mean=%.2f \n        var =%.2f' %(np.mean(abs(diffA  * 100)), np.var(diffA  * 10)))
            
            # Calculate the mean and variance of the differences for g2*
            diffB = np.zeros(N * N * N * N)
            it = 0
            for i in range(N):
                for k in range(N):
                    for j1 in range(N):
                        for j2 in range(N):
                            diffB[it] = Y1[i, j1, k] - Y1[i, j2, k]
                            it = it + 1
            print('    Process g2*')            
            print('        mean=%.2f \n        var =%.2f' %(np.mean(abs(diffB  * 100)), np.var(diffB  * 10)))
             
            # Calculate the mean and variance of the differences for g3*
            diffC = np.zeros(N * N * N * N)
            it = 0
            for i in range(N):
                for j in range(N):
                    for k1 in range(N):
                        for k2 in range(N):
                            diffC[it] = Y1[i, j, k1] - Y1[i, j, k2]
                            it = it + 1
            print('    Process g3*')               
            print('        mean=%.2f \n        var =%.2f' %(np.mean(abs(diffC  * 100)), np.var(diffC  * 10)))
              