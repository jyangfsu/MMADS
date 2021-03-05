# -*- coding: utf-8 -*-
"""
Created on Fri May 15 14:16:28 2020

@author: Jing
"""
import numpy as np
import numba as nb
import matplotlib.pyplot as plt

from latin251 import lhs
from sobol_G_fun import evaluate

# Consider three processes each of which with alternative process models
delta = np.array([0, 0, 0])

alpha = np.array([[1,  2], [1, 2], [1, 2]])
a = np.array([[1.5, 1.2], [4.2, 1.8], [6.5, 2.3]])

# Model info
N = 500
Ma, Mb, Mc = 2, 2, 2
PMA = np.array([0.5, 0.5])
PMB = np.array([0.5, 0.5])
PMC = np.array([0.5, 0.5])

# Compute the model output
def cmpt_Y_B(N):
    print('Computing system output...')
    
    # Sampling from the stratum using LHS251
    problem = {'nvars': 3,
               'names': ['x1', 'y1', 'z1'],
               'bounds': [[0, 1], [0, 1], [0, 1]],
               'dists' :['UNIFORM', 'UNIFORM', 'UNIFORM']
               }
    param_values = lhs(problem, N, seed=933090934)

    Y = np.zeros((Ma, Mc, N, Mb, N), dtype=np.float32)
    for i in range(Ma):
        for m in range(Mc):
            
            if (i==0 and m==0):
                theta_non_K_set = param_values[:, [0, 2]] 
                for j in range(N):
                    for k in range(Mb):
                        if k==0:
                            theta_K_set = param_values[:, 1]
                        else:
                            theta_K_set = param_values[:, 1]
                        values = np.hstack((np.repeat(theta_non_K_set[j, 0].reshape(1, -1), N, axis=0), theta_K_set.reshape(-1, 1), np.repeat(theta_non_K_set[j, 1].reshape(1, -1), N, axis=0)))
                        Y[i, m, j, k, :] = evaluate(values, delta, np.array([alpha[0, i], alpha[1, k], alpha[2, m]]), np.array([a[0, i], a[1, k], a[2, m]]))
            
            if (i==0 and m==1):
                theta_non_K_set = param_values[:, [0, 2]] 
                for j in range(N):
                    for k in range(Mb):
                        if k==0:
                            theta_K_set = param_values[:, 1]
                        else:
                            theta_K_set = param_values[:, 1]
                        values = np.hstack((np.repeat(theta_non_K_set[j, 0].reshape(1, -1), N, axis=0), theta_K_set.reshape(-1, 1), np.repeat(theta_non_K_set[j, 1].reshape(1, -1), N, axis=0)))
                        Y[i, m, j, k, :] = evaluate(values, delta, np.array([alpha[0, i], alpha[1, k], alpha[2, m]]), np.array([a[0, i], a[1, k], a[2, m]]))
          
            if (i==1 and m==0):
                theta_non_K_set = param_values[:, [0, 2]] 
                for j in range(N):
                    for k in range(Mb):
                        if k==0:
                            theta_K_set = param_values[:, 1]
                        else:
                            theta_K_set = param_values[:, 1]
                        values = np.hstack((np.repeat(theta_non_K_set[j, 0].reshape(1, -1), N, axis=0), theta_K_set.reshape(-1, 1), np.repeat(theta_non_K_set[j, 1].reshape(1, -1), N, axis=0)))
                        Y[i, m, j, k, :] = evaluate(values, delta, np.array([alpha[0, i], alpha[1, k], alpha[2, m]]), np.array([a[0, i], a[1, k], a[2, m]]))
          
            if (i==1 and m==1):
                theta_non_K_set = param_values[:, [0, 2]] 
                for j in range(N):
                    for k in range(Mb):
                        if k==0:
                            theta_K_set = param_values[:, 1]
                        else:
                            theta_K_set = param_values[:, 1]
                        values = np.hstack((np.repeat(theta_non_K_set[j, 0].reshape(1, -1), N, axis=0), theta_K_set.reshape(-1, 1), np.repeat(theta_non_K_set[j, 1].reshape(1, -1), N, axis=0)))
                        Y[i, m, j, k, :] = evaluate(values, delta, np.array([alpha[0, i], alpha[1, k], alpha[2, m]]), np.array([a[0, i], a[1, k], a[2, m]]))
 
    print('Numerical mean Y =', np.mean(Y))        
    return Y

# Compute the output difference
@nb.njit(fastmath=True)
def diff_B(Y):
    print('Computing output difference...')
    dB = np.zeros((Ma, Mc, N, Mb, N,  Mb, N), dtype=np.float32)
    dB2 = np.zeros((Ma, Mc, N, Mb, N, Mb, N), dtype=np.float32)
    for i in range(Ma):
        for m in range(Mc):
            for j in range(N):
                for k1 in range(Ma):
                    for l1 in range(N):
                        for k2 in range(Ma):
                            for l2 in range(N):
                                dB[i, m, j, k1, l1, k2, l2] = abs(Y[i, m, j, k1, l1] - Y[i, m, j, k2, l2])
                                dB2[i, m, j, k1, l1, k2, l2] = dB[i, m, j, k1, l1, k2, l2]**2
                                
    return dB, dB2
    
# Compute the sensitivity measures
@nb.njit(fastmath=True)
def mean_var_B(dB, dB2):
    print('Computing sensitivty measures...')
    E_tb = np.zeros((Ma, Mc, N, Mb, Mb), dtype=np.float32)
    E_tb2 = np.zeros((Ma, Mc, N, Mb, Mb), dtype=np.float32)
    E_b = np.zeros((Ma, Mc, N), dtype=np.float32)
    E_b2 = np.zeros((Ma, Mc, N), dtype=np.float32)
    E_tc = np.zeros((Ma, Mc), dtype=np.float32)
    E_tc2 = np.zeros((Ma, Mc), dtype=np.float32)
    E_ta = np.zeros(Ma, dtype=np.float32)
    E_ta2 = np.zeros(Ma, dtype=np.float32)
    for i in range(Ma):
        for m in range(Mc):
            for j in range(N):
                for k1 in range(Mb):
                    for k2 in range(Mb):
                        
                        E_tb[i, m, j, k1, k2] = np.mean(dB[i, m, j, k1, :, k2, :])
                        E_tb2[i, m, j, k1, k2] = np.mean(dB2[i, m, j, k1, :, k2, :])
                            
                E_b[i, m, j] = E_tb[i, m, j, 0, 0] * PMB[0] * PMB[0] + E_tb[i, m, j, 0, 1] * PMB[0] * PMB[1] + \
                               E_tb[i, m, j, 1, 0] * PMB[1] * PMB[0] + E_tb[i, m, j, 1, 1] * PMB[1] * PMB[1]
                E_b2[i, m, j] = E_tb2[i, m, j, 0, 0] * PMB[0] * PMB[0] + E_tb2[i, m, j, 0, 1] * PMB[0] * PMB[1] + \
                                E_tb2[i, m, j, 1, 0] * PMB[1] * PMB[0] + E_tb2[i, m, j, 1, 1] * PMB[1] * PMB[1]
                                 
            E_tc[i, m] = np.mean(E_b[i, m, :])
            E_tc2[i, m] = np.mean(E_b2[i, m, :])
            
        E_ta[i] = E_tc[i, 0] * PMC[0] + E_tc[i, 1] * PMC[1]
        E_ta2[i] = E_tc2[i, 0] * PMC[0] + E_tc2[i, 1] * PMC[1]
    
    E_a = E_ta[0] * PMA[0] + E_ta[1] * PMA[1]
    E_a2 = E_ta2[0] * PMA[0] + E_ta2[1] * PMA[1]
        
    V_a = E_a2 - E_a**2
    
    return E_a, V_a

# Compute the two sensitvity measures using all samples
Y = cmpt_Y_B(N)
dB, dB2 = diff_B(Y)
mean_B, var_B = mean_var_B(dB, dB2)
print('N = %d. mean_B = %.4f. var_B = %.4f' %(N, mean_B, var_B))

# Save the results to local disk
# np.save('dB_LHS_' + str(N) + '.npy', dB)
# np.save('dB2_LHS_' + str(N) + '.npy', dB2)
