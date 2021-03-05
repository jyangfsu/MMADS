# -*- coding: utf-8 -*-
"""
Created on Fri May 15 20:58:57 2020

@author: Jing
"""
import numpy as np
import numba as nb
import matplotlib.pyplot as plt

from latin251 import lhs
from sobol_G_fun import evaluate

# Consider three processes each of which with two alternative process models
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
def cmpt_Y_C(N):
    print('Computing system output...')
    
    # Sampling from the stratum using LHS251
    problem = {'nvars': 3,
               'names': ['x1', 'y1', 'z1'],
               'bounds': [[0, 1], [0, 1], [0, 1]],
               'dists' :['UNIFORM', 'UNIFORM', 'UNIFORM']
               }
    param_values = lhs(problem, N, seed=933090934)

    Y = np.zeros((Ma, Mb, N, Mc, N), dtype=np.float32)
    for i in range(Ma):
        for k in range(Mb):
            
            if (i==0 and k==0):
                theta_non_K_set = param_values[:, [0, 1]] 
                for j in range(N):
                    for m in range(Mc):
                        if m==0:
                            theta_K_set = param_values[:, 0]
                        else:
                            theta_K_set = param_values[:, 0]
                        values = np.hstack((np.repeat(theta_non_K_set[j, :].reshape(1, -1), N, axis=0), theta_K_set.reshape(-1, 1)))
                        Y[i, k, j, m, :] = evaluate(values, delta, np.array([alpha[0, i], alpha[1, k], alpha[2, m]]), np.array([a[0, i], a[1, k], a[2, m]]))
            
            if (i==0 and k==1):
                theta_non_K_set = param_values[:, [0, 1]] 
                for j in range(N):
                    for m in range(Mc):
                        if m==0:
                            theta_K_set = param_values[:, 0]
                        else:
                            theta_K_set = param_values[:, 0]
                        values = np.hstack((np.repeat(theta_non_K_set[j, :].reshape(1, -1), N, axis=0), theta_K_set.reshape(-1, 1)))
                        Y[i, k, j, m, :] = evaluate(values, delta, np.array([alpha[0, i], alpha[1, k], alpha[2, m]]), np.array([a[0, i], a[1, k], a[2, m]]))
            
            if (i==1 and k==0):
                theta_non_K_set = param_values[:, [0, 1]] 
                for j in range(N):
                    for m in range(Mc):
                        if m==0:
                            theta_K_set = param_values[:, 0]
                        else:
                            theta_K_set = param_values[:, 0]
                        values = np.hstack((np.repeat(theta_non_K_set[j, :].reshape(1, -1), N, axis=0), theta_K_set.reshape(-1, 1)))
                        Y[i, k, j, m, :] = evaluate(values, delta, np.array([alpha[0, i], alpha[1, k], alpha[2, m]]), np.array([a[0, i], a[1, k], a[2, m]]))
            
            if (i==1 and k==1):
                theta_non_K_set = param_values[:, [0, 1]] 
                for j in range(N):
                    for m in range(Mc):
                        if m==0:
                            theta_K_set = param_values[:, 0]
                        else:
                            theta_K_set = param_values[:, 0]
                        values = np.hstack((np.repeat(theta_non_K_set[j, :].reshape(1, -1), N, axis=0), theta_K_set.reshape(-1, 1)))
                        Y[i, k, j, m, :] = evaluate(values, delta, np.array([alpha[0, i], alpha[1, k], alpha[2, m]]), np.array([a[0, i], a[1, k], a[2, m]]))
            
    print('Numerical mean Y =', np.mean(Y))        
    return Y

# Compute the output difference
@nb.njit(fastmath=True)
def diff_C(Y):
    print('Computing output difference...')
    dC = np.zeros((Ma, Mb, N, Mc, N, Mc, N), dtype=np.float32)
    dC2 = np.zeros((Ma, Mb, N, Mc, N, Mc, N), dtype=np.float32)
    for i in range(Ma):
        for k in range(Mb):
            for j in range(N):
                for m1 in range(Mc):
                    for n1 in range(N):
                        for m2 in range(Mc):
                            for n2 in range(N):
                                dC[i, k, j, m1, n1, m2, n2] = abs(Y[i, k, j, m1, n1] - Y[i, k, j, m2, n2])
                                dC2[i, k, j, m1, n1, m2, n2] = dC[i, k, j, m1, n1, m2, n2]**2
                                
    return dC, dC2
    
# Compute the sensitivity measures
@nb.njit(fastmath=True)
def mean_var_C(dC, dC2):
    E_tc = np.zeros((Ma, Mb, N, Mc, Mc), dtype=np.float32)
    E_tc2 = np.zeros((Ma, Mb, N, Mc, Mc), dtype=np.float32)
    E_c = np.zeros((Ma, Mb, N), dtype=np.float32)
    E_c2 = np.zeros((Ma, Mb, N), dtype=np.float32)
    E_tb = np.zeros((Ma, Mb), dtype=np.float32)
    E_tb2 = np.zeros((Ma, Mb), dtype=np.float32)
    E_ta = np.zeros(Ma, dtype=np.float32)
    E_ta2 = np.zeros(Ma, dtype=np.float32)
    
    for i in range(Ma):
        for k in range(Mb):
            for j in range(N):
                for m1 in range(Mc):
                    for m2 in range(Mc):
                        
                        E_tc[i, k, j, m1, m2] = np.mean(dC[i, k, j, m1, :, m2, :])
                        E_tc2[i, k, j, m1, m2] = np.mean(dC2[i, k, j, m1, :, m2, :])
                            
                E_c[i, k, j] = E_tc[i, k, j, 0, 0] * PMC[0] * PMC[0] + E_tc[i, k, j, 0, 1] * PMC[0] * PMC[1] + \
                               E_tc[i, k, j, 1, 0] * PMC[1] * PMC[0] + E_tc[i, k, j, 1, 1] * PMC[1] * PMC[1]
                E_c2[i, k, j] = E_tc2[i, k, j, 0, 0] * PMC[0] * PMC[0] + E_tc2[i, k, j, 0, 1] * PMC[0] * PMC[1] + \
                                E_tc2[i, k, j, 1, 0] * PMC[1] * PMC[0] + E_tc2[i, k, j, 1, 1] * PMC[1] * PMC[1]
            E_tb[i, k] = np.mean(E_c[i, k, :])
            E_tb2[i, k] = np.mean(E_c2[i, k, :])
            
        E_ta[i] = E_tb[i, 0] * PMB[0] + E_tb[i, 1] * PMB[1]
        E_ta2[i] = E_tb2[i, 0] * PMB[0] + E_tb2[i, 1] * PMB[1]
    
    E_a = E_ta[0] * PMA[0] + E_ta[1] * PMA[1]
    E_a2 = E_ta2[0] * PMA[0] + E_ta2[1] * PMA[1]
        
    V_a = E_a2 - E_a**2
    return E_a, V_a

# Compute the two sensitvity measures using all samples
Y = cmpt_Y_C(N)
dC, dC2 = diff_C(Y)
mean_C, var_C = mean_var_C(dC, dC2)

print('N = %d. mean_C = %.4f. var_C = %.4f' %(N, mean_C, var_C))

# Save the results to local disk
# np.save('dC_LHS_' + str(N) + '.npy', dC)
# np.save('dC2_LHS_' + str(N) + '.npy', dC2)
