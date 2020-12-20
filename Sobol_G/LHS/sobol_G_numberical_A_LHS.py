# -*- coding: utf-8 -*-
"""
Created on Fri May 15 20:58:57 2020

This scripts generates the results of the two sensitivity measures (mean and 
    variance of process g1* in Sobol's  G*-function). 
    The results is 36.64 and 9.71 used in Table 1.

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
N = 20
Ma, Mb, Mc = 2, 2, 2 
PMA = np.array([0.5, 0.5])
PMB = np.array([0.5, 0.5])
PMC = np.array([0.5, 0.5])


# Compute the model output
def cmpt_Y_A(N):
    print('Computing system output...')
    
    # Sampling from the stratum using LHS251
    problem = {'nvars': 3,
               'names': ['x1', 'y1', 'z1'],
               'bounds': [[0, 1], [0, 1], [0, 1]],
               'dists' :['UNIFORM', 'UNIFORM', 'UNIFORM']
               }
    param_values = lhs(problem, N, seed=933090934)

    Y = np.zeros((Mb, Mc, N, Ma, N), dtype=np.float32)
    for k in range(Mb):
        for m in range(Mc):
            
            if (k==0 and m==0):
                theta_non_K_set = param_values[:, [1, 2]] 
                for l in range(N):
                    for i in range(Ma):
                        if i==0:
                            theta_K_set = param_values[:, 0]
                        else:
                            theta_K_set = param_values[:, 0]
                        values = np.hstack((theta_K_set.reshape(-1, 1), np.repeat(theta_non_K_set[l, :].reshape(1, -1), N, axis=0)))
                        Y[k, m, l, i, :] = evaluate(values, delta, np.array([alpha[0, i], alpha[1, k], alpha[2, m]]), np.array([a[0, i], a[1, k], a[2, m]]))
            
            if (k==0 and m==1):
                theta_non_K_set = param_values[:, [1, 2]] 
                for l in range(N):
                    for i in range(Ma):
                        if i==0:
                            theta_K_set = param_values[:, 0]
                        else:
                            theta_K_set = param_values[:, 0]
                        values = np.hstack((theta_K_set.reshape(-1, 1), np.repeat(theta_non_K_set[l, :].reshape(1, -1), N, axis=0)))
                        Y[k, m, l, i, :] = evaluate(values, delta, np.array([alpha[0, i], alpha[1, k], alpha[2, m]]), np.array([a[0, i], a[1, k], a[2, m]]))
            
            if (k==1 and m==0):
                theta_non_K_set = param_values[:, [1, 2]] 
                for l in range(N):
                    for i in range(Ma):
                        if i==0:
                            theta_K_set = param_values[:, 0]
                        else:
                            theta_K_set = param_values[:, 0]
                        values = np.hstack((theta_K_set.reshape(-1, 1), np.repeat(theta_non_K_set[l, :].reshape(1, -1), N, axis=0)))
                        Y[k, m, l, i, :] = evaluate(values, delta, np.array([alpha[0, i], alpha[1, k], alpha[2, m]]), np.array([a[0, i], a[1, k], a[2, m]]))
            
            if (k==1 and m==1):
                theta_non_K_set = param_values[:, [1, 2]] 
                for l in range(N):
                    for i in range(Ma):
                        if i==0:
                            theta_K_set = param_values[:, 0]
                        else:
                            theta_K_set = param_values[:, 0]
                        values = np.hstack((theta_K_set.reshape(-1, 1), np.repeat(theta_non_K_set[l, :].reshape(1, -1), N, axis=0)))
                        Y[k, m, l, i, :] = evaluate(values, delta, np.array([alpha[0, i], alpha[1, k], alpha[2, m]]), np.array([a[0, i], a[1, k], a[2, m]]))
    print('Numerical mean Y =', np.mean(Y))        
    return Y

# Compute the output difference
@nb.njit(fastmath=True)
def diff_A(Y):
    print('Computing output difference...')
    dA = np.zeros((Mb, Mc, N, Ma, N,  Ma, N), dtype=np.float32)
    dA2 = np.zeros((Mb, Mc, N, Ma, N, Ma, N), dtype=np.float32)
    for k in range(Mb):
        for m in range(Mc):
            for l in range(N):
                for i1 in range(Ma):
                    for j1 in range(N):
                        for i2 in range(Ma):
                            for j2 in range(N):
                                dA[k, m, l, i1, j1, i2, j2] = abs(Y[k, m, l, i1, j1] - Y[k, m, l, i2, j2])
                                dA2[k, m, l, i1, j1, i2, j2] = dA[k, m, l, i1, j1, i2, j2]**2
                                
    return dA, dA2
    
# Compute the sensitivity measures
@nb.njit(fastmath=True)
def mean_var_A(dA, dA2):
    print('Computing sensitivty measures...')
    E_ta = np.zeros((Mb, Mc, N, Ma, Ma), dtype=np.float32)
    E_ta2 = np.zeros((Mb, Mc, N, Ma, Ma), dtype=np.float32)
    E_a = np.zeros((Mb, Mc, N), dtype=np.float32)
    E_a2 = np.zeros((Mb, Mc, N), dtype=np.float32)
    E_tc = np.zeros((Mb, Mc), dtype=np.float32)
    E_tc2 = np.zeros((Mb, Mc), dtype=np.float32)
    E_tb = np.zeros(Mb, dtype=np.float32)
    E_tb2 = np.zeros(Mb, dtype=np.float32)
    
    for k in range(Mb):
        for m in range(Mc):
            for l in range(N):
                for i1 in range(Ma):
                    for i2 in range(Ma):
                        
                        E_ta[k, m, l, i1, i2] = np.mean(dA[k, m, l, i1, :, i2, :])
                        E_ta2[k, m, l, i1, i2] = np.mean(dA2[k, m, l, i1, :, i2, :])
                            
                E_a[k, m, l] = E_ta[k, m, l, 0, 0] * PMA[0] * PMA[0] + E_ta[k, m, l, 0, 1] * PMA[0] * PMA[1] + \
                               E_ta[k, m, l, 1, 0] * PMA[1] * PMA[0] + E_ta[k, m, l, 1, 1] * PMA[1] * PMA[1]
                E_a2[k, m, l] = E_ta2[k, m, l, 0, 0] * PMA[0] * PMA[0] + E_ta2[k, m, l, 0, 1] * PMA[0] * PMA[1] + \
                                E_ta2[k, m, l, 1, 0] * PMA[1] * PMA[0] + E_ta2[k, m, l, 1, 1] * PMA[1] * PMA[1]
                                
            E_tc[k, m] = np.mean(E_a[k, m, :])
            E_tc2[k, m] = np.mean(E_a2[k, m, :])
        E_tb[k] = E_tc[k, 0] * PMC[0] + E_tc[k, 1] * PMC[1]
        E_tb2[k] = E_tc2[k, 0] * PMC[1] + E_tc2[k, 1] * PMC[1]
    
    E_b = E_tb[0] * PMB[0] + E_tb[1] * PMB[1]
    E_b2 = E_tb2[0] * PMB[0] + E_tb2[1] * PMB[1]
        
    V_b = E_b2 - E_b**2
    return E_b, V_b

# Compute the two sensitvity measures using all samples
Y = cmpt_Y_A(N)
dA, dA2 = diff_A(Y)
mean_A, var_A = mean_var_A(dA, dA2)
np.save('dA_LHS_' + str(N) + '.npy', dA)
np.save('dA2_LHS_' + str(N) + '.npy', dA2)
print('N = %d. mean_A = %.4f. var_A = %.4f' %(N, mean_A, var_A))


'''
# Convergence test
print('Convergence test...')
N = 1000
mean_A = np.zeros(len(range(10, N, 10)))
var_A = np.zeros(len(range(10, N, 10)))
for idn, n in enumerate(range(10, N, 10)):
    Y = cmpt_Y_A(n)
    dA, dA2 = diff_A(Y)
    mean_A[idn] = np.mean(dA)
    var_A[idn] = np.var(dA)
    if n % 100 ==0:
        print('    n = %d. mean_A = %.4f. var_A = %.4f' %(n, mean_A[idn], var_A[idn]))
             
# Save mean and variacne  
np.save('mean_A_LHS_' + str(N) + '.npy', mean_A)
np.save('var_A_LHS_' + str(N) + '.npy', var_A)

# Plot the convergence
plt.figure(figsize=(8, 3))
plt.plot(range(10, N, 10), mean_A, label='Direct Monte Carlo')
plt.axhline(0.3728, color='#E24A33', label='Analytical')
plt.xlabel('Sample number', fontsize=12)
plt.ylabel('$E(d\Delta|g_1^*)$', fontsize=12)
plt.legend(loc='best')
plt.xlim(10, N)
plt.ylim(0.36, 0.40)
plt.xticks([10, 2000, 6000, 8000, 10000])
plt.savefig('convergence_A_LHS.png', dpi=300)
plt.show()
'''
