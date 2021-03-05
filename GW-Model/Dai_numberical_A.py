# -*- coding: utf-8 -*-
"""
Created on Fri May 15 20:58:57 2020

@author: Jing
"""
import numpy as np
import numba as nb
from lhsdrv import lhs

# Model information
N = 20                         # Number of samples generated for each  parameter
Ma = 2                         # Number of alterantive models for recharge process
Mb = 2                         # Number of alterantive models for geloogy process
Mc = 2                         # Number of alterantive models for snow melt process
PMA = np.array([0.5, 0.5])     # Process model weight for recharge process
PMB = np.array([0.5, 0.5])     # Process model weight for geology process
PMC = np.array([0.5, 0.5])     # Process model weight for snowmelt process

# Parameters for snow melt process
P = 60                # Precipation in inch/yr
Ta = 7                # Average temperature for a given day in degree 
Tm = 0                # Critical snow melt point in degree
Csn = 0.8             # Runoff confficient
SVC = 0.7             # Snow cover fraction 
A = 2000 * 1e6        # Upper catchment area in  km2
Rn = 80               # Surface radiation in w/m2

# Left boundary condition
h1 = 300              # Head in the left 

# Domain information
z0 = 289              # Elevation of river bed in meters    
L = 10000 
x0 = 7000
nx = 101
x = np.linspace(0, L, nx, endpoint=True)

# Parameter bounds and distributions
# Parameters for snow melt process
P = 60                # Precipation in inch/yr
Ta = 7                # Average temperature for a given day in degree 
Tm = 0                # Critical snow melt point in degree
Csn = 0.8             # Runoff confficient
SVC = 0.7             # Snow cover fraction 
A = 2000 * 1e6        # Upper catchment area in  km2
Rn = 80               # Surface radiation in w/m2

# Left boundary condition
h1 = 300              # Head in the left 

# Domain information
z0 = 289              # Elevation of river bed in meters    
L = 10000   
x0 = 7000
Nx = 21
qid = 14
X = np.linspace(0, L, Nx, endpoint=True)

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

@nb.njit
def model_R1(a):
    """
    Compute recharge[m/d] using recharge model R1 by Chaturvedi(1936)
    
    """
    return a * (P - 14)**0.5 * 25.4 * 0.001 / 365

@nb.njit
def model_R2(b):
    """
    Compute recharge[m/d] using recharge model R2 by Krishna Rao (1970)
    
    """
    return b * (P - 15.7) * 25.4 * 0.001 / 365

@nb.njit
def model_M1(f1):
    """
    Compute river stage h2 [m] using degree-day method
 
    """
    M = f1 * (Ta - Tm)
    Q = Csn * M * SVC * A * 0.001 / 86400
    h2 = 0.3 * Q**0.6 + z0
    
    return h2

@nb.njit
def model_M2(f2, r):
    """
    Compute river stage h2 [m] using restricted degree-day radiation balance approach

    """
    M = f2 * (Ta - Tm) + r * Rn
    Q = Csn * M * SVC * A * 0.001 / 86400
    h2 = 0.3 * Q**0.6 + z0
    
    return h2

# Define the analytical discharge solution
@nb.njit
def analytical_dsc_solver(w, hk1, hk2, h2):
    """
    Compute discharge per unit [m2/d] at x=x0 using anaytical solution
    
    """
    C1 = (h1**2 - h2**2 - w / hk1 * x0**2 + w / hk2 * x0**2 - w / hk2 * L**2) / (hk1 / hk2 * x0 - hk1 / hk2 * L - x0)

    return w * x0 - hk1 * C1 / 2

param_values = np.load('param_values_' + str(N) + '.npy')

# Compute the model output
@nb.njit
def cmpt_Y_A(N):
    print('Computing system output...')
    '''
    # Sampling from the stratum using LHSDRV
    problem = {'nvars': 8,
               'names': ['a', 'b', 'hk', 'hk1', 'hk2', 'f1', 'f2', 'r'],
               'bounds': [bounds['a'], bounds['b'], bounds['hk'], bounds['hk1'], bounds['hk2'], bounds['f1'], bounds['f2'], bounds['r']],
               'dists': [dists['a'], dists['b'], dists['hk'], dists['hk1'], dists['hk2'], dists['f1'], dists['f2'], dists['r']]
               }
    param_values = lhs(problem, N, seed=933090934)
    
    '''
    Y = np.zeros((Mb, Mc, N, Ma, N), dtype=np.float32)
    for k in range(Mb):
        for m in range(Mc):
            
            if (k==0 and m==0):
                hk1 = param_values[:, 2]
                hk2 = param_values[:, 2]
                h2 = model_M1(param_values[:, 5])
                for l in range(N):
                    for i in range(Ma):
                        if i==0:
                            w = model_R1(param_values[:, 0])
                        else:
                            w = model_R2(param_values[:, 1])
                        Y[k, m, l, i, :] = analytical_dsc_solver(w, np.repeat(hk1[l], N), np.repeat(hk2[l], N), np.repeat(h2[l], N))
            
            if (k==0 and m==1):
                hk1 = param_values[:, 2]
                hk2 = param_values[:, 2]
                h2 = model_M2(param_values[:, 6], param_values[:, 7])
                for l in range(N):
                    for i in range(Ma):
                        if i==0:
                            w = model_R1(param_values[:, 0])
                        else:
                            w = model_R2(param_values[:, 1])
                        Y[k, m, l, i, :] = analytical_dsc_solver(w, np.repeat(hk1[l], N), np.repeat(hk2[l], N), np.repeat(h2[l], N))
            
            if (k==1 and m==0):
                hk1 = param_values[:, 3]
                hk2 = param_values[:, 4]
                h2 = model_M1(param_values[:, 5])
                for l in range(N):
                    for i in range(Ma):
                        if i==0:
                            w = model_R1(param_values[:, 0])
                        else:
                            w = model_R2(param_values[:, 1])
                        Y[k, m, l, i, :] = analytical_dsc_solver(w, np.repeat(hk1[l], N), np.repeat(hk2[l], N), np.repeat(h2[l], N))
            
            if (k==1 and m==1):
                hk1 = param_values[:, 3]
                hk2 = param_values[:, 4]
                h2 = model_M2(param_values[:, 6], param_values[:, 7])
                for l in range(N):
                    for i in range(Ma):
                        if i==0:
                            w = model_R1(param_values[:, 0])
                        else:
                            w = model_R2(param_values[:, 1])
                        Y[k, m, l, i, :] = analytical_dsc_solver(w, np.repeat(hk1[l], N), np.repeat(hk2[l], N), np.repeat(h2[l], N))
    
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

print('N = %d. mean_A = %.4f. var_A = %.4f' %(N, mean_A, var_A))

# Save the results to local disk
# np.save('dA_LHS_' + str(N) + '.npy', dA)
# np.save('dA2_LHS_' + str(N) + '.npy', dA2)















































































