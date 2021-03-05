# -*- coding: utf-8 -*-
"""
Created on Sun May 17 08:17:40 2020

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
def cmpt_Y_C(N):
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

    Y = np.zeros((Ma, Mb, N, Mc, N), dtype=np.float32)
    for i in range(Ma):
        for k in range(Mb):
            
            if (i==0 and k==0):
                w = model_R1(param_values[:, 0]) 
                hk1 = param_values[:, 2]
                hk2 = param_values[:, 2]
                for j in range(N):
                    for m in range(Mc):
                        if m==0:
                            h2 = model_M1(param_values[:, 5])
                        else:
                            h2 = model_M2(param_values[:, 6], param_values[:, 7])
                        Y[i, k, j, m, :] = analytical_dsc_solver(np.repeat(w[j], N), np.repeat(hk1[j], N), np.repeat(hk2[j], N), h2)
            
            if (i==0 and k==1):
                w = model_R1(param_values[:, 0]) 
                hk1 = param_values[:, 3]
                hk2 = param_values[:, 4]
                for j in range(N):
                    for m in range(Mc):
                        if m==0:
                            h2 = model_M1(param_values[:, 5])
                        else:
                            h2 = model_M2(param_values[:, 6], param_values[:, 7])
                        Y[i, k, j, m, :] = analytical_dsc_solver(np.repeat(w[j], N), np.repeat(hk1[j], N), np.repeat(hk2[j], N), h2)
            
            if (i==1 and k==0):
                w = model_R2(param_values[:, 1]) 
                hk1 = param_values[:, 2]
                hk2 = param_values[:, 2]
                for j in range(N):
                    for m in range(Mc):
                        if m==0:
                            h2 = model_M1(param_values[:, 5])
                        else:
                            h2 = model_M2(param_values[:, 6], param_values[:, 7])
                        Y[i, k, j, m, :] = analytical_dsc_solver(np.repeat(w[j], N), np.repeat(hk1[j], N), np.repeat(hk2[j], N), h2)
            
            if (i==1 and k==1):
                w = model_R2(param_values[:, 1]) 
                hk1 = param_values[:, 3]
                hk2 = param_values[:, 4]
                for j in range(N):
                    for m in range(Mc):
                        if m==0:
                            h2 = model_M1(param_values[:, 5])
                        else:
                            h2 = model_M2(param_values[:, 6], param_values[:, 7])
                        Y[i, k, j, m, :] = analytical_dsc_solver(np.repeat(w[j], N), np.repeat(hk1[j], N), np.repeat(hk2[j], N), h2)
            
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
np.save('dC_LHS_' + str(N) + '.npy', dC)
print('N = %d. mean_C = %.4f. var_C = %.4f' %(N, mean_C, var_C))
