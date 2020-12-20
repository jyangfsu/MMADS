# -*- coding: utf-8 -*-
"""
Created on Sun May 17 08:17:40 2020

@author: Jing
"""
import numpy as np
import numba as nb
from sklearn.metrics import pairwise_distances

# Model information
N = 100                       # Number of samples generated for each  parameter
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
    # np.float32 is not right
    Y = np.zeros((Ma, Mb, N, Mc, N), dtype=np.float64)
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
@nb.njit
def diff_C(Y):
    print('Computing output difference...')
    dC = np.zeros((Ma, Mb, N, Mc, Mc))
    dC2 = np.zeros((Ma, Mb, N, Mc, Mc))
    tmp_dA = np.zeros((N, N))
    for i in range(Ma):
        for k in range(Mb):
            for j in range(N):
                print('    With i =', i, 'k =', k, 'j =', j)
                for m1 in range(Mb):
                    for m2 in range(Mb):
                        for n1 in nb.prange(N):
                            for n2 in nb.prange(N):
                                tmp_dA[n1, n2] = abs(Y[i, k, j, m1, n1] - Y[i, k, j, m2, n2])
                        dC[i, k, j, m1, m2] = np.mean(tmp_dA)
                        dC2[i, k, j, m1, m2] = np.mean(tmp_dA**2)

    return dC, dC2

# Compute the two sensitvity measures
Y = cmpt_Y_C(N)
dC, dC2 = diff_C(Y)
# np.save('dC_' + str(N) + '.npy', dC)
# np.save('dC2_' + str(N) + '.npy', dC2)

# Print info
print('============ Considering process model uncertainty ============')
mean_C = np.mean(dC)
var_C = np.mean(dC2) - mean_C**2
print('N = %d. mean_C = %.4f. var_C = %.4f' %(N, mean_C, var_C))

print('======== Without Considering process model uncertainty ========')
for i in range(Ma):
    for k in range(Mb):
        for m in range(Mc):
            mean_C = np.mean(dC[i, k, :, m, m])
            var_C = np.mean(dC2[i, k, :, m, m]) - mean_C**2
            print('i=%d, k=%d, m=%d. mean_C = %.4f. var_C = %.4f' %(i, k, m, mean_C, var_C))


