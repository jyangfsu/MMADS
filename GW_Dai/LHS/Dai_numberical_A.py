# -*- coding: utf-8 -*-
"""
Created on Fri May 15 20:58:57 2020

@author: Jing
"""
import numpy as np
import numba as nb

# Model information
N = 100                        # Number of samples generated for each  parameter
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

# Load parameters
param_values = np.load('param_values_' + str(N) + '.npy')

# Compute the model output
@nb.njit
def cmpt_Y_A(N):
    print('Computing system output...')

    Y = np.zeros((Mb, Mc, N, Ma, N))
    
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
@nb.njit
def diff_A(Y):
    print('Computing output difference...')
    dA = np.zeros((Mb, Mc, N, Ma, Ma))
    dA2 = np.zeros((Mb, Mc, N, Ma, Ma))
    tmp_dA = np.zeros((N, N))
    for k in range(Mb):
        for m in range(Mc):
            for l in range(N):
                print('    With k =', k, 'm =', m, 'l =', l)
                for i1 in range(Ma):
                    for i2 in range(Ma):
                        for j1 in nb.prange(N):
                            for j2 in nb.prange(N):
                                tmp_dA[j1, j2] = abs(Y[k, m, l, i1, j1] - Y[k, m, l, i2, j2])
                        dA[k, m, l, i1, i2] = np.mean(tmp_dA)
                        dA2[k, m, l, i1, i2] = np.mean(tmp_dA**2)
                        
    return dA, dA2

# Compute the two sensitvity measures
Y = cmpt_Y_A(N)
dA, dA2 = diff_A(Y)
# np.save('dA_' + str(N) + '.npy', dA)
# np.save('dA2_' + str(N) + '.npy', dA2)

# Print info
print('============ Considering process model uncertainty ============')
mean_A = np.mean(dA)
var_A = np.mean(dA2) - mean_A**2
print('N = %d. mean_A = %.4f. var_A = %.4f' %(N, mean_A, var_A))

print('======== Without Considering process model uncertainty ========')
for k in range(Mb):
    for m in range(Mc):
        for i in range(Ma):
            mean_A = np.mean(dA[k, m, :, i, i])
            var_A = np.mean(dA2[k, m, :, i, i]) - mean_A**2
            print('k=%d, m=%d, i=%d. mean_A = %.4f. var_A = %.4f' %(k, m, i, mean_A, var_A))
