# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 20:12:19 2020

@author: Jing
"""
import numpy as np
import pandas as pd
from itertools import chain
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('default')
Sobol_G_param_values = np.load('Sobol_G_param_values_binning_100.npy')
Dai_param_values = np.load('Dai_param_values_bin_100.npy')

hk_inf = np.exp(stats.norm.ppf(0.999, loc=2.9, scale=0.5))
f1_inf = stats.norm.ppf(0.999, loc=3.5, scale=0.75)
f2_inf = stats.norm.ppf(0.999, loc=2.5, scale=0.3)
r_inf = stats.norm.ppf(0.999, loc=0.3, scale=0.05)

bin_hk = [3.88, 14.65, 22.54, hk_inf]
bin_f1 = [1.18, 3.18, 3.82, f1_inf]
bin_b = [0.2, 0.3, 0.4, 0.5]
bin_f2 = [1.57, 2.37078, 2.62922, f2_inf]
bin_r = [0.15, 0.278464, 0.321526, r_inf]


# 1D groundwter flow
g = sns.jointplot(Dai_param_values[:, 2], Dai_param_values[:, 5], 
                  marginal_kws=dict(bins=20, rug=False), kind='reg',
                  height=5, space=0)

#Clear the axes containing the scatter plot
g.ax_joint.cla()

#Plot each individual point separately
g.ax_joint.scatter(Dai_param_values[:, 2], Dai_param_values[:, 5], 
                    color='#1F77B4', marker='o', s=15)

plt.xlabel('$K$', fontsize=14)
plt.ylabel('$f_1$', fontsize=14)
plt.xlim(3.88, 85.21)
plt.ylim(1.18, 5.82)
plt.xticks(bin_hk, ['3.88', '14.65', '22.54', '85.21'], rotation=30)
plt.yticks(bin_f1, ['1.18', '3.18', '3.82', '5.82'])
plt.text(7.95, 5.45, '(b)', fontsize=16)
plt.axhline(y=3.82, linestyle=':', linewidth=2, alpha=0.6)
plt.axhline(y=3.18, linestyle=':', linewidth=2, alpha=0.6)
plt.axvline(x=14.65, linestyle=':', linewidth=2, alpha=0.6)
plt.axvline(x=22.54, linestyle=':', linewidth=2, alpha=0.6)

plt.savefig('Fig6b.pdf', dpi=300)

# Sobl-G
g = sns.jointplot(Sobol_G_param_values[:, 1], Sobol_G_param_values[:, 2], 
                  marginal_kws=dict(bins=25, rug=False), kind='reg',
                  height=5, space=0)

for line in chain(g.ax_marg_x.axes.lines,g.ax_marg_y.axes.lines):
    line.set_linestyle('-')
    # line.set_linewidth(2)
    line.set_color('none')
g.ax_marg_x.plot([0, 1], [1, 1])
g.ax_marg_y.plot([1, 1], [0, 1])

#Clear the axes containing the scatter plot
g.ax_joint.cla()

#Plot each individual point separately
g.ax_joint.scatter(Sobol_G_param_values[:, 1], Sobol_G_param_values[:, 2], 
                   color='#1F77B4', marker='o', s=15)

plt.xlabel('$X_2$', fontsize=14)
plt.ylabel('$X_3$', fontsize=14)
plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0.00', '0.20', '0.40', '0.60', '0.80', '1.00'], rotation=30)
plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0.00', '0.20', '0.40', '0.60', '0.80', '1.00'])
plt.xlim(0, 1)
plt.ylim(0, 1)

plt.text(0.05, 0.92, '(a)', fontsize=16)
plt.axhline(y=0.2, linestyle=':', linewidth=2, alpha=0.6)
plt.axhline(y=0.4, linestyle=':', linewidth=2, alpha=0.6)
plt.axhline(y=0.6, linestyle=':', linewidth=2, alpha=0.6)
plt.axhline(y=0.8, linestyle=':', linewidth=2, alpha=0.6)


plt.axvline(x=0.2, linestyle=':', linewidth=2, alpha=0.6)
plt.axvline(x=0.4, linestyle=':', linewidth=2, alpha=0.6)
plt.axvline(x=0.6, linestyle=':', linewidth=2, alpha=0.6)
plt.axvline(x=0.8, linestyle=':', linewidth=2, alpha=0.6)

plt.savefig('Fig6a.pdf', dpi=300)
