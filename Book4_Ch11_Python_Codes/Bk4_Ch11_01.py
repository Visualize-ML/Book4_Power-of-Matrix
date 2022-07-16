
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

# Bk4_Ch11_01.py

import numpy as np
import seaborn as sns
from scipy.linalg import lu
from matplotlib import pyplot as plt

A = np.array([[ 5,  2, -2,  3], 
              [-2,  5, -8,  7], 
              [ 7, -5,  1, -6], 
              [-5,  4, -4,  8]])

P,L,U = lu(A, permute_l = False)
# P, permutation matrix
# L, lower triangular with unit diagonal elements
# U, upper triangular
# Default do not perform the multiplication P*L

fig, axs = plt.subplots(1, 7, figsize=(12, 3))

plt.sca(axs[0])
ax = sns.heatmap(A,cmap='RdBu_r',vmax = 10,vmin = -10,
                 cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('A')

plt.sca(axs[1])
plt.title('=')
plt.axis('off')

plt.sca(axs[2])
ax = sns.heatmap(P,cmap='RdBu_r',vmax = 10,vmin = -10,
                 cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('P')

plt.sca(axs[3])
plt.title('@')
plt.axis('off')

plt.sca(axs[4])
ax = sns.heatmap(L,cmap='RdBu_r',vmax = 10,vmin = -10,
                 cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('L')

plt.sca(axs[5])
plt.title('@')
plt.axis('off')

plt.sca(axs[6])
ax = sns.heatmap(U,cmap='RdBu_r', vmax = 10,vmin = -10,
                 cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('U')
