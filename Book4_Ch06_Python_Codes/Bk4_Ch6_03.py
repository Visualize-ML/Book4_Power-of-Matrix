
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############


# Bk4_Ch6_03.py

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

A = np.array([[-1,  1],
              [0.7, -0.4]])

B = np.array([[0.5, -0.6],
             [-0.8, 0.3]])

A_kron_B = np.kron(A, B)

fig, axs = plt.subplots(1, 5, figsize=(12, 5))

plt.sca(axs[0])
ax = sns.heatmap(A,cmap='RdYlBu_r',vmax = 1,vmin = -1,
                 cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('A')

plt.sca(axs[1])
plt.title('$\otimes$')
plt.axis('off')

plt.sca(axs[2])
ax = sns.heatmap(B,cmap='RdYlBu_r',vmax = 1,vmin = -1,
                 cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('B')

plt.sca(axs[3])
plt.title('=')
plt.axis('off')

plt.sca(axs[4])
ax = sns.heatmap(A_kron_B,cmap='RdYlBu_r',vmax = 1,vmin = -1,
                 cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('C')
