
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

# Bk4_Ch11_02.py

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

# generate original data matrix, X
np.random.default_rng()
X = np.random.randn(9, 5)

#%% QR decomposition， complete version

Q_complete, R_complete = np.linalg.qr(X, mode = 'complete')


fig, axs = plt.subplots(1, 5, figsize=(12, 3))

plt.sca(axs[0])
ax = sns.heatmap(X,cmap='RdBu_r',vmax = 2.5,vmin = -2.5,
                 cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('X')

plt.sca(axs[1])
plt.title('=')
plt.axis('off')

plt.sca(axs[2])
ax = sns.heatmap(Q_complete,cmap='RdBu_r',vmax = 2.5,vmin = -2.5,
                 cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('Qc')

plt.sca(axs[3])
plt.title('@')
plt.axis('off')

plt.sca(axs[4])
ax = sns.heatmap(R_complete,cmap='RdBu_r',vmax = 2.5,vmin = -2.5,
                 cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('Rc')

# properties of Q (reduced)

fig, axs = plt.subplots(1, 5, figsize=(12, 3))

plt.sca(axs[0])
ax = sns.heatmap(Q_complete.T@Q_complete,cmap='RdBu_r',vmax = 2.5,vmin = -2.5,
                 cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('Qc.T@Qc')

plt.sca(axs[1])
plt.title('=')
plt.axis('off')

plt.sca(axs[2])
ax = sns.heatmap(Q_complete.T,cmap='RdBu_r',vmax = 2.5,vmin = -2.5,
                 cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('Qc.T')

plt.sca(axs[3])
plt.title('@')
plt.axis('off')

plt.sca(axs[4])
ax = sns.heatmap(Q_complete,cmap='RdBu_r',vmax = 2.5,vmin = -2.5,
                 cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('Qc')

#%% QR decomposition， reduced version

Q, R = np.linalg.qr(X)
# default: reduced

fig, axs = plt.subplots(1, 5, figsize=(12, 3))

plt.sca(axs[0])
ax = sns.heatmap(X,cmap='RdBu_r',vmax = 2.5,vmin = -2.5,
                 cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('X')

plt.sca(axs[1])
plt.title('=')
plt.axis('off')

plt.sca(axs[2])
ax = sns.heatmap(Q,cmap='RdBu_r',vmax = 2.5,vmin = -2.5,
                 cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('Q')

plt.sca(axs[3])
plt.title('@')
plt.axis('off')

plt.sca(axs[4])
ax = sns.heatmap(R,cmap='RdBu_r',vmax = 2.5,vmin = -2.5,
                 cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('R')

# properties of Q (reduced)

fig, axs = plt.subplots(1, 5, figsize=(12, 3))

plt.sca(axs[0])
ax = sns.heatmap(Q.T@Q,cmap='RdBu_r',vmax = 2.5,vmin = -2.5,
                 cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('Q.T@Q')

plt.sca(axs[1])
plt.title('=')
plt.axis('off')

plt.sca(axs[2])
ax = sns.heatmap(Q.T,cmap='RdBu_r',vmax = 2.5,vmin = -2.5,
                 cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('Q.T')

plt.sca(axs[3])
plt.title('@')
plt.axis('off')

plt.sca(axs[4])
ax = sns.heatmap(Q,cmap='RdBu_r',vmax = 2.5,vmin = -2.5,
                 cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('Q')
