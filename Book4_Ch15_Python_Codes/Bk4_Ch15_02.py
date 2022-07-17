
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

# Bk4_Ch15_02_A

import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'RdBu_r'
import seaborn as sns
PRECISION = 3

def svd(X):

    full_matrices = True
    
    U, s, Vt = np.linalg.svd(X,full_matrices = full_matrices)

    # Put the vector singular values into a padded matrix
    
    if full_matrices:
        S = np.zeros(X.shape)
        np.fill_diagonal(S, s)
    else:
        S = np.diag(s)

    # Rounding for display
    return np.round(U, PRECISION), np.round(S, PRECISION), np.round(Vt.T, PRECISION)


def visualize_svd(X,title_X,title_U,title_S,title_V, fig_height=5):

    # Run SVD, as defined above
    U, S, V = svd(X)
    
    all_ = np.r_[X.flatten(order='C'),U.flatten(order='C'),
                 S.flatten(order='C'),V.flatten(order='C')]
    
    # all_max = max(all_.max(),all_.min())
    # all_min = -max(all_.max(),all_.min())
    all_max = 6
    all_min = -6
    # Visualization
    fig, axs = plt.subplots(1, 7, figsize=(12, fig_height))

    plt.sca(axs[0])
    ax = sns.heatmap(X,cmap='RdBu_r',vmax = all_max,vmin = all_min,
                     cbar_kws={"orientation": "horizontal"})
    ax.set_aspect("equal")
    plt.title(title_X)

    plt.sca(axs[1])
    plt.title('@')
    plt.axis('off')

    plt.sca(axs[2])
    ax = sns.heatmap(U,cmap='RdBu_r',vmax = all_max,vmin = all_min,
                     cbar_kws={"orientation": "horizontal"})
    ax.set_aspect("equal")
    plt.title(title_U)

    plt.sca(axs[3])
    plt.title('@')
    plt.axis('off')

    plt.sca(axs[4])
    ax = sns.heatmap(S,cmap='RdBu_r',vmax = all_max,vmin = all_min,
                     cbar_kws={"orientation": "horizontal"})
    ax.set_aspect("equal")
    plt.title(title_S)

    plt.sca(axs[5])
    plt.title('@')
    plt.axis('off')

    plt.sca(axs[6])
    ax = sns.heatmap(V.T,cmap='RdBu_r',vmax = all_max,vmin = all_min, 
                     cbar_kws={"orientation": "horizontal"})
    ax.set_aspect("equal")
    plt.title(title_V)
    return X, U, S, V

# Repeatability
np.random.seed(1)

# Generate random matrix
X = np.random.randn(6, 4)

# manipulate X and reduce rank to 3
# X[:,3] = X[:,0] + X[:,1]

X, U, S, V = visualize_svd(X,'$X$','$U$','$S$','$V^T$', fig_height=3)

X_2, U_2, S_2, V_2 = visualize_svd(X.T@X,'$X^TX$','$V$','$S^TS$','$V^T$', fig_height=3)

X_3, U_3, S_3, V_3 = visualize_svd(X@X.T,'$XX^T$','$U$','$SS^T$','$U^T$', fig_height=3)

#%%

# Bk4_Ch15_02_B

#%% U*U.T = I

all_max = 6
all_min = -6

fig, axs = plt.subplots(1, 3, figsize=(12, 3))

plt.sca(axs[0])
ax = sns.heatmap(U,cmap='RdBu_r',vmax = all_max,vmin = all_min,
                 cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('$U$')


plt.sca(axs[1])
ax = sns.heatmap(U.T,cmap='RdBu_r',vmax = all_max,vmin = all_min,
                 cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('$U^T$')

plt.sca(axs[2])
ax = sns.heatmap(U@U.T,cmap='RdBu_r',vmax = all_max,vmin = all_min,
                 cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('$I$')

#%%

# Bk4_Ch15_02_C

#%% V*V.T = I

all_max = 6
all_min = -6

fig, axs = plt.subplots(1, 3, figsize=(12, 3))

plt.sca(axs[0])
ax = sns.heatmap(V,cmap='RdBu_r',vmax = all_max,vmin = all_min,
                 cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('$V$')

plt.sca(axs[1])
ax = sns.heatmap(V.T,cmap='RdBu_r',vmax = all_max,vmin = all_min,
                 cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('$V^T$')

plt.sca(axs[2])
ax = sns.heatmap(V@V.T,cmap='RdBu_r',vmax = all_max,vmin = all_min,
                 cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('$I$')

#%%

# Bk4_Ch15_02_D

#%% analysis of singular value matrix

fig, axs = plt.subplots(1, 4, figsize=(12, 3))

for j in [0, 1, 2, 3]:
    X_j = S[j,j]*U[:,j][:, None]@V[:,j][None, :];
    plt.sca(axs[j])
    ax = sns.heatmap(X_j,cmap='RdBu_r',vmax = all_max,vmin = all_min,
                     cbar_kws={"orientation": "horizontal"})
    ax.set_aspect("equal")
    title_txt = '$s_'+ str(j+1) + 'u_'+ str(j+1) + 'v_'+ str(j+1) + '^T$'
    plt.title(title_txt)

#%% projection

for j in [0, 1, 2, 3]:
    
    fig, axs = plt.subplots(1, 7, figsize=(12, 3))
    
    v_j = V[:,j]
    v_j = np.matrix(v_j).T
    s_j = S[j,j]
    s_j = np.matrix(s_j)
    u_j = U[:,j]
    u_j = np.matrix(u_j).T
    
    plt.sca(axs[0])
    ax = sns.heatmap(X,cmap='RdBu_r',vmax = all_max,vmin = all_min,
                     cbar_kws={"orientation": "horizontal"})
    ax.set_aspect("equal")
    plt.title('X')
    
    plt.sca(axs[1])
    plt.title('@')
    plt.axis('off')
    
    plt.sca(axs[2])
    ax = sns.heatmap(v_j,cmap='RdBu_r',vmax = all_max,vmin = all_min,
                     cbar_kws={"orientation": "horizontal"})
    ax.set_aspect("equal")
    plt.title('v_'+ str(j+1))
    
    plt.sca(axs[3])
    plt.title('=')
    plt.axis('off')
    
    plt.sca(axs[4])
    ax = sns.heatmap(s_j,cmap='RdBu_r',vmax = all_max,vmin = all_min,
                     cbar_kws={"orientation": "horizontal"})
    ax.set_aspect("equal")
    plt.title('s_'+ str(j+1))
    
    plt.sca(axs[5])
    plt.title('@')
    plt.axis('off')
    
    plt.sca(axs[6])
    ax = sns.heatmap(u_j,cmap='RdBu_r', vmax = all_max,vmin = all_min,
                     cbar_kws={"orientation": "horizontal"})
    ax.set_aspect("equal")
    plt.title('u_'+ str(j+1))
