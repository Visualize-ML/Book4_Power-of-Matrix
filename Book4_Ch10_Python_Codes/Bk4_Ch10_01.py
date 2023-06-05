
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

# Bk4_Ch10_01.py

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd  
from sklearn.datasets import load_iris

# A copy from Seaborn
iris = load_iris()

X = iris.data
y = iris.target

feature_names = ['Sepal length, x1','Sepal width, x2',
                 'Petal length, x3','Petal width, x4']

# Convert X array to dataframe
X_df = pd.DataFrame(X, columns=feature_names)

#%% Original data, X

X = X_df.to_numpy();

# Gram matrix, G and orthogonal basis V

G = X.T@X
D, V = np.linalg.eig(G)

#%%

def heatmap(Matrices,Titles,Ranges,Equal_tags):
    
    M1 = Matrices[0]
    M2 = Matrices[1]
    M3 = Matrices[2]
    
    Title_1 = Titles[0]
    Title_2 = Titles[1]
    Title_3 = Titles[2]
    
    fig, axs = plt.subplots(1, 5, figsize=(12, 3))
    
    plt.sca(axs[0])
    ax = sns.heatmap(M1,cmap='RdYlBu_r',
                     vmin = Ranges[0][0], 
                     vmax = Ranges[0][1],
                     cbar=False,
                     xticklabels=False,
                     yticklabels=False)
    
    if Equal_tags[0] == True:
        ax.set_aspect("equal")
        
    plt.title(Title_1)
    
    plt.sca(axs[1])
    plt.title('=')
    plt.axis('off')
    
    plt.sca(axs[2])
    ax = sns.heatmap(M2,cmap='RdYlBu_r',
                     vmin = Ranges[1][0], 
                     vmax = Ranges[1][1],
                     cbar=False,
                     xticklabels=False,
                     yticklabels=False)
    if Equal_tags[1] == True:
        ax.set_aspect("equal")
    plt.title(Title_2)
    
    plt.sca(axs[3])
    plt.title('@')
    plt.axis('off')
    
    plt.sca(axs[4])
    ax = sns.heatmap(M3,cmap='RdYlBu_r',
                     vmin = Ranges[2][0], 
                     vmax = Ranges[2][1],
                     cbar=False,
                     xticklabels=False,
                     yticklabels=False)
    
    if Equal_tags[2] == True:
        ax.set_aspect("equal")
    plt.title(Title_3)

#%%

def plot_four_figs(X,v_j,idx):

    # Fig 1: X@v_j = z_j
    
    z_j = X@v_j
    Titles = ['$X$',
              '$v_' + str(idx) + '$',
              '$z_' + str(idx) + '$']
    
    Ranges = [[-2,11],
              [-1,1],
              [-2,11]]
    
    Equal_tags = [False,True,False]
    heatmap([X,v_j,z_j],Titles,Ranges,Equal_tags)

    # Fig 2: z@v_j.T = X_j
    X_j = z_j@v_j.T
    Titles = ['$z_' + str(idx) + '$',
              '$v_' + str(idx) + '^T$',
              '$X_' + str(idx) + '$']
    
    Ranges = [[-2,11],
              [-1,1],
              [-2,11]]
    
    Equal_tags = [False,True,False]
    
    heatmap([z_j,v_j.T,X_j],Titles,Ranges,Equal_tags)

    # Fig 3: T_j = v_j@v_j.T
    T_j = v_j@v_j.T
    
    Titles = ['$v_' + str(idx) + '$',
              '$v_' + str(idx) + '^T$',
              '$T_' + str(idx) + '$']
    
    Ranges = [[-1,1],
              [-1,1],
              [-1,1]]
    
    Equal_tags = [True,True,True]
    
    heatmap([v_j,v_j.T,T_j],Titles,Ranges,Equal_tags)

    
    # Fig 4: X@T_j = X_j
    
    T_j = X@T_j
    
    Titles = ['$X$',
              '$T_' + str(idx) + '$',
              '$X_' + str(idx) + '$']
    
    Ranges = [[-2,11],
              [-1,1],
              [-2,11]]
    
    Equal_tags = [False,True,False]
    
    heatmap([X,T_j,X_j],Titles,Ranges,Equal_tags)


#%% First basis vector

v1 = V[:, 0].reshape((-1, 1))

plot_four_figs(X,v1,1)

#%% Second basis vector

v2 = V[:, 1].reshape((-1, 1))

plot_four_figs(X,v2,2)

#%% Third basis vector

v3 = V[:, 2].reshape((-1, 1))

plot_four_figs(X,v3,3)

#%% Fourth basis vector

v4 = V[:, 3].reshape((-1, 1))

plot_four_figs(X,v4,4)

