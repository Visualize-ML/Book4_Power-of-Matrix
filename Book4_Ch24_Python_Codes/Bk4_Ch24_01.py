
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

# Bk4_Ch24_01_A

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

#%% Gram matrix, G

G = X.T@X

#%% Cosine similarity matrix, C

# from sklearn.metrics.pairwise import cosine_similarity
# C = cosine_similarity(X)
from numpy.linalg import inv

S_norm = np.diag(np.sqrt(np.diag(G)))
# scaling matrix, diagnal element is the norm of x_j

C = inv(S_norm)@G@inv(S_norm)

#%% centroid of data matrix, E(X)

E_X = X_df.mean().to_frame().T

#%% Demean, centralize, X_c

X_c = X_df.sub(X_df.mean())

#%% covariance matrix, Sigma

SIGMA = X_df.cov()

#%% correlation matrix, P

RHO = X_df.corr()

#%% Normalize data, Z_X

from scipy.stats import zscore

Z_X = zscore(X_df)

#%%

# Bk4_Ch24_01_B

#%% QR decomposition

from numpy.linalg import qr

Q, R = qr(X_df,mode = 'reduced')

#%%

# Bk4_Ch24_01_C

#%% Cholesky decomposition

from numpy.linalg import cholesky as chol

L_G = chol(G)
R_G = L_G.T

#%% Cholesky decompose covariance matrix, SIGMA

L_Sigma = chol(SIGMA)

R_Sigma = L_Sigma.T

#%%

# Bk4_Ch24_01_D

#%% eigen decompose G

from numpy.linalg import eig

Lambs_G,V_G = eig(G)
Lambs_G = np.diag(Lambs_G)

#%% eigen decompose Sigma, covariance matrix

Lambs_sigma,V_sigma = eig(SIGMA)
Lambs_sigma = np.diag(Lambs_sigma)

#%% eigen decompose P, correlation matrix

Lambs_P,V_P = eig(RHO)
Lambs_P = np.diag(Lambs_P)

#%%

# Bk4_Ch24_01_E

#%% SVD, original data X

from numpy.linalg import svd

U_X,S_X_,V_X = svd(X_df, full_matrices=False)
V_X = V_X.T

# full_matrices=True
# indices_diagonal = np.diag_indices(4)
# S_X = np.zeros_like(X_df)
# S_X[indices_diagonal] = S_X_

# full_matrices=False
S_X = np.diag(S_X_)

#%% SVD, original data Xc

U_Xc,S_Xc,V_Xc = svd(X_c, full_matrices=False)
V_Xc = V_Xc.T
S_Xc = np.diag(S_Xc)

#%% SVD, z scores

U_Z,S_Z,V_Z = svd(Z_X, full_matrices=False)
V_Z = V_Z.T
S_Z = np.diag(S_Z)

