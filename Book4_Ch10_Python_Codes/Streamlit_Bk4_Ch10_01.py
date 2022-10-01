# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 19:46:17 2022

@author: Work
"""


###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############


import streamlit as st
import plotly.express as px

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd  
from sklearn.datasets import load_iris


def bmatrix(a):
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv +=  [r'\end{bmatrix}']
    return '\n'.join(rv)


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
np.set_printoptions(suppress=True)
D = np.diag(D)
st.latex(r'G = X^T X = ' + bmatrix(G))
st.latex(r'G = V \Lambda V^T')

st.latex(r'G = ' + 
         bmatrix(np.round(V,2)) + '@' +
         bmatrix(np.round(D,2)) + '@' +
         bmatrix(np.round(V.T,2)))

st.write('Mapped data:')
st.latex('Z = XV')

#%%

Z = X@V

df = pd.DataFrame(Z, columns = ['PC1','PC2','PC3','PC4'])

mapping_rule = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
df.insert(4, "species", y)
df['species'] = df['species'].map(mapping_rule)

#%%
features = df.columns.to_list()[:-1]

#%% Table of mapped data

with st.expander('Mapped data'):
    st.write(df)
    
#%% Heatmap
with st.expander('Heatmap'):
    fig_1 = px.imshow(df.iloc[:,0:4],
                      color_continuous_scale='RdYlBu_r')
    st.plotly_chart(fig_1)
    
#%% 2D scatter plot

with st.sidebar:
    st.write('2D scatter plot')
    x_feature = st.radio('Horizontal axis',
             features)
    y_feature = st.radio('Vertical axis',
             features)  
    
with st.expander('2D scatter plot'):
    fig_2 = px.scatter(df, x=x_feature, y=y_feature, color="species")
    st.plotly_chart(fig_2)

#%% 3D scatter plot
with st.expander('3D scatter plot'):
    fig_3 = px.scatter_3d(df, 
                          x='PC1', 
                          y='PC2', 
                          z='PC3',
                  color='species')
    st.plotly_chart(fig_3)

#%% Pairwise scatter plot
with st.expander('Pairwise scatter plot'):
    fig_4 = px.scatter_matrix(df, 
                            dimensions=["PC1", 
                                        "PC2", 
                                        "PC3", 
                                        "PC4"], 
                            color="species")
    st.plotly_chart(fig_4)
    
    
    
    
