###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import streamlit as st
import plotly.express as px

import numpy as np
import pandas as pd  
from sklearn.datasets import load_iris

#%%

# A copy from Seaborn
iris = load_iris()
X = iris.data
y = iris.target

feature_names = ['Sepal length, x1','Sepal width, x2',
                 'Petal length, x3','Petal width, x4']

# Convert X array to dataframe
X_df = pd.DataFrame(X, columns=feature_names)

with st.sidebar:
    st.latex('X = USV^T')
    st.latex('X = \sum_{j=1}^{D} s_j u_j v_j^T')
    st.latex('X \simeq \sum_{j=1}^{p} s_j u_j v_j^T')

#%% Original data, X

X = X_df.to_numpy();

U, S, V_T = np.linalg.svd(X, full_matrices=False)
S = np.diag(S)
V = V_T.T

with st.sidebar:
    p = st.slider('Choose p, number of component to approximate X:',
                  1,4,step = 1)

#%% Approximate X

X_apprx = U[:,0:p]@S[0:p,0:p]@V[:,0:p].T
X_apprx_df = pd.DataFrame(X_apprx, columns = feature_names)

Error_df = X_df - X_apprx_df
#%%
col1, col2, col3 = st.columns(3)
with col1:
    st.latex('X')
    fig_1 = px.imshow(X_df,
                      color_continuous_scale='RdYlBu_r',
                      range_color = [0,8])

    fig_1.layout.height = 500
    fig_1.layout.width = 300
    fig_1.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_1)

with col2:
    
    st.latex('\hat{X}')
    fig_2 = px.imshow(X_apprx_df,
                      color_continuous_scale='RdYlBu_r',
                      range_color = [0,8])
    
    fig_2.layout.height = 500
    fig_2.layout.width = 300
    fig_2.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_2)
    
with col3:
    
    st.latex('X - \hat{X}')
    fig_3 = px.imshow(Error_df,
                      color_continuous_scale='RdYlBu_r',
                      range_color = [0,8])
    
    fig_3.layout.height = 500
    fig_3.layout.width = 300
    fig_3.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_3)

