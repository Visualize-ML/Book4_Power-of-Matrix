
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import plotly.graph_objects as go
import streamlit as st
import numpy as np
import plotly.express as px
import pandas as pd
import sympy
from scipy.spatial import distance

#%% define a function for distances

def fcn_Minkowski(xx, yy, mu, p = 2, Chebychev = False):
    
    if Chebychev:
        
        zz = np.maximum(np.abs(xx - mu[0]),np.abs(yy - mu[1]))
        
    else:
        zz = ((np.abs((xx - mu[0]))**p) + (np.abs((yy - mu[1]))**p))**(1./p)
    
    return zz

def fcn_mahal(xx, yy, mu, Sigma, standardized = False):
    
    if standardized:
        
        D = np.diag(np.diag(Sigma))
        Sigma_inv = np.linalg.inv(D)
        
    else:
        Sigma_inv = np.linalg.inv(Sigma)
    
    xy_ = np.stack((xx.flatten(), yy.flatten())).T

    zz = np.diag(np.sqrt(np.dot(np.dot((xy_-mu),Sigma_inv),(xy_-mu).T)))
    
    zz = np.reshape(zz,xx.shape)
    
    return zz


#%%
df = px.data.iris()

with st.sidebar:
    
    dist_type = st.radio('Choose a type of distance: ',
             options = ['Euclidean',
                        'City block',
                        'Minkowski',
                        'Chebychev',
                        'Mahalanobis',
                        'Standardized Euclidean'])
    
    if dist_type == 'Minkowski':
        
        with st.sidebar:
            p = st.slider('Specify a p value:',1.0, 20.0, step = 0.5)
            
            
#%% compute distance

X = df[['sepal_length', 'petal_length']]

mu = X.mean().to_numpy()
Sigma = X.cov().to_numpy()

# st.write(mu)
# st.write(Sigma)

x_array = np.linspace(0,10,101)
y_array = np.linspace(0,10,101)

xx,yy = np.meshgrid(x_array,y_array)


if dist_type == 'Minkowski':
    
    zz = fcn_Minkowski(xx, yy, mu, p)

elif dist_type == 'Euclidean':
    
    zz = fcn_Minkowski(xx, yy, mu, 2)

elif dist_type == 'Chebychev':
    
    zz = fcn_Minkowski(xx, yy, mu, Chebychev = True)
    
elif dist_type == 'Mahalanobis':
    
    zz = fcn_mahal(xx, yy, mu, Sigma)
    
elif dist_type == 'City block':
    zz = fcn_Minkowski(xx, yy, mu, 1)
    
elif dist_type == 'Standardized Euclidean':
    
    zz = fcn_mahal(xx, yy, mu, Sigma, True)
    # st.write(zz)

#%% Visualization

st.title(dist_type + ' distance')

# Scatter plot
fig_2 = px.scatter(df, x='sepal_length', y='petal_length')

# plot distance contour
fig_2.add_trace(go.Contour(
    x = x_array,
    y = y_array,
    z = zz,
    contours_coloring='lines',
    showscale=False)
    )

# st.write(X.mean().to_frame().T)

# plot centroid
# fig_2.add_traces(
#     px.scatter(X.mean().to_frame().T, 
#                 x='sepal_length', 
#                 y='petal_length').update_traces(
#         marker_size=20, 
#         marker_color="yellow").data)

fig_2.add_traces(
    px.scatter(X.mean().to_frame().T, 
               x='sepal_length', 
               y='petal_length').update_traces(
        marker_size=20, 
        marker_color="red",
        marker_symbol= 'x').data)

fig_2.update_layout(yaxis_range=[0,10])
fig_2.update_layout(xaxis_range=[0,10])
fig_2.add_hline(y=mu[1])
fig_2.add_vline(x=mu[0])
fig_2.update_yaxes(
    scaleratio = 1,
  )


fig_2.update_layout(width=600, height=600)

st.plotly_chart(fig_2)


    