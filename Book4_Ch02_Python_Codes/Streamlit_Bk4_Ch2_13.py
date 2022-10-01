
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import streamlit as st
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

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

#%%
with st.sidebar:
    
    num_a = st.slider('Number of rows, a:',
              3,6,step = 1)
    num_b = st.slider('Number of rows, b:',
              3,6,step = 1)
    
a = np.random.uniform(0,1,num_a).reshape((-1,1))
a = np.round(a,1)
b = np.random.uniform(0,1,num_b).reshape((-1,1))
b = np.round(b,1)

show_number = False
with st.sidebar:
    show_number = st.checkbox('Display values')

tensor_a_b = a@b.T

#%% visualization

st.latex('a = ' + bmatrix(a))
st.latex('b = ' + bmatrix(b))
st.latex('a \otimes b = ab^{T}')
st.latex( bmatrix(a) + '@' + bmatrix(b.T) + ' = ' + bmatrix(tensor_a_b))
col1, col2, col3 = st.columns(3)

with col1:
    fig_a = px.imshow(a, text_auto=show_number, 
                  color_continuous_scale='viridis',
                  aspect = 'equal')
    
    fig_a.update_layout(height=400, width=300)
    fig_a.layout.coloraxis.showscale = False
    st.plotly_chart(fig_a)

with col2:
    fig_b = px.imshow(b, text_auto=show_number, 
                  color_continuous_scale='viridis',
                  aspect = 'equal')
    
    fig_b.update_layout(height=400, width=300)
    fig_b.layout.coloraxis.showscale = False
    st.plotly_chart(fig_b)

with col3:
    fig_ab = px.imshow(tensor_a_b, text_auto=show_number, 
                  color_continuous_scale='viridis',
                  aspect = 'equal')
    
    fig_ab.update_layout(height=400, width=400)
    
    fig_ab.layout.coloraxis.showscale = False
    st.plotly_chart(fig_ab)






