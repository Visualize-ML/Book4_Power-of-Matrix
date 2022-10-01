
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############


import streamlit as st
import numpy as np
import plotly.express as px
import pandas as pd


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

with st.sidebar:
    
    st.latex(r'''
             A = \begin{bmatrix}
    a & b\\
    c & d
    \end{bmatrix}''')
    
    a = st.slider('a',-2.0, 2.0, step = 0.1, value = 1.0)
    b = st.slider('b',-2.0, 2.0, step = 0.1, value = 0.0)  
    c = st.slider('c',-2.0, 2.0, step = 0.1, value = 0.0)  
    d = st.slider('d',-2.0, 2.0, step = 0.1, value = 1.0) 
    
#%%

x1_ = np.linspace(-1, 1, 11)
x2_ = np.linspace(-1, 1, 11)

xx1,xx2 = np.meshgrid(x1_, x2_)

X = np.column_stack((xx1.flatten(), xx2.flatten()))

# st.write(X)
A = np.array([[a, b],
              [c, d]])

X = X@A

# st.write(len(X))
#%% 
color_array = np.linspace(0,1,len(X))
# st.write(color_array)
X = np.column_stack((X, color_array))
df = pd.DataFrame(X, columns=['z1','z2', 'color'])

#%% Scatter

st.latex('A = ' + bmatrix(A))

fig = px.scatter(df, 
                 x="z1", 
                 y="z2", 
                 color='color',
                 color_continuous_scale = 'rainbow')

fig.update_layout(
    autosize=False,
    width=500,
    height=500)

fig.add_hline(y=0, line_color = 'black')
fig.add_vline(x=0, line_color = 'black')

fig.update_xaxes(range=[-3, 3])
fig.update_yaxes(range=[-3, 3])
fig.update_coloraxes(showscale=False)

st.plotly_chart(fig)



