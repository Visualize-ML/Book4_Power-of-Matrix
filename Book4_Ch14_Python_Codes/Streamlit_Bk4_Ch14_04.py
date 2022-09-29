
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
    b & c
    \end{bmatrix}''')
    
    a = st.slider('a',-2.0, 2.0, step = 0.05, value = 1.0)
    b = st.slider('b',-2.0, 2.0, step = 0.05, value = 0.0)  
    c = st.slider('c',-2.0, 2.0, step = 0.05, value = 1.0)
    
#%%

theta_array = np.linspace(0, 2*np.pi, 36)

X = np.column_stack((np.cos(theta_array), 
                     np.sin(theta_array)))

# st.write(X)
A = np.array([[a, b],
              [b, c]])

st.latex(r'''z^Tz = 1''')
st.latex(r'''x = Az''')

st.latex('A =' + bmatrix(A))

X_ = X@A

#define symbolic vars, function
x1,x2 = sympy.symbols('x1 x2')
y1,y2 = sympy.symbols('y1 y2')
x = np.array([[x1,x2]]).T
y = np.array([[y1,y2]]).T

Q = np.linalg.inv(A@A.T)
D,V = np.linalg.eig(Q)
D = np.diag(D)

st.latex(r'Q = \left( AA^T\right)^{-1} = ' + bmatrix(np.round(Q, 3)))

st.latex(r'''Q = V \Lambda V^{T}''')
st.latex(bmatrix(np.around(Q, decimals=3)) + '=' + 
         bmatrix(np.around(V, decimals=3)) + '@' + 
         bmatrix(np.around(D, decimals=3)) + '@' + 
         bmatrix(np.around(V.T, decimals=3)))

f_x = x.T@np.round(Q, 3)@x
f_y = y.T@np.round(D, 3)@y

from sympy import *
st.write('The formula of the ellipse:')
st.latex(latex(simplify(f_x[0][0])) + ' = 1')

st.write('The formula of the transformed ellipse:')
st.latex(latex(simplify(f_y[0][0])) + ' = 1')

#%% 
color_array = np.linspace(0,1,len(X))
# st.write(color_array)
X_c = np.column_stack((X_, color_array))
df = pd.DataFrame(X_c, columns=['x1','x2', 'color'])

#%% Scatter

fig = px.scatter(df, 
                 x="x1", 
                 y="x2", 
                 color='color',
                 color_continuous_scale=px.colors.sequential.Rainbow)



fig.update_layout(
    autosize=False,
    width=500,
    height=500)


fig.add_hline(y=0, line_color = 'black')
fig.add_vline(x=0, line_color = 'black')
fig.update_layout(coloraxis_showscale=False)
fig.update_xaxes(range=[-3, 3])
fig.update_yaxes(range=[-3, 3])

st.plotly_chart(fig)



