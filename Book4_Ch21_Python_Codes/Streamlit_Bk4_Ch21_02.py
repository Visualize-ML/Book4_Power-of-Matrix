
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############


import streamlit as st
import plotly.graph_objects as go
import sympy
import numpy as np

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
    
    st.latex(r'''
             f(x_1,x_2) = ax_1^2 + 2bx_1x_2 + cx_2^2
             ''')
    
    
    a = st.slider('a',-2.0, 2.0, step = 0.1)
    b = st.slider('b',-2.0, 2.0, step = 0.1)  
    c = st.slider('c',-2.0, 2.0, step = 0.1)  
    
#%%

x1_ = np.linspace(-2, 2, 101)
x2_ = np.linspace(-2, 2, 101)

xx1,xx2 = np.meshgrid(x1_, x2_)

#define symbolic vars, function
x1,x2 = sympy.symbols('x1 x2')

A = np.array([[a, b],
              [b, c]])

D,V = np.linalg.eig(A)
D = np.diag(D)

st.latex(r'''A = \begin{bmatrix}%s & %s\\%s & %s\end{bmatrix}''' %(a, b, b, c))
st.latex(r'''A = V \Lambda V^{T}''')
st.latex(bmatrix(A) + '=' + 
         bmatrix(np.around(V, decimals=3)) + '@' + 
         bmatrix(np.around(D, decimals=3)) + '@' + 
         bmatrix(np.around(V.T, decimals=3)))

x = np.array([[x1,x2]]).T

f_x = a*x1**2 + 2*b*x1*x2 + c*x2**2

st.latex(r'''f(x_1,x_2) = ''')
st.write(f_x)

f_x_fcn = sympy.lambdify([x1,x2],f_x)

ff_x = f_x_fcn(xx1,xx2)

#%% Plot 3D surface

fig_surface = go.Figure(go.Surface(
    x = x1_,
    y = x2_,
    z = ff_x,
    colorscale= 'RdYlBu_r'))

fig_surface.update_layout(
    autosize=False,
    width=500,
    height=500)
st.plotly_chart(fig_surface)

#%% Plot 2D contour

fig_contour = go.Figure(
    go.Contour(
        z=ff_x,
        x=x1_,
        y=x2_,
        colorscale= 'RdYlBu_r'
    ))

fig_contour.update_layout(
    autosize=False,
    width=500,
    height=500)

st.plotly_chart(fig_contour)

