
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
from scipy.stats import multivariate_normal

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
             \Sigma = \begin{bmatrix}
    \sigma_1^2 & 
    \rho \sigma_1 \sigma_2 \\
    \rho \sigma_1 \sigma_2 & 
    \sigma_2^2
    \end{bmatrix}''')
    
    
    st.write('$\sigma_1$')
    sigma_1 = st.slider('sigma_1',1.0, 2.0, step = 0.1)

    st.write('$\sigma_2$')
    sigma_2 = st.slider('sigma_2',1.0, 2.0, step = 0.1)
    
    st.write('$\u03C1$')
    rho_12 = st.slider('rho',-0.9, 0.9, step = 0.1)
        
#%%

st.latex(r'''
   f(x) = \frac{1}{\sqrt{2\pi} \sigma} 
          \exp\left( -\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^{\!2}\,\right)
          ''')
  
st.latex(r'''
   f(x) = \frac{1}{\left( 2 \pi \right)^{\frac{D}{2}} 
          \begin{vmatrix}
          \Sigma 
          \end{vmatrix}^{\frac{1}{2}}} 
          \exp\left( 
         -\frac{1}{2}
          \left( x - \mu \right)^{T} \Sigma^{-1} \left( x - \mu \right)
          \right)
          ''')
          
#%%
x1 = np.linspace(-3,3,101)
x2 = np.linspace(-3,3,101)

xx1, xx2 = np.meshgrid(x1,x2)
pos = np.dstack((xx1, xx2))

Sigma = [[sigma_1**2, rho_12*sigma_1*sigma_2], 
         [rho_12*sigma_1*sigma_2, sigma_2**2]]
rv = multivariate_normal([0, 0], 
                         Sigma)
PDF_zz = rv.pdf(pos)

#%%

Sigma = np.array(Sigma)

D,V = np.linalg.eig(Sigma)
D = np.diag(D)

st.latex(r'''\Sigma = \begin{bmatrix}%s & %s\\%s & %s\end{bmatrix}''' 
         %(sigma_1**2, 
           rho_12*sigma_1*sigma_2, 
           rho_12*sigma_1*sigma_2, 
           sigma_2**2))
st.latex(r'''\Sigma = V \Lambda V^{T}''')
st.latex(bmatrix(Sigma) + '=' + 
         bmatrix(np.around(V, decimals=3)) + '@' + 
         bmatrix(np.around(D, decimals=3)) + '@' + 
         bmatrix(np.around(V.T, decimals=3)))

#%% Plot 3D surface

fig_surface = go.Figure(go.Surface(
    x = x1,
    y = x2,
    z = PDF_zz,
    colorscale= 'RdYlBu_r'))
fig_surface.update_layout(
    autosize=False,
    width=500,
    height=500)
st.plotly_chart(fig_surface)

#%% Plot 2D contour

fig_contour = go.Figure(
    go.Contour(
        z=PDF_zz,
        x=x1,
        y=x2,
        colorscale= 'RdYlBu_r'
    ))

fig_contour.update_layout(
    autosize=False,
    width=500,
    height=500)

st.plotly_chart(fig_contour)       
        
        
        
    