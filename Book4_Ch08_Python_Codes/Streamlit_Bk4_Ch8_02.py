
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import streamlit as st


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

n = m = 20

fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.035)

xv = []
yv = [] 

for k in range(-n, n+1):
    xv.extend([k, k, np.nan])
    yv.extend([-m, m, np.nan])
lw= 1 #line_width
fig.add_trace(go.Scatter(x=xv, y=yv, mode="lines", line_width=lw,
                         line_color = 'red'), 1, 1)
#set up  the lists  of  horizontal line x and y-end coordinates

xh=[]
yh=[]
for k in range(-m, m+1):
    xh.extend([-m, m, np.nan])
    yh.extend([k, k, np.nan])
    
fig.add_trace(go.Scatter(x=xh, y=yh, mode="lines", line_width=lw,
                         line_color = 'blue'), 1, 1)


with st.sidebar:
    
    st.latex(r'''
             R = \begin{bmatrix}
    \cos(\theta) & -\sin(\theta)\\
    \sin(\theta) & \cos(\theta)
    \end{bmatrix}''')
    
    theta = st.slider('Theta degree: ',-180, 180, step = 5, value = 0)
    
    theta = theta/180*np.pi


R = np.array([[np.cos(theta), -np.sin(theta)], 
              [np.sin(theta), np.cos(theta)]], dtype=float)

#get only the coordinates from -3 to 3
# X = np.array(xv[6:-6])
# Y = np.array(yv[6:-6])

X = np.array(xv)
Y = np.array(yv)

# transform by T the vector of coordinates [x, y]^T where the vector runs over the columns of np.stack((X, Y))
Txvyv = R@np.stack((X, Y)) #transform by T the vertical lines

# X = np.array(xh[6:-6])
# Y = np.array(yh[6:-6])

X = np.array(xh)
Y = np.array(yh)

Txhyh = R@np.stack((X, Y))# #transform by T the horizontal lines

st.latex(r'R = ' + bmatrix(R))

r1 = R[:,0].reshape((-1, 1))
r2 = R[:,1].reshape((-1, 1))

st.latex(r'''
         r_1 = R e_1 = ''' + bmatrix(R) + 
         'e_1 = ' + bmatrix(r1)
         )

st.latex(r'''
         r_2 = R e_2 = ''' + bmatrix(R) + 
         'e_2 = ' + bmatrix(r2)
         )

st.latex(r'\begin{vmatrix} R \end{vmatrix} = ' + str(np.linalg.det(R)))

fig.add_trace(go.Scatter(x=Txvyv[0], y=Txvyv[1], 
                         mode="lines", line_width=lw,
                         line_color = 'red'), 1, 2)

fig.add_trace(go.Scatter(x=Txhyh[0], y=Txhyh[1], 
                         mode="lines", line_width=lw,
                         line_color = 'blue'), 1, 2)

fig.update_xaxes(range=[-4, 4])
fig.update_yaxes(range=[-4, 4])
fig.update_layout(width=800, height=500, showlegend=False, template="none",
                   plot_bgcolor="white", yaxis2_showgrid=False, xaxis2_showgrid=False)

st.plotly_chart(fig)