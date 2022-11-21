
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
             A = \begin{bmatrix}
    a & b\\
    c & d
    \end{bmatrix}''')
    
    a = st.slider('a',-2.0, 2.0, step = 0.1, value = 1.0)
    b = st.slider('b',-2.0, 2.0, step = 0.1, value = 0.0)  
    c = st.slider('c',-2.0, 2.0, step = 0.1, value = 0.0)  
    d = st.slider('d',-2.0, 2.0, step = 0.1, value = 1.0) 

theta = np.pi/6
A = np.array([[a, b], 
              [c, d]], dtype=float)

#get only the coordinates from -3 to 3
# X = np.array(xv[6:-6])
# Y = np.array(yv[6:-6])

X = np.array(xv)
Y = np.array(yv)

# transform by T the vector of coordinates [x, y]^T where the vector runs over the columns of np.stack((X, Y))
Txvyv = A@np.stack((X, Y)) #transform by T the vertical lines

# X = np.array(xh[6:-6])
# Y = np.array(yh[6:-6])

X = np.array(xh)
Y = np.array(yh)

Txhyh = A@np.stack((X, Y))# #transform by T the horizontal lines

st.latex(r'A = ' + bmatrix(A))

a1 = A[:,0].reshape((-1, 1))
a2 = A[:,1].reshape((-1, 1))

st.latex(r'''
         a_1 = Ae_1 = ''' + bmatrix(A) + 
         'e_1 = ' + bmatrix(a1)
         )

st.latex(r'''
         a_2 = Ae_2 = ''' + bmatrix(A) + 
         'e_2 = ' + bmatrix(a2)
         )

st.latex(r'\begin{vmatrix} A \end{vmatrix} = ' + str(np.linalg.det(A)))

theta_array = np.linspace(0, 2*np.pi, 101)
circle_x = np.cos(theta_array)
circle_y = np.sin(theta_array)
circle_array = np.stack((circle_x, circle_y))

fig.add_trace(go.Scatter(x=circle_x, y=circle_y, 
                         fill="toself", line_color='orange'), 1, 1)

A_times_circle_array = A@circle_array

fig.add_trace(go.Scatter(x=A_times_circle_array[0,:], 
                         y=A_times_circle_array[1,:], 
                         fill="toself", line_color='orange'), 1, 2)

fig.add_trace(go.Scatter(x=Txvyv[0], y=Txvyv[1], 
                         mode="lines", line_width=lw,
                         line_color = 'blue'), 1, 2)

fig.add_trace(go.Scatter(x=Txhyh[0], y=Txhyh[1], 
                         mode="lines", line_width=lw,
                         line_color = 'red'), 1, 2)
fig.update_xaxes(range=[-4, 4])
fig.update_yaxes(range=[-4, 4])
fig.update_layout(width=800, height=500, showlegend=False, template="none",
                   plot_bgcolor="white", yaxis2_showgrid=False, xaxis2_showgrid=False)

st.plotly_chart(fig)