
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############


import sympy
import numpy as np
from sympy.functions import exp
import streamlit as st
import plotly.figure_factory as ff
import plotly.graph_objects as go

#define symbolic vars, function
x1,x2 = sympy.symbols('x1 x2')

f_x = x1*exp(-(x1**2 + x2**2))

st.latex('f(x_1, x_2) = ' + sympy.latex(f_x))

#take the gradient symbolically
grad_f = [sympy.diff(f_x,var) for var in (x1,x2)]
st.latex(r'\nabla f = ' + sympy.latex(grad_f) + '^T')

f_x_fcn = sympy.lambdify([x1,x2],f_x)

#turn into a bivariate lambda for numpy
grad_fcn = sympy.lambdify([x1,x2],grad_f)

x1_array = np.linspace(-2,2,100)
x2_array = np.linspace(-2,2,100)

xx1, xx2 = np.meshgrid(x1_array,x2_array)

# coarse mesh
xx1_, xx2_ = np.meshgrid(np.linspace(-2,2,20),np.linspace(-2,2,20))
V = grad_fcn(xx1_,xx2_)

ff_x = f_x_fcn(xx1,xx2)


#%% visualizations

fig_surface = go.Figure(go.Surface(
    x = x1_array,
    y = x2_array,
    z = ff_x,
    showscale=False))
fig_surface.update_layout(
    autosize=False,
    width =800,
    height=800)

st.plotly_chart(fig_surface)  


f = ff.create_quiver(xx1_, xx2_, V[0], V[1])
trace1 = f.data[0]
trace2 = go.Contour(
    x = x1_array,
    y = x2_array,
    z = ff_x,
    showscale=False)

data=[trace1,trace2]
fig = go.FigureWidget(data)
fig.update_layout(
    autosize=False,
    width =800,
    height=800)


fig.add_hline(y=0, line_color = 'black')
fig.add_vline(x=0, line_color = 'black')

fig.update_xaxes(range=[-2, 2])
fig.update_yaxes(range=[-2, 2])
fig.update_coloraxes(showscale=False)

st.plotly_chart(fig)  


