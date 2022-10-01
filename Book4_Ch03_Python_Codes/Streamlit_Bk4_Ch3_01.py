
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import streamlit as st
import plotly.graph_objects as go
import numpy as np

x1 = np.linspace(-2.5, 2.5, num=101);
x2 = x1;
xx1, xx2 = np.meshgrid(x1,x2)

with st.sidebar:
    
    st.write('Note: Lp norm, p >= 1')
    p = st.slider('p', 
                  min_value = -20.0, 
                  max_value = 20.,
                  step = 0.2)
    
zz = ((np.abs((xx1))**p) + (np.abs((xx2))**p))**(1./p)

fig_1 = go.Figure(data = 
                go.Contour(z = zz,
                x = x1,
                y = x2,
                colorscale='RdYlBu_r'))

fig_1.update_layout(
    autosize=False,
    width=500,
    height=500,
    margin=dict(
        l=50,
        r=50,
        b=50,
        t=50))

fig_2 = go.Figure(
    go.Surface(
    x = x1,
    y = x2,
    z = zz,
    colorscale='RdYlBu_r'))


with st.expander('2D contour'):
    st.plotly_chart(fig_1)

with st.expander('3D surface'):
    st.plotly_chart(fig_2)