# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 21:19:47 2022

@author: Work
"""

import pandas as pd
import plotly.graph_objs as go
import streamlit as st
import numpy as np

with st.sidebar:
    num = st.slider('Number of points for each dimension',
                    max_value = 20,
                    min_value = 10,
                    step = 1)
    
x1 = np.linspace(0,1,num)
x2 = x1
x3 = x1

xx1,xx2,xx3 = np.meshgrid(x1,x2,x3)

x1_ = xx1.ravel()
x2_ = xx2.ravel()
x3_ = xx3.ravel()

#%%
df = pd.DataFrame({'X': x1_,
                   'Y': x2_,
                   'Z': x3_,
                   'R': (x1_*256).round(),
                   'G': (x2_*256).round(),
                   'B': (x3_*256).round()})

trace = go.Scatter3d(x=df.X,
                      y=df.Y,
                      z=df.Z,
                      mode='markers',
                      marker=dict(size=3,
                                  color=['rgb({},{},{})'.format(r,g,b) 
                                         for r,g,b in 
                                         zip(df.R.values, df.G.values, df.B.values)],
                                  opacity=0.9,))

data = [trace]

layout = go.Layout(margin=dict(l=0,
                               r=0,
                               b=0,
                               t=0),
                   scene = dict(
    xaxis = dict(title='e_1'),
    yaxis = dict(title='e_2'),
    zaxis = dict(title='e_3'),),
)

fig = go.Figure(data=data, layout=layout)

st.plotly_chart(fig)
