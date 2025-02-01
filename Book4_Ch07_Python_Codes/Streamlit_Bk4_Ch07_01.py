
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2025
###############

import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import streamlit as st

# 定义一个函数，用于返回LaTeX格式的bmatrix
def bmatrix(a):
    """返回LaTeX bmatrix
    :a: numpy数组
    :returns: 作为字符串的LaTeX bmatrix
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix最多只能显示二维数组')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv += [r'\end{bmatrix}']
    return '\n'.join(rv)

# 设置网格的范围
n = m = 20

# 创建一个具有两个子图的图表
fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.035)

# 初始化垂直线的x和y坐标列表
xv = []
yv = []

# 添加垂直线的坐标
for k in range(-n, n + 1):
    xv.extend([k, k, np.nan])
    yv.extend([-m, m, np.nan])

# 设置线宽
lw = 1
# 添加垂直线到图表
fig.add_trace(go.Scatter(x=xv, y=yv, mode="lines", line_width=lw,
                         line_color='red'), 1, 1)

# 初始化水平线的x和y坐标列表
xh = []
yh = []
# 添加水平线的坐标
for k in range(-m, m + 1):
    xh.extend([-m, m, np.nan])
    yh.extend([k, k, np.nan])
    fig.add_trace(go.Scatter(x=xh, y=yh, mode="lines", line_width=lw,
                             line_color='blue'), 1, 1)

# 在侧边栏中添加滑块控件
with st.sidebar:
    # 显示LaTeX矩阵
    st.latex(r'''
             A = \begin{bmatrix}
    a & b\\
    c & d
    \end{bmatrix}''')
    
    # 添加矩阵A的参数滑块
    a = st.slider('a', -2.0, 2.0, step=0.1, value=1.0)
    b = st.slider('b', -2.0, 2.0, step=0.1, value=0.0)
    c = st.slider('c', -2.0, 2.0, step=0.1, value=0.0)
    d = st.slider('d', -2.0, 2.0, step=0.1, value=1.0)

# 定义旋转角度
theta = np.pi / 6
# 定义矩阵A
A = np.array([[a, b],
              [c, d]], dtype=float)

# 将垂直线的坐标转换为NumPy数组
X = np.array(xv)
Y = np.array(yv)

# 通过矩阵A变换垂直线的坐标
Txvyv = A @ np.stack((X, Y))

# 将水平线的坐标转换为NumPy数组
X = np.array(xh)
Y = np.array(yh)

# 通过矩阵A变换水平线的坐标
Txhyh = A @ np.stack((X, Y))

# 显示矩阵A的LaTeX格式
st.latex(bmatrix(A))

# 提取矩阵A的列向量
a1 = A[:, 0].reshape((-1, 1))
a2 = A[:, 1].reshape((-1, 1))

# 显示列向量的LaTeX表达式
st.latex(r'''
         a_1 = Ae_1 = ''' + bmatrix(A) +
         'e_1 = ' + bmatrix(a1)
         )

st.latex(r'''
         a_2 = Ae_2 = ''' + bmatrix(A) +
         'e_2 = ' + bmatrix(a2)
         )

# 添加变换后的垂直线到图表
fig.add_trace(go.Scatter(x=Txvyv[0], y=Txvyv[1],
                         mode="lines", line_width=lw,
                         line_color='red'), 1, 2)

# 添加变换后的水平线到图表
fig.add_trace(go.Scatter(x=Txhyh[0], y=Txhyh[1],
                         mode="lines", line_width=lw,
                         line_color='blue'), 1, 2)

# 设置x轴和y轴的范围
fig.update_xaxes(range=[-4, 4])
fig.update_yaxes(range=[-4, 4])

# 设置图表的布局和样式
fig.update_layout(width=800, height=500, showlegend=False, template="none",
                  plot_bgcolor="white", yaxis2_showgrid=False, xaxis2_showgrid=False)

# 在Streamlit应用中显示图表
st.plotly_chart(fig)
