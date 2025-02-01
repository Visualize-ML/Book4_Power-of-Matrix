
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

# 定义一个函数，用于将矩阵转换为LaTeX格式的bmatrix
def bmatrix(a):
    """返回LaTeX bmatrix

    :a: numpy数组
    :returns: 作为字符串的LaTeX bmatrix
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix最多支持显示二维矩阵')  # 确保输入矩阵是二维的
    lines = str(a).replace('[', '').replace(']', '').splitlines()  # 格式化矩阵为字符串
    rv = [r'\begin{bmatrix}']  # LaTeX矩阵的开始部分
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]  # 添加每一行数据
    rv += [r'\end{bmatrix}']  # LaTeX矩阵的结束部分
    return '\n'.join(rv)

# 设置网格大小
n = m = 20

# 创建带两个子图的图表
fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.035)

# 初始化垂直线的x和y坐标列表
xv = []
yv = []

# 添加垂直线的坐标
for k in range(-n, n + 1):
    xv.extend([k, k, np.nan])  # 添加x坐标并在每条线后加入NaN以断开
    yv.extend([-m, m, np.nan])  # 添加y坐标并在每条线后加入NaN以断开
# 设置线宽
lw = 1  # line_width
# 将垂直线添加到第一个子图
fig.add_trace(go.Scatter(x=xv, y=yv, mode="lines", line_width=lw,
                         line_color='red'), 1, 1)

# 初始化水平线的x和y坐标列表
xh = []
yh = []
# 添加水平线的坐标
for k in range(-m, m + 1):
    xh.extend([-m, m, np.nan])  # 添加x坐标并在每条线后加入NaN以断开
    yh.extend([k, k, np.nan])  # 添加y坐标并在每条线后加入NaN以断开
# 将水平线添加到第一个子图
fig.add_trace(go.Scatter(x=xh, y=yh, mode="lines", line_width=lw,
                         line_color='blue'), 1, 1)

# 在Streamlit侧边栏中定义滑块和显示LaTeX公式
with st.sidebar:
    # 显示旋转矩阵的公式
    st.latex(r'''
             R = \begin{bmatrix}
    \cos(\theta) & -\sin(\theta)\\
    \sin(\theta) & \cos(\theta)
    \end{bmatrix}''')
    
    # 添加滑块，用于选择旋转角度theta
    theta = st.slider('Theta degree: ', -180, 180, step=5, value=0)  # 滑块范围为-180到180度
    theta = theta / 180 * np.pi  # 将角度转换为弧度

# 定义旋转矩阵R
R = np.array([[np.cos(theta), -np.sin(theta)], 
              [np.sin(theta), np.cos(theta)]], dtype=float)

# 将垂直线的坐标转换为NumPy数组
X = np.array(xv)
Y = np.array(yv)

# 使用旋转矩阵R变换垂直线的坐标
Txvyv = R @ np.stack((X, Y))  # 对每个坐标点应用旋转变换

# 将水平线的坐标转换为NumPy数组
X = np.array(xh)
Y = np.array(yh)

# 使用旋转矩阵R变换水平线的坐标
Txhyh = R @ np.stack((X, Y))  # 对每个坐标点应用旋转变换

# 显示旋转矩阵的LaTeX表达式
st.latex(r'R = ' + bmatrix(R))

# 提取旋转矩阵的列向量
r1 = R[:, 0].reshape((-1, 1))  # 第一列向量
r2 = R[:, 1].reshape((-1, 1))  # 第二列向量

# 显示列向量的LaTeX公式
st.latex(r'''
         r_1 = R e_1 = ''' + bmatrix(R) + 
         'e_1 = ' + bmatrix(r1)
         )
st.latex(r'''
         r_2 = R e_2 = ''' + bmatrix(R) + 
         'e_2 = ' + bmatrix(r2)
         )

# 显示旋转矩阵的行列式值
st.latex(r'\begin{vmatrix} R \end{vmatrix} = ' + str(np.linalg.det(R)))

# 将变换后的垂直线添加到第二个子图
fig.add_trace(go.Scatter(x=Txvyv[0], y=Txvyv[1], 
                         mode="lines", line_width=lw,
                         line_color='red'), 1, 2)

# 将变换后的水平线添加到第二个子图
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
