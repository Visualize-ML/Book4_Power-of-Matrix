
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

# 定义一个函数，用于生成LaTeX格式的矩阵表示
def bmatrix(a):
    """返回LaTeX bmatrix
    :a: numpy数组
    :returns: LaTeX bmatrix格式的字符串
    """
    if len(a.shape) > 2:  # 检查矩阵维度是否超过2
        raise ValueError('bmatrix最多支持二维矩阵')
    lines = str(a).replace('[', '').replace(']', '').splitlines()  # 格式化矩阵为字符串
    rv = [r'\begin{bmatrix}']  # 开始LaTeX矩阵表示
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]  # 添加每一行
    rv += [r'\end{bmatrix}']  # 结束LaTeX矩阵表示
    return '\n'.join(rv)

# 设置网格大小
n = m = 20

# 创建具有两个子图的图表
fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.035)

# 初始化垂直线的x和y坐标列表
xv = []
yv = []

# 添加垂直线的坐标
for k in range(-n, n + 1):
    xv.extend([k, k, np.nan])  # 添加垂直线的x坐标并在每条线后加入NaN以断开
    yv.extend([-m, m, np.nan])  # 添加垂直线的y坐标并在每条线后加入NaN以断开

# 设置线宽
lw = 1  # 线宽
# 将垂直线添加到图表的第一个子图
fig.add_trace(go.Scatter(x=xv, y=yv, mode="lines", line_width=lw,
                         line_color='red'), 1, 1)

# 初始化水平线的x和y坐标列表
xh = []
yh = []

# 添加水平线的坐标
for k in range(-m, m + 1):
    xh.extend([-m, m, np.nan])  # 添加水平线的x坐标并在每条线后加入NaN以断开
    yh.extend([k, k, np.nan])  # 添加水平线的y坐标并在每条线后加入NaN以断开

# 将水平线添加到图表的第一个子图
fig.add_trace(go.Scatter(x=xh, y=yh, mode="lines", line_width=lw,
                         line_color='blue'), 1, 1)

# 使用Streamlit的侧边栏显示滑块和LaTeX矩阵公式
with st.sidebar:
    st.latex(r'''
             A = \begin{bmatrix}
    a & b\\
    c & d
    \end{bmatrix}''')  # 显示矩阵A的LaTeX公式

    # 添加滑块控件，供用户调整矩阵A的元素值
    a = st.slider('a', -2.0, 2.0, step=0.1, value=1.0)
    b = st.slider('b', -2.0, 2.0, step=0.1, value=0.0)
    c = st.slider('c', -2.0, 2.0, step=0.1, value=0.0)
    d = st.slider('d', -2.0, 2.0, step=0.1, value=1.0)

# 定义一个固定的角度值
theta = np.pi / 6

# 创建矩阵A
A = np.array([[a, b],
              [c, d]], dtype=float)

# 获取垂直线的坐标并转换为NumPy数组
X = np.array(xv)
Y = np.array(yv)

# 使用矩阵A变换垂直线的坐标
Txvyv = A @ np.stack((X, Y))  # 对坐标点应用线性变换

# 获取水平线的坐标并转换为NumPy数组
X = np.array(xh)
Y = np.array(yh)

# 使用矩阵A变换水平线的坐标
Txhyh = A @ np.stack((X, Y))  # 对坐标点应用线性变换

# 显示矩阵A的LaTeX格式
st.latex(r'A = ' + bmatrix(A))

# 提取矩阵A的列向量
a1 = A[:, 0].reshape((-1, 1))  # 第一列
a2 = A[:, 1].reshape((-1, 1))  # 第二列

# 显示矩阵列向量的LaTeX公式
st.latex(r'''
         a_1 = Ae_1 = ''' + bmatrix(A) +
         'e_1 = ' + bmatrix(a1)
         )
st.latex(r'''
         a_2 = Ae_2 = ''' + bmatrix(A) +
         'e_2 = ' + bmatrix(a2)
         )

# 显示矩阵A的行列式值
st.latex(r'\begin{vmatrix} A \end{vmatrix} = ' + str(np.linalg.det(A)))

# 生成一个单位圆的坐标
theta_array = np.linspace(0, 2 * np.pi, 101)  # 定义角度数组
circle_x = np.cos(theta_array)  # 单位圆x坐标
circle_y = np.sin(theta_array)  # 单位圆y坐标
circle_array = np.stack((circle_x, circle_y))  # 将x和y坐标堆叠为二维数组

# 将单位圆添加到第一个子图
fig.add_trace(go.Scatter(x=circle_x, y=circle_y,
                         fill="toself", line_color='orange'), 1, 1)

# 使用矩阵A变换单位圆的坐标
A_times_circle_array = A @ circle_array

# 将变换后的单位圆添加到第二个子图
fig.add_trace(go.Scatter(x=A_times_circle_array[0, :],
                         y=A_times_circle_array[1, :],
                         fill="toself", line_color='orange'), 1, 2)

# 将变换后的垂直线添加到第二个子图
fig.add_trace(go.Scatter(x=Txvyv[0], y=Txvyv[1],
                         mode="lines", line_width=lw,
                         line_color='blue'), 1, 2)

# 将变换后的水平线添加到第二个子图
fig.add_trace(go.Scatter(x=Txhyh[0], y=Txhyh[1],
                         mode="lines", line_width=lw,
                         line_color='red'), 1, 2)

# 设置x轴和y轴的范围
fig.update_xaxes(range=[-4, 4])
fig.update_yaxes(range=[-4, 4])

# 设置图表的布局和样式
fig.update_layout(width=800, height=500, showlegend=False, template="none",
                  plot_bgcolor="white", yaxis2_showgrid=False, xaxis2_showgrid=False)

# 在Streamlit应用中显示图表
st.plotly_chart(fig)
