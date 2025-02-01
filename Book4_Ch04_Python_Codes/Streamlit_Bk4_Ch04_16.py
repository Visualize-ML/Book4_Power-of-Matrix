
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2025
###############

## 导入必要的库
import plotly.graph_objects as go  # 用于创建绘图对象
import numpy as np  # 用于数值计算
from plotly.subplots import make_subplots  # 用于创建子图
import streamlit as st  # 用于构建交互式Web应用

## 定义一个函数，将NumPy数组格式化为LaTeX矩阵
def bmatrix(a):
    """返回一个LaTeX bmatrix格式的字符串
    :param a: 输入的NumPy数组
    :return: 返回LaTeX矩阵的字符串表示
    """
    if len(a.shape) > 2:  # 检查数组是否为二维
        raise ValueError('bmatrix最多只能展示二维数组')  # 如果不是二维数组，抛出异常
    lines = str(a).replace('[', '').replace(']', '').splitlines()  # 去掉数组表示中的方括号并分行
    rv = [r'\begin{bmatrix}']  # 添加LaTeX矩阵的起始标签
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]  # 将每行元素用&连接并加上换行符
    rv += [r'\end{bmatrix}']  # 添加LaTeX矩阵的结束标签
    return '\n'.join(rv)  # 返回拼接后的字符串

## 初始化网格范围
n = m = 20  # 定义网格的横纵坐标范围

## 创建子图
fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.035)  # 创建一个包含两列的子图

## 绘制垂直线的坐标
xv = []  # 存储垂直线的x坐标
yv = []  # 存储垂直线的y坐标
for k in range(-n, n + 1):  # 遍历网格范围
    xv.extend([k, k, np.nan])  # 每条垂直线的起始和结束x坐标
    yv.extend([-m, m, np.nan])  # 每条垂直线的起始和结束y坐标

## 绘制垂直线
lw = 1  # 定义线条宽度
fig.add_trace(go.Scatter(x=xv, y=yv, mode="lines", line_width=lw, line_color='red'), 1, 1)  # 添加垂直线到第一个子图

## 绘制水平线的坐标
xh = []  # 存储水平线的x坐标
yh = []  # 存储水平线的y坐标
for k in range(-m, m + 1):  # 遍历网格范围
    xh.extend([-m, m, np.nan])  # 每条水平线的起始和结束x坐标
    yh.extend([k, k, np.nan])  # 每条水平线的起始和结束y坐标
    fig.add_trace(go.Scatter(x=xh, y=yh, mode="lines", line_width=lw, line_color='blue'), 1, 1)  # 添加水平线到第一个子图

## 创建交互式侧边栏
with st.sidebar:  # 定义侧边栏
    st.latex(r'''
             A = \begin{bmatrix}
    a & b\\
    c & d
    \end{bmatrix}''')  # 显示矩阵A的LaTeX格式
    a = st.slider('a', -2.0, 2.0, step=0.1, value=1.0)  # 创建滑块选择矩阵A的元素a
    b = st.slider('b', -2.0, 2.0, step=0.1, value=0.0)  # 创建滑块选择矩阵A的元素b
    c = st.slider('c', -2.0, 2.0, step=0.1, value=0.0)  # 创建滑块选择矩阵A的元素c
    d = st.slider('c', -2.0, 2.0, step=0.1, value=1.0)  # 创建滑块选择矩阵A的元素d

## 定义矩阵变换
theta = np.pi / 6  # 定义角度theta
A = np.array([[a, b], [c, d]], dtype=float)  # 根据用户输入定义矩阵A

## 应用矩阵变换到垂直线
X = np.array(xv)  # 垂直线的x坐标
Y = np.array(yv)  # 垂直线的y坐标
Txvyv = A @ np.stack((X, Y))  # 对垂直线应用矩阵A的变换

## 应用矩阵变换到水平线
X = np.array(xh)  # 水平线的x坐标
Y = np.array(yh)  # 水平线的y坐标
Txhyh = A @ np.stack((X, Y))  # 对水平线应用矩阵A的变换

## 显示矩阵及其计算结果
st.latex(r'A = ' + bmatrix(A))  # 显示矩阵A的LaTeX格式
a1 = A[:, 0].reshape((-1, 1))  # 提取矩阵A的第一列
a2 = A[:, 1].reshape((-1, 1))  # 提取矩阵A的第二列

st.latex(r'''
         a_1 = Ae_1 = ''' + bmatrix(A) + 'e_1 = ' + bmatrix(a1))  # 显示第一列向量的计算
st.latex(r'''
         a_2 = Ae_2 = ''' + bmatrix(A) + 'e_2 = ' + bmatrix(a2))  # 显示第二列向量的计算
st.latex(r'\begin{vmatrix} A \end{vmatrix} = ' + str(np.linalg.det(A)))  # 显示矩阵A的行列式

## 定义一个正方形并应用矩阵变换
square_x = np.array([0, 1, 1, 0])  # 定义正方形的x坐标
square_y = np.array([0, 0, 1, 1])  # 定义正方形的y坐标
square_array = np.stack((square_x, square_y))  # 将正方形的坐标堆叠为二维数组

fig.add_trace(go.Scatter(x=square_x, y=square_y, fill="toself", line_color='orange'), 1, 1)  # 在第一个子图中绘制原始正方形
A_times_square_array = A @ square_array  # 将正方形应用矩阵A的变换

fig.add_trace(go.Scatter(x=A_times_square_array[0, :], y=A_times_square_array[1, :], fill="toself", line_color='orange'), 1, 2)  # 绘制变换后的正方形
fig.add_trace(go.Scatter(x=Txvyv[0], y=Txvyv[1], mode="lines", line_width=lw, line_color='red'), 1, 2)  # 绘制变换后的垂直线
fig.add_trace(go.Scatter(x=Txhyh[0], y=Txhyh[1], mode="lines", line_width=lw, line_color='blue'), 1, 2)  # 绘制变换后的水平线

## 更新图形布局
fig.update_xaxes(range=[-4, 4])  # 设置x轴范围
fig.update_yaxes(range=[-4, 4])  # 设置y轴范围
fig.update_layout(width=800, height=500, showlegend=False, template="none", plot_bgcolor="white", yaxis2_showgrid=False, xaxis2_showgrid=False)  # 设置图形布局

## 在Streamlit中显示图形
st.plotly_chart(fig)  # 显示绘图
