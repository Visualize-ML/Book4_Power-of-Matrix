
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2025
###############

import streamlit as st  # 导入 Streamlit，用于构建交互式 Web 应用
import plotly.graph_objects as go  # 导入 Plotly 的图形对象模块，用于绘图
import sympy  # 导入 SymPy，用于符号运算
import numpy as np  # 导入 NumPy，用于数值计算

def bmatrix(a):  # 定义一个函数，将 NumPy 数组转换为 LaTeX bmatrix 格式的字符串
    """返回一个 LaTeX 矩阵表示"""
    if len(a.shape) > 2:  # 如果输入数组维度大于 2，抛出异常
        raise ValueError('bmatrix 函数只支持二维矩阵')  
    lines = str(a).replace('[', '').replace(']', '').splitlines()  # 将数组转换为字符串并移除方括号
    rv = [r'\begin{bmatrix}']  # LaTeX 矩阵开始符号
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]  # 按行格式化
    rv += [r'\end{bmatrix}']  # LaTeX 矩阵结束符号
    return '\n'.join(rv)  # 返回拼接后的 LaTeX 字符串

with st.sidebar:  # 在侧边栏中创建交互内容
    st.latex(r'''A = \begin{bmatrix} a & b\\ b & c \end{bmatrix}''')  # 显示矩阵 A 的 LaTeX 表示
    st.latex(r'''f(x_1,x_2) = ax_1^2 + 2bx_1x_2 + cx_2^2''')  # 显示二次形式的 LaTeX 表示
    a = st.slider('a', -2.0, 2.0, step=0.1)  # 创建滑块，用于设置矩阵 A 的元素 a
    b = st.slider('b', -2.0, 2.0, step=0.1)  # 创建滑块，用于设置矩阵 A 的元素 b
    c = st.slider('c', -2.0, 2.0, step=0.1)  # 创建滑块，用于设置矩阵 A 的元素 c

x1_ = np.linspace(-2, 2, 101)  # 在 [-2, 2] 范围内生成 101 个均匀点，用于 x1
x2_ = np.linspace(-2, 2, 101)  # 同样生成 x2 的点
xx1, xx2 = np.meshgrid(x1_, x2_)  # 生成网格点，方便绘制 3D 和等高线图

x1, x2 = sympy.symbols('x1 x2')  # 定义符号变量 x1 和 x2
A = np.array([[a, b],  # 定义矩阵 A 的第一行
              [b, c]])  # 定义矩阵 A 的第二行
D, V = np.linalg.eig(A)  # 计算矩阵 A 的特征值 D 和特征向量 V
D = np.diag(D)  # 将特征值转化为对角矩阵

st.latex(r'''A = \begin{bmatrix}%s & %s\\%s & %s\end{bmatrix}''' % (a, b, b, c))  # 显示矩阵 A 的 LaTeX 表示
st.latex(r'''A = V \Lambda V^{T}''')  # 显示特征分解公式
st.latex(bmatrix(A) + '=' + bmatrix(np.around(V, decimals=3)) + '@' + bmatrix(np.around(D, decimals=3)) + '@' + bmatrix(np.around(V.T, decimals=3)))  # 显示特征分解的详细过程

x = np.array([[x1, x2]]).T  # 定义符号向量 x
f_x = a * x1**2 + 2 * b * x1 * x2 + c * x2**2  # 定义二次形式 f(x1, x2)
st.latex(r'''f(x_1,x_2) = ''')  # 显示二次形式的 LaTeX 表示
st.write(f_x)  # 显示二次形式的符号表达式

f_x_fcn = sympy.lambdify([x1, x2], f_x)  # 将符号函数 f(x1, x2) 转换为数值计算函数
ff_x = f_x_fcn(xx1, xx2)  # 在网格点上计算二次形式的值

fig_surface = go.Figure(go.Surface(  # 创建 3D 表面图
    x=x1_,  # 表面图的 x 轴为 x1 的值
    y=x2_,  # 表面图的 y 轴为 x2 的值
    z=ff_x,  # 表面图的 z 轴为二次形式的值
    colorscale='RdYlBu_r'))  # 使用红黄蓝颜色映射
fig_surface.update_layout(
    autosize=False,  # 禁用自动调整尺寸
    width=500,  # 设置图表宽度为 500 像素
    height=500)  # 设置图表高度为 500 像素
st.plotly_chart(fig_surface)  # 在 Streamlit 页面上显示 3D 表面图

fig_contour = go.Figure(  # 创建 2D 等高线图
    go.Contour(
        z=ff_x,  # 等高线的高度值
        x=x1_,  # 等高线图的 x 轴为 x1 的值
        y=x2_,  # 等高线图的 y 轴为 x2 的值
        colorscale='RdYlBu_r'  # 使用红黄蓝颜色映射
    ))
fig_contour.update_layout(
    autosize=False,  # 禁用自动调整尺寸
    width=500,  # 设置图表宽度为 500 像素
    height=500)  # 设置图表高度为 500 像素
st.plotly_chart(fig_contour)  # 在 Streamlit 页面上显示 2D 等高线图

