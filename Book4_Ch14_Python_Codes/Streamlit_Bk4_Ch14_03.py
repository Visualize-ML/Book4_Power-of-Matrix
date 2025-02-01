
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2025
###############


import plotly.graph_objects as go  # 导入 Plotly 的图形对象模块，用于创建复杂的图形
import streamlit as st  # 导入 Streamlit 库，用于创建交互式 Web 应用
import numpy as np  # 导入 NumPy，用于数值计算
import plotly.express as px  # 导入 Plotly Express，用于快速绘制图表
import pandas as pd  # 导入 Pandas，用于数据处理
import sympy  # 导入 SymPy，用于符号运算和公式化表达

# 定义函数 bmatrix，将 NumPy 数组转换为 LaTeX 格式的矩阵表示
def bmatrix(a):
    """返回一个 LaTeX 矩阵表示"""
    if len(a.shape) > 2:  # 检查输入是否为二维数组
        raise ValueError('bmatrix 函数最多显示二维矩阵')  # 如果不是二维，抛出异常
    lines = str(a).replace('[', '').replace(']', '').splitlines()  # 将数组转换为字符串并去掉方括号
    rv = [r'\begin{bmatrix}']  # 开始 LaTeX 矩阵的格式
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]  # 逐行添加 LaTeX 矩阵行
    rv += [r'\end{bmatrix}']  # 结束 LaTeX 矩阵的格式
    return '\n'.join(rv)  # 返回拼接后的 LaTeX 字符串

# 创建 Streamlit 侧边栏，用于调整矩阵 A 的参数
with st.sidebar:
    # 显示矩阵 A 的 LaTeX 表示
    st.latex(r'''
             A = \begin{bmatrix}
    a & b\\
    b & c
    \end{bmatrix}''')

    # 创建滑块，允许用户调整矩阵 A 的元素 a, b, c 的值
    a = st.slider('a', -2.0, 2.0, step=0.05, value=1.0)  # 滑块用于调整 a 的值
    b = st.slider('b', -2.0, 2.0, step=0.05, value=0.0)  # 滑块用于调整 b 的值
    c = st.slider('c', -2.0, 2.0, step=0.05, value=1.0)  # 滑块用于调整 c 的值

#%% 创建一个单位圆的点集
theta_array = np.linspace(0, 2 * np.pi, 36)  # 在 [0, 2π] 区间生成 36 个点，表示角度
X = np.column_stack((np.cos(theta_array),  # 用 cos 和 sin 创建单位圆上的点
                     np.sin(theta_array)))

# 创建矩阵 A
A = np.array([[a, b],  # 矩阵 A 的第一行
              [b, c]])  # 矩阵 A 的第二行

# 显示单位圆的方程和线性变换后的方程
st.latex(r'''z^Tz = 1''')  # 显示单位圆的方程
st.latex(r'''x = Az''')  # 显示线性变换的方程

# 显示矩阵 A 的 LaTeX 表示
st.latex('A =' + bmatrix(A))

# 对单位圆的点集进行线性变换
X_ = X @ A  # 对单位圆上的点集 X 应用线性变换矩阵 A

#%% 使用符号运算求解椭圆的方程
x1, x2 = sympy.symbols('x1 x2')  # 定义符号变量 x1 和 x2
y1, y2 = sympy.symbols('y1 y2')  # 定义符号变量 y1 和 y2
x = np.array([[x1, x2]]).T  # 定义符号向量 x
y = np.array([[y1, y2]]).T  # 定义符号向量 y

# 计算 Q 矩阵
Q = np.linalg.inv(A @ A.T)  # Q = (AA^T)^(-1)
D, V = np.linalg.eig(Q)  # 计算 Q 的特征值和特征向量
D = np.diag(D)  # 将特征值转化为对角矩阵

# 显示 Q 矩阵的分解
st.latex(r'Q = \left( AA^T\right)^{-1} = ' + bmatrix(np.round(Q, 3)))  # 显示 Q 矩阵
st.latex(r'''Q = V \Lambda V^{T}''')  # 显示特征分解公式
st.latex(bmatrix(np.around(Q, decimals=3)) + '=' + 
         bmatrix(np.around(V, decimals=3)) + '@' + 
         bmatrix(np.around(D, decimals=3)) + '@' + 
         bmatrix(np.around(V.T, decimals=3)))  # 显示分解过程

# 定义单位圆和变换后的椭圆方程
f_x = x.T @ np.round(Q, 3) @ x  # 单位圆在 Q 矩阵下的方程
f_y = y.T @ np.round(D, 3) @ y  # 椭圆在对角矩阵 D 下的方程

# 显示椭圆方程
from sympy import *
st.write('The formula of the ellipse:')  # 显示椭圆方程的标题
st.latex(latex(simplify(f_x[0][0])) + ' = 1')  # 显示椭圆方程
st.write('The formula of the transformed ellipse:')  # 显示变换后椭圆方程的标题
st.latex(latex(simplify(f_y[0][0])) + ' = 1')  # 显示变换后的椭圆方程

#%% 添加颜色信息到变换后的点集
color_array = np.linspace(0, 1, len(X))  # 为每个点生成一个颜色值
X_c = np.column_stack((X_, color_array))  # 将颜色信息添加到点集中
df = pd.DataFrame(X_c, columns=['x1', 'x2', 'color'])  # 将点集转换为 DataFrame 格式

#%% 绘制散点图
fig = px.scatter(df,  # 使用 Pandas 数据框作为数据源
                 x="x1",  # 横轴为 x1
                 y="x2",  # 纵轴为 x2
                 color='color',  # 根据颜色值为点上色
                 color_continuous_scale=px.colors.sequential.Rainbow)  # 使用彩虹色带

# 设置图形布局
fig.update_layout(
    autosize=False,  # 禁用自动调整尺寸
    width=500,  # 图表宽度为 500 像素
    height=500)  # 图表高度为 500 像素

# 添加横轴和纵轴的参考线
fig.add_hline(y=0, line_color='black')  # 添加黑色的水平参考线
fig.add_vline(x=0, line_color='black')  # 添加黑色的垂直参考线
fig.update_layout(coloraxis_showscale=False)  # 隐藏颜色条
fig.update_xaxes(range=[-3, 3])  # 设置 x 轴范围
fig.update_yaxes(range=[-3, 3])  # 设置 y 轴范围

# 在 Streamlit 页面上显示图表
st.plotly_chart(fig)


