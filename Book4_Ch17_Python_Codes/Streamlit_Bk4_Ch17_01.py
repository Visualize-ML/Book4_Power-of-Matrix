
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2025
###############

import sympy  # 导入 sympy，用于符号计算
import numpy as np  # 导入 NumPy，用于数值计算
from sympy.functions import exp  # 从 sympy 导入指数函数
import streamlit as st  # 导入 Streamlit，用于创建交互式 Web 应用
import plotly.figure_factory as ff  # 导入 Plotly 工厂方法，用于可视化
import plotly.graph_objects as go  # 导入 Plotly 图形对象模块，用于复杂图形

# 定义符号变量和函数
x1, x2 = sympy.symbols('x1 x2')  # 定义符号变量 x1 和 x2
f_x = x1 * exp(-(x1**2 + x2**2))  # 定义函数 f(x1, x2) = x1 * exp(-(x1^2 + x2^2))

# 在页面上显示函数的 LaTeX 表示
st.latex('f(x_1, x_2) = ' + sympy.latex(f_x))  # 显示 f(x1, x2) 的 LaTeX 表达式

# 计算函数的梯度
grad_f = [sympy.diff(f_x, var) for var in (x1, x2)]  # 对 x1 和 x2 求偏导，得到梯度
st.latex(r'\nabla f = ' + sympy.latex(grad_f) + '^T')  # 显示梯度的 LaTeX 表达式

# 将符号函数转换为数值计算函数
f_x_fcn = sympy.lambdify([x1, x2], f_x)  # 将 f_x 转换为 Python 函数
grad_fcn = sympy.lambdify([x1, x2], grad_f)  # 将梯度 grad_f 转换为 Python 函数

# 定义 x1 和 x2 的值域，用于生成网格
x1_array = np.linspace(-2, 2, 100)  # 在 [-2, 2] 范围内生成 100 个均匀点
x2_array = np.linspace(-2, 2, 100)  # 同样生成 x2 的点

# 创建细网格，用于绘制函数表面
xx1, xx2 = np.meshgrid(x1_array, x2_array)  # 创建细网格
# 创建粗网格，用于绘制梯度场
xx1_, xx2_ = np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-2, 2, 20))  # 创建粗网格

# 在粗网格上计算梯度向量
V = grad_fcn(xx1_, xx2_)  # 使用梯度函数计算粗网格上的梯度向量
# 在细网格上计算函数值
ff_x = f_x_fcn(xx1, xx2)  # 使用函数 f_x 计算细网格上的函数值

#%% 可视化

# 绘制函数表面
fig_surface = go.Figure(go.Surface(
    x=x1_array,  # 表面图的 x 轴为 x1 的值
    y=x2_array,  # 表面图的 y 轴为 x2 的值
    z=ff_x,  # 表面图的 z 轴为函数值
    showscale=False))  # 禁用颜色条
fig_surface.update_layout(
    autosize=False,  # 禁用自动调整大小
    width=800,  # 设置图表宽度为 800 像素
    height=800)  # 设置图表高度为 800 像素

# 在 Streamlit 页面上显示表面图
st.plotly_chart(fig_surface)

# 创建梯度场和等高线图
f = ff.create_quiver(xx1_, xx2_, V[0], V[1])  # 创建梯度场图
trace1 = f.data[0]  # 提取梯度场的图层数据
trace2 = go.Contour(  # 创建等高线图
    x=x1_array,  # 等高线图的 x 轴为 x1 的值
    y=x2_array,  # 等高线图的 y 轴为 x2 的值
    z=ff_x,  # 等高线图的 z 轴为函数值
    showscale=False)  # 禁用颜色条

# 将梯度场和等高线图合并为一个图形
data = [trace1, trace2]  # 合并两个图层数据
fig = go.FigureWidget(data)  # 创建图形对象
fig.update_layout(
    autosize=False,  # 禁用自动调整大小
    width=800,  # 设置图表宽度为 800 像素
    height=800)  # 设置图表高度为 800 像素

# 添加辅助线
fig.add_hline(y=0, line_color='black')  # 添加水平辅助线
fig.add_vline(x=0, line_color='black')  # 添加垂直辅助线

# 设置坐标轴范围
fig.update_xaxes(range=[-2, 2])  # 设置 x 轴范围为 [-2, 2]
fig.update_yaxes(range=[-2, 2])  # 设置 y 轴范围为 [-2, 2]
fig.update_coloraxes(showscale=False)  # 禁用颜色条

# 在 Streamlit 页面上显示梯度场和等高线图
st.plotly_chart(fig)



