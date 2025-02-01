
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2025
###############

import numpy as np  # 导入 NumPy，用于数值计算
from sympy import lambdify, diff, exp, latex, simplify, symbols  # 从 SymPy 导入符号计算相关模块
import plotly.figure_factory as ff  # 从 Plotly 导入工厂方法，用于绘制梯度向量和流线
import plotly.graph_objects as go  # 从 Plotly 导入图形对象模块，用于绘图
import streamlit as st  # 导入 Streamlit，用于构建交互式 Web 应用

x1, x2 = symbols('x1 x2')  # 定义符号变量 x1 和 x2
num = 301  # 设置网格点数量
x1_array = np.linspace(-3, 3, num)  # 在 [-3, 3] 范围内生成 301 个均匀点用于 x1
x2_array = np.linspace(-3, 3, num)  # 在 [-3, 3] 范围内生成 301 个均匀点用于 x2
xx1, xx2 = np.meshgrid(x1_array, x2_array)  # 创建网格点，用于绘制函数和梯度图

# 定义函数 f(x1, x2)
f_x = 3 * (1 - x1)**2 * exp(-(x1**2) - (x2 + 1)**2) \
    - 10 * (x1 / 5 - x1**3 - x2**5) * exp(-x1**2 - x2**2) \
    - 1 / 3 * exp(-(x1 + 1)**2 - x2**2)  # 定义复杂的二元函数

f_x_fcn = lambdify([x1, x2], f_x)  # 将符号函数 f_x 转换为数值计算函数
f_zz = f_x_fcn(xx1, xx2)  # 在网格点上计算函数值，用于绘制表面图和等高线图

st.latex('f(x_1, x_2) = ' + latex(f_x))  # 在 Streamlit 页面中显示函数的 LaTeX 表示

# 计算梯度
grad_f = [diff(f_x, var) for var in (x1, x2)]  # 对 f_x 分别对 x1 和 x2 求偏导，得到梯度向量
grad_fcn = lambdify([x1, x2], grad_f)  # 将梯度向量转换为数值计算函数

x1__ = np.linspace(-3, 3, 40)  # 在 [-3, 3] 范围内生成 40 个点用于 x1（粗网格）
x2__ = np.linspace(-3, 3, 40)  # 在 [-3, 3] 范围内生成 40 个点用于 x2（粗网格）
xx1_, xx2_ = np.meshgrid(x1__, x2__)  # 创建粗网格点
V = grad_fcn(xx1_, xx2_)  # 在粗网格点上计算梯度向量，用于绘制梯度图

# 绘制 3D 表面图
fig_surface = go.Figure(go.Surface(
    x=x1_array,  # 表面图的 x 轴为 x1 网格点
    y=x2_array,  # 表面图的 y 轴为 x2 网格点
    z=f_zz,  # 表面图的 z 轴为函数值
    showscale=False,  # 禁用颜色条
    colorscale='RdYlBu_r'))  # 使用红黄蓝色带
fig_surface.update_layout(
    autosize=False,  # 禁用自动调整尺寸
    width=800,  # 设置图表宽度为 800 像素
    height=600)  # 设置图表高度为 600 像素
st.plotly_chart(fig_surface)  # 在 Streamlit 页面中显示 3D 表面图

# 绘制梯度向量图和等高线图
f = ff.create_quiver(xx1_, xx2_, V[0], V[1], arrow_scale=.1, scale=0.03)  # 创建梯度向量图
f_stream = ff.create_streamline(x1__, x2__, V[0], V[1], arrow_scale=.1)  # 创建流线图
trace1 = f.data[0]  # 提取梯度向量的图层数据
trace3 = f_stream.data[0]  # 提取流线的图层数据
trace2 = go.Contour(
    x=x1_array,  # 等高线图的 x 轴为 x1 网格点
    y=x2_array,  # 等高线图的 y 轴为 x2 网格点
    z=f_zz,  # 等高线图的高度值为函数值
    showscale=False,  # 禁用颜色条
    colorscale='RdYlBu_r')  # 使用红黄蓝色带

data = [trace1, trace2]  # 将梯度向量图和等高线图组合
fig = go.FigureWidget(data)  # 创建图形对象
fig.update_layout(
    autosize=False,  # 禁用自动调整尺寸
    width=800,  # 设置图表宽度为 800 像素
    height=800)  # 设置图表高度为 800 像素
fig.add_hline(y=0, line_color='black')  # 添加水平辅助线
fig.add_vline(x=0, line_color='black')  # 添加垂直辅助线
fig.update_xaxes(range=[-2, 2])  # 设置 x 轴范围为 [-2, 2]
fig.update_yaxes(range=[-2, 2])  # 设置 y 轴范围为 [-2, 2]
fig.update_coloraxes(showscale=False)  # 禁用颜色条
st.plotly_chart(fig)  # 在 Streamlit 页面中显示组合图形

# 绘制流线图和等高线图
data2 = [trace3, trace2]  # 将流线图和等高线图组合
fig2 = go.FigureWidget(data2)  # 创建图形对象
fig2.update_layout(
    autosize=False,  # 禁用自动调整尺寸
    width=800,  # 设置图表宽度为 800 像素
    height=800)  # 设置图表高度为 800 像素
fig2.add_hline(y=0, line_color='black')  # 添加水平辅助线
fig2.add_vline(x=0, line_color='black')  # 添加垂直辅助线
fig2.update_xaxes(range=[-2, 2])  # 设置 x 轴范围为 [-2, 2]
fig2.update_yaxes(range=[-2, 2])  # 设置 y 轴范围为 [-2, 2]
fig2.update_coloraxes(showscale=False)  # 禁用颜色条
st.plotly_chart(fig2)  # 在 Streamlit 页面中显示流线图和等高线图
