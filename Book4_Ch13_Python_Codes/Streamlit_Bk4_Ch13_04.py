
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import streamlit as st  # 导入 Streamlit 库，用于创建交互式 Web 应用
import numpy as np  # 导入 NumPy 库，用于数值计算
import plotly.express as px  # 导入 Plotly Express 库，用于绘制交互式图表
import pandas as pd  # 导入 Pandas 库，用于数据处理

# 定义函数 bmatrix，用于将 NumPy 数组转化为 LaTeX 矩阵格式
def bmatrix(a):
    """返回一个 LaTeX 矩阵表示"""
    if len(a.shape) > 2:  # 检查输入的数组是否为二维
        raise ValueError('bmatrix 函数最多显示二维矩阵')  # 如果不是二维数组，抛出异常
    lines = str(a).replace('[', '').replace(']', '').splitlines()  # 去掉数组的方括号并按行拆分
    rv = [r'\begin{bmatrix}']  # 开始 LaTeX 矩阵的表示
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]  # 将每一行的元素用 LaTeX 格式化
    rv += [r'\end{bmatrix}']  # 结束 LaTeX 矩阵的表示
    return '\n'.join(rv)  # 返回拼接后的 LaTeX 字符串

# 在侧边栏创建交互式滑块，用户可调整矩阵 A 的元素
with st.sidebar:
    # 在侧边栏中展示一个 LaTeX 格式的矩阵模板
    st.latex(r'''
             A = \begin{bmatrix}
    a & b\\
    c & d
    \end{bmatrix}''')

    # 为矩阵 A 的元素 a, b, c, d 创建滑块，用户可调整这些值
    a = st.slider('a', -2.0, 2.0, step=0.1, value=1.0)  # 滑块用于设置 a 的值，默认值为 1.0
    b = st.slider('b', -2.0, 2.0, step=0.1, value=0.0)  # 滑块用于设置 b 的值，默认值为 0.0
    c = st.slider('c', -2.0, 2.0, step=0.1, value=0.0)  # 滑块用于设置 c 的值，默认值为 0.0
    d = st.slider('d', -2.0, 2.0, step=0.1, value=1.0)  # 滑块用于设置 d 的值，默认值为 1.0

#%% 创建网格点用于二维平面上的点
x1_ = np.linspace(-1, 1, 11)  # 在 [-1, 1] 区间内生成 11 个均匀分布的点，用于 x1
x2_ = np.linspace(-1, 1, 11)  # 在 [-1, 1] 区间内生成 11 个均匀分布的点，用于 x2

xx1, xx2 = np.meshgrid(x1_, x2_)  # 创建二维网格，用于生成所有点的坐标
X = np.column_stack((xx1.flatten(), xx2.flatten()))  # 将网格点展开为二维数组，每行一个点的坐标

# 定义矩阵 A，由用户调整的滑块值确定
A = np.array([[a, b],  # 矩阵 A 的第一行
              [c, d]])  # 矩阵 A 的第二行

X = X @ A  # 使用矩阵乘法，将点集 X 通过矩阵 A 进行线性变换

#%% 创建颜色数组并将其添加到点数据中
color_array = np.linspace(0, 1, len(X))  # 为每个点生成一个对应的颜色值，范围为 [0, 1]
X = np.column_stack((X, color_array))  # 将颜色值添加到点数据中，作为第三列
df = pd.DataFrame(X, columns=['z1', 'z2', 'color'])  # 将点数据转换为 DataFrame，并命名列为 z1, z2, 和 color

#%% 绘制散点图
st.latex('A = ' + bmatrix(A))  # 在页面上以 LaTeX 格式展示矩阵 A

# 使用 Plotly Express 绘制散点图
fig = px.scatter(df,  # 数据来源为 DataFrame
                 x="z1",  # z1 作为横轴
                 y="z2",  # z2 作为纵轴
                 color='color',  # 根据 color 列设置点的颜色
                 color_continuous_scale='rainbow')  # 使用彩虹色带表示颜色

# 设置图形的布局参数
fig.update_layout(
    autosize=False,  # 禁用自动尺寸调整
    width=500,  # 设置图形宽度为 500 像素
    height=500)  # 设置图形高度为 500 像素

# 添加横轴和纵轴的黑色参考线
fig.add_hline(y=0, line_color='black')  # 添加横轴参考线
fig.add_vline(x=0, line_color='black')  # 添加纵轴参考线

# 设置坐标轴的显示范围
fig.update_xaxes(range=[-3, 3])  # 设置 x 轴范围为 [-3, 3]
fig.update_yaxes(range=[-3, 3])  # 设置 y 轴范围为 [-3, 3]

# 禁用颜色条显示
fig.update_coloraxes(showscale=False)  # 隐藏颜色条

# 在 Streamlit 页面中展示绘制的散点图
st.plotly_chart(fig)




