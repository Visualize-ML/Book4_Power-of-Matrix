
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2025
###############


## 导入必要的库
import streamlit as st  # 引入Streamlit库，用于创建交互式Web应用
import numpy as np  # 引入Numpy库，用于数值计算
from plotly.subplots import make_subplots  # 引入Plotly的子图模块，用于创建复杂布局
import plotly.graph_objects as go  # 引入Plotly的图形对象模块，用于绘图
import plotly.express as px  # 引入Plotly Express模块，用于快速绘图

## 定义自定义函数 bmatrix
# 该函数将一个numpy数组转换为LaTeX格式的bmatrix矩阵
def bmatrix(a):
    """返回一个LaTeX格式的bmatrix矩阵表示形式

    :a: numpy数组
    :returns: 返回字符串形式的LaTeX bmatrix矩阵
    """
    if len(a.shape) > 2:  # 如果数组维度大于2，抛出异常
        raise ValueError('bmatrix最多只能展示二维数据')
    lines = str(a).replace('[', '').replace(']', '').splitlines()  # 移除括号并分行
    rv = [r'\begin{bmatrix}']  # LaTeX矩阵开始标记
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]  # 拼接矩阵中的元素
    rv += [r'\end{bmatrix}']  # LaTeX矩阵结束标记
    return '\n'.join(rv)  # 返回完整的LaTeX矩阵字符串

## 侧边栏设置
with st.sidebar:  # 创建侧边栏
    # 创建滑块以选择a的行数，范围为3到6，默认值为3
    num_a = st.slider('Number of rows, a:', 3, 6, step=1)  
    # 创建滑块以选择b的行数，范围为3到6，默认值为3
    num_b = st.slider('Number of rows, b:', 3, 6, step=1)

# 随机生成矩阵a，值在0到1之间，形状为(num_a, 1)
a = np.random.uniform(0, 1, num_a).reshape((-1, 1))  
a = np.round(a, 1)  # 将矩阵a的值保留一位小数
# 随机生成矩阵b，值在0到1之间，形状为(num_b, 1)
b = np.random.uniform(0, 1, num_b).reshape((-1, 1))  
b = np.round(b, 1)  # 将矩阵b的值保留一位小数

show_number = False  # 初始化显示数值的变量为False
with st.sidebar:  # 在侧边栏中添加复选框
    # 创建复选框用于控制是否显示矩阵数值
    show_number = st.checkbox('Display values')

# 计算a与b转置的张量积
tensor_a_b = a @ b.T  

## 数据可视化部分
# 在Streamlit中以LaTeX格式展示矩阵a
st.latex('a = ' + bmatrix(a))  
# 在Streamlit中以LaTeX格式展示矩阵b
st.latex('b = ' + bmatrix(b))  
# 展示张量积公式
st.latex('a \\otimes b = ab^{T}')  
# 展示矩阵a与b的张量积计算结果
st.latex(bmatrix(a) + '@' + bmatrix(b.T) + ' = ' + bmatrix(tensor_a_b))  

# 创建三个列布局
col1, col2, col3 = st.columns(3)

# 在第一列展示矩阵a的热图
with col1:
    fig_a = px.imshow(a, text_auto=show_number,  # 绘制热图，并根据show_number显示数值
                      color_continuous_scale='viridis',  # 设置颜色映射
                      aspect='equal')  # 设置宽高比为1:1
    fig_a.update_layout(height=400, width=300)  # 设置图形大小
    fig_a.layout.coloraxis.showscale = False  # 隐藏颜色条
    st.plotly_chart(fig_a)  # 在Streamlit中显示图形

# 在第二列展示矩阵b的热图
with col2:
    fig_b = px.imshow(b, text_auto=show_number,  # 绘制热图，并根据show_number显示数值
                      color_continuous_scale='viridis',  # 设置颜色映射
                      aspect='equal')  # 设置宽高比为1:1
    fig_b.update_layout(height=400, width=300)  # 设置图形大小
    fig_b.layout.coloraxis.showscale = False  # 隐藏颜色条
    st.plotly_chart(fig_b)  # 在Streamlit中显示图形

# 在第三列展示张量积矩阵的热图
with col3:
    fig_ab = px.imshow(tensor_a_b, text_auto=show_number,  # 绘制热图，并根据show_number显示数值
                       color_continuous_scale='viridis',  # 设置颜色映射
                       aspect='equal')  # 设置宽高比为1:1
    fig_ab.update_layout(height=400, width=400)  # 设置图形大小
    fig_ab.layout.coloraxis.showscale = False  # 隐藏颜色条
    st.plotly_chart(fig_ab)  # 在Streamlit中显示图形







