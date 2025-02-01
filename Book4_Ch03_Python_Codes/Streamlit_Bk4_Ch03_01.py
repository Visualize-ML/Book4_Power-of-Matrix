
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2025
###############

## 导入必要的库
import streamlit as st  # 引入Streamlit库，用于创建交互式Web应用
import plotly.graph_objects as go  # 引入Plotly图形对象模块，用于绘图
import numpy as np  # 引入NumPy库，用于数值计算

## 定义网格数据
x1 = np.linspace(-2.5, 2.5, num=101)  # 生成x1的线性等间距数组，范围为-2.5到2.5，共101个点
x2 = x1  # 定义x2与x1相同
xx1, xx2 = np.meshgrid(x1, x2)  # 使用meshgrid生成二维网格数据

## 创建侧边栏以选择参数p
with st.sidebar:  # 在侧边栏中设置参数
    st.write('Note: Lp norm, p >= 1')  # 显示提示信息：Lp范数，p >= 1
    p = st.slider('p',  # 创建滑块选择p值
                  min_value=-20.0,  # p值的最小值为-20.0
                  max_value=20.0,  # p值的最大值为20.0
                  step=0.2)  # p值的步长为0.2

## 计算Lp范数的二维数据
zz = ((np.abs((xx1))**p) + (np.abs((xx2))**p))**(1. / p)  # 根据Lp范数公式计算二维数据zz

## 创建二维等高线图
fig_1 = go.Figure(data=
                  go.Contour(z=zz,  # 设置z轴数据为zz
                             x=x1,  # 设置x轴数据为x1
                             y=x2,  # 设置y轴数据为x2
                             colorscale='RdYlBu_r'))  # 设置颜色映射为红黄蓝反转

fig_1.update_layout(  # 更新图形布局
    autosize=False,  # 关闭自动调整大小
    width=500,  # 设置图形宽度为500
    height=500,  # 设置图形高度为500
    margin=dict(  # 设置图形边距
        l=50,  # 左边距为50
        r=50,  # 右边距为50
        b=50,  # 底边距为50
        t=50))  # 顶边距为50

## 创建三维表面图
fig_2 = go.Figure(
    go.Surface(
        x=x1,  # 设置x轴数据为x1
        y=x2,  # 设置y轴数据为x2
        z=zz,  # 设置z轴数据为zz
        colorscale='RdYlBu_r'))  # 设置颜色映射为红黄蓝反转

## 展示二维等高线图
with st.expander('2D contour'):  # 创建可展开部分，标题为“2D contour”
    st.plotly_chart(fig_1)  # 在Streamlit中显示二维等高线图

## 展示三维表面图
with st.expander('3D surface'):  # 创建可展开部分，标题为“3D surface”
    st.plotly_chart(fig_2)  # 在Streamlit中显示三维表面图
