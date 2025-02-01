
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2025
###############

import pandas as pd
import plotly.graph_objs as go
import streamlit as st
import numpy as np

# 使用 Streamlit 的侧边栏设置滑块，用于选择每个维度的数据点数量
with st.sidebar:
    num = st.slider('Number of points for each dimension',  # 滑块标题
                    max_value=20,  # 最大值为20
                    min_value=10,  # 最小值为10
                    step=1)  # 步长为1

# 生成从0到1均匀分布的线性空间数据点
x1 = np.linspace(0, 1, num)
x2 = x1  # x2 和 x1 相同
x3 = x1  # x3 和 x1 相同

# 生成三维网格，用于三维坐标的组合
xx1, xx2, xx3 = np.meshgrid(x1, x2, x3)

# 将网格展开为一维数组
x1_ = xx1.ravel()
x2_ = xx2.ravel()
x3_ = xx3.ravel()

# 创建一个 Pandas DataFrame，存储三维坐标和对应的RGB颜色分量
df = pd.DataFrame({'X': x1_,  # x 坐标
                   'Y': x2_,  # y 坐标
                   'Z': x3_,  # z 坐标
                   'R': (x1_ * 256).round(),  # R 通道值
                   'G': (x2_ * 256).round(),  # G 通道值
                   'B': (x3_ * 256).round()})  # B 通道值

# 创建 3D 散点图的跟踪对象
trace = go.Scatter3d(
    x=df.X,  # x 轴数据
    y=df.Y,  # y 轴数据
    z=df.Z,  # z 轴数据
    mode='markers',  # 数据点的显示模式为散点
    marker=dict(
        size=3,  # 数据点的大小
        color=['rgb({},{},{})'.format(r, g, b)  # 将 RGB 分量转换为颜色字符串
               for r, g, b in zip(df.R.values, df.G.values, df.B.values)],
        opacity=0.9,  # 数据点的不透明度
    )
)

# 将散点图添加到数据列表中
data = [trace]

# 定义 3D 图的布局，包括坐标轴和边距
layout = go.Layout(
    margin=dict(l=0, r=0, b=0, t=0),  # 图形边距设置为0
    scene=dict(
        xaxis=dict(title='e_1'),  # x 轴标题
        yaxis=dict(title='e_2'),  # y 轴标题
        zaxis=dict(title='e_3'),  # z 轴标题
    ),
)

# 创建包含数据和布局的图表对象
fig = go.Figure(data=data, layout=layout)

# 使用 Streamlit 显示图表
st.plotly_chart(fig)

