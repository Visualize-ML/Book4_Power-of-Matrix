
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2025
###############

## 导入必要的库
import plotly.graph_objects as go  # 引入Plotly库，用于绘图
import streamlit as st  # 引入Streamlit库，用于创建交互式Web应用
import numpy as np  # 引入NumPy库，用于数值计算
import plotly.express as px  # 引入Plotly Express库，用于快速绘图
import pandas as pd  # 引入Pandas库，用于数据操作
import sympy  # 引入Sympy库，用于符号计算
from scipy.spatial import distance  # 引入SciPy库中的距离模块

## 定义距离计算函数
# 定义计算Minkowski距离的函数
# 如果设置Chebychev为True，则计算Chebychev距离，否则根据参数p计算Minkowski距离
def fcn_Minkowski(xx, yy, mu, p=2, Chebychev=False):
    if Chebychev:  # 如果选择Chebychev距离
        zz = np.maximum(np.abs(xx - mu[0]), np.abs(yy - mu[1]))  # 计算Chebychev距离
    else:  # 否则计算Minkowski距离
        zz = ((np.abs((xx - mu[0]))**p) + (np.abs((yy - mu[1]))**p))**(1. / p)  # 计算公式
    return zz  # 返回计算结果

# 定义计算Mahalanobis距离的函数
# 可以选择是否标准化Sigma矩阵

def fcn_mahal(xx, yy, mu, Sigma, standardized=False):
    if standardized:  # 如果选择标准化
        D = np.diag(np.diag(Sigma))  # 提取对角线元素
        Sigma_inv = np.linalg.inv(D)  # 计算逆矩阵
    else:  # 否则直接计算Sigma的逆矩阵
        Sigma_inv = np.linalg.inv(Sigma)
    xy_ = np.stack((xx.flatten(), yy.flatten())).T  # 将输入数组展平并堆叠
    zz = np.diag(np.sqrt(np.dot(np.dot((xy_ - mu), Sigma_inv), (xy_ - mu).T)))  # 计算Mahalanobis距离
    zz = np.reshape(zz, xx.shape)  # 重塑计算结果为网格形状
    return zz  # 返回计算结果

## 加载数据集
# 使用Plotly自带的鸢尾花数据集

df = px.data.iris()

## 创建侧边栏
with st.sidebar:  # 定义侧边栏区域
    dist_type = st.radio('Choose a type of distance: ',  # 添加单选按钮选择距离类型
                         options=['Euclidean', 'City block', 'Minkowski', 'Chebychev', 'Mahalanobis', 'Standardized Euclidean'])
    if dist_type == 'Minkowski':  # 如果选择Minkowski距离
        with st.sidebar:  # 添加滑块以选择p值
            p = st.slider('Specify a p value:', 1.0, 20.0, step=0.5)

## 计算距离
X = df[['sepal_length', 'petal_length']]  # 提取鸢尾花数据集中感兴趣的特征
mu = X.mean().to_numpy()  # 计算特征的均值
Sigma = X.cov().to_numpy()  # 计算特征的协方差矩阵
x_array = np.linspace(0, 10, 101)  # 定义x轴网格点
y_array = np.linspace(0, 10, 101)  # 定义y轴网格点
xx, yy = np.meshgrid(x_array, y_array)  # 创建二维网格

if dist_type == 'Minkowski':  # 如果选择Minkowski距离
    zz = fcn_Minkowski(xx, yy, mu, p)  # 计算Minkowski距离
elif dist_type == 'Euclidean':  # 如果选择欧几里得距离
    zz = fcn_Minkowski(xx, yy, mu, 2)  # 使用p=2计算Minkowski距离
elif dist_type == 'Chebychev':  # 如果选择Chebychev距离
    zz = fcn_Minkowski(xx, yy, mu, Chebychev=True)  # 调用函数计算Chebychev距离
elif dist_type == 'Mahalanobis':  # 如果选择Mahalanobis距离
    zz = fcn_mahal(xx, yy, mu, Sigma)  # 调用函数计算Mahalanobis距离
elif dist_type == 'City block':  # 如果选择曼哈顿距离
    zz = fcn_Minkowski(xx, yy, mu, 1)  # 使用p=1计算Minkowski距离
elif dist_type == 'Standardized Euclidean':  # 如果选择标准化欧几里得距离
    zz = fcn_mahal(xx, yy, mu, Sigma, True)  # 调用函数计算标准化欧几里得距离

## 数据可视化
st.title(dist_type + ' distance')  # 设置标题
fig_2 = px.scatter(df, x='sepal_length', y='petal_length')  # 绘制散点图
fig_2.add_trace(go.Contour(  # 添加等高线
    x=x_array,  # 设置x轴网格点
    y=y_array,  # 设置y轴网格点
    z=zz,  # 设置计算的距离数据
    contours_coloring='lines',  # 等高线的样式
    showscale=False))  # 不显示颜色条
fig_2.add_traces(
    px.scatter(X.mean().to_frame().T,  # 绘制均值点
               x='sepal_length',  # x轴为花萼长度
               y='petal_length').update_traces(
        marker_size=20,  # 设置标记大小
        marker_color="red",  # 设置标记颜色
        marker_symbol='x').data)
fig_2.update_layout(yaxis_range=[0, 10])  # 设置y轴范围
fig_2.update_layout(xaxis_range=[0, 10])  # 设置x轴范围
fig_2.add_hline(y=mu[1])  # 添加水平线
fig_2.add_vline(x=mu[0])  # 添加垂直线
fig_2.update_yaxes(
    scaleratio=1,  # 设置y轴比例
)
fig_2.update_layout(width=600, height=600)  # 设置图形尺寸
st.plotly_chart(fig_2)  # 在Streamlit中显示图形



    