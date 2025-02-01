
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2025
###############

## 导入必要的库
import streamlit as st  # 导入 Streamlit，用于创建交互式 Web 应用
import plotly.express as px  # 导入 Plotly Express，用于绘图

import seaborn as sns  # 导入 Seaborn，用于数据可视化
import numpy as np  # 导入 NumPy，用于数值计算
import matplotlib.pyplot as plt  # 导入 Matplotlib，用于绘图
import pandas as pd  # 导入 Pandas，用于数据处理
from sklearn.datasets import load_iris  # 从 scikit-learn 导入 Iris 数据集

## 定义函数 bmatrix，用于生成 LaTeX 矩阵
def bmatrix(a):
    """返回一个 LaTeX 矩阵"""
    if len(a.shape) > 2:  # 检查输入是否为二维数组
        raise ValueError('bmatrix 函数最多显示二维矩阵')  # 如果不是二维，抛出异常
    lines = str(a).replace('[', '').replace(']', '').splitlines()  # 将数组转为字符串并去除括号
    rv = [r'\begin{bmatrix}']  # 开始 LaTeX 矩阵的格式
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]  # 为每一行添加 LaTeX 格式
    rv += [r'\end{bmatrix}']  # 结束 LaTeX 矩阵的格式
    return '\n'.join(rv)  # 将所有行连接为字符串并返回

## 加载 Iris 数据集
iris = load_iris()  # 加载 Iris 数据集
X = iris.data  # 提取特征数据
y = iris.target  # 提取目标标签

## 定义特征名称
feature_names = ['Sepal length, x1', 'Sepal width, x2',
                 'Petal length, x3', 'Petal width, x4']  # 定义特征名称

## 将特征数据转换为 DataFrame
X_df = pd.DataFrame(X, columns=feature_names)  # 创建 DataFrame，列名为特征名称

## 原始数据 X
X = X_df.to_numpy()  # 将 DataFrame 转换为 NumPy 数组

## 计算 Gram 矩阵和正交基
G = X.T @ X  # 计算 Gram 矩阵 G
D, V = np.linalg.eig(G)  # 对 Gram 矩阵求特征值和特征向量
np.set_printoptions(suppress=True)  # 设置 NumPy 打印选项，抑制科学记数法
D = np.diag(D)  # 将特征值转换为对角矩阵

## 在 Streamlit 应用中展示计算结果
st.latex(r'G = X^T X = ' + bmatrix(G))  # 展示 Gram 矩阵 G 的 LaTeX 表示
st.latex(r'G = V \Lambda V^T')  # 展示特征分解公式
st.latex(r'G = ' +
         bmatrix(np.round(V, 2)) + '@' +
         bmatrix(np.round(D, 2)) + '@' +
         bmatrix(np.round(V.T, 2)))  # 展示分解结果
st.write('Mapped data:')  # 显示映射数据标题
st.latex('Z = XV')  # 显示 Z 的 LaTeX 表示

## 映射数据 Z
Z = X @ V  # 计算映射数据 Z

## 创建映射数据的 DataFrame
df = pd.DataFrame(Z, columns=['PC1', 'PC2', 'PC3', 'PC4'])  # 创建 DataFrame，列名为主成分
mapping_rule = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}  # 定义类别映射规则
df.insert(4, "species", y)  # 插入类别列
df['species'] = df['species'].map(mapping_rule)  # 应用类别映射

## 提取特征列名称
features = df.columns.to_list()[:-1]  # 提取特征列名称，不包括类别列

## 映射数据表格
with st.expander('Mapped data'):  # 创建可展开区域，标题为 "Mapped data"
    st.write(df)  # 显示映射数据的 DataFrame

## 热力图
with st.expander('Heatmap'):  # 创建可展开区域，标题为 "Heatmap"
    fig_1 = px.imshow(df.iloc[:, 0:4],  # 绘制热力图，仅包含特征列
                      color_continuous_scale='RdYlBu_r')  # 使用指定的颜色映射
    st.plotly_chart(fig_1)  # 显示热力图

## 2D 散点图
with st.sidebar:  # 创建侧边栏
    st.write('2D scatter plot')  # 显示标题 "2D scatter plot"
    x_feature = st.radio('Horizontal axis', features)  # 在侧边栏中选择横轴特征
    y_feature = st.radio('Vertical axis', features)  # 在侧边栏中选择纵轴特征

with st.expander('2D scatter plot'):  # 创建可展开区域，标题为 "2D scatter plot"
    fig_2 = px.scatter(df, x=x_feature, y=y_feature, color="species")  # 绘制 2D 散点图
    st.plotly_chart(fig_2)  # 显示 2D 散点图

## 3D 散点图
with st.expander('3D scatter plot'):  # 创建可展开区域，标题为 "3D scatter plot"
    fig_3 = px.scatter_3d(df,  # 绘制 3D 散点图
                          x='PC1', 
                          y='PC2', 
                          z='PC3',
                          color='species')  # 按类别着色
    st.plotly_chart(fig_3)  # 显示 3D 散点图

## 成对散点图
with st.expander('Pairwise scatter plot'):  # 创建可展开区域，标题为 "Pairwise scatter plot"
    fig_4 = px.scatter_matrix(df,  # 绘制成对散点图
                              dimensions=["PC1", "PC2", "PC3", "PC4"],  # 指定维度
                              color="species")  # 按类别着色
    st.plotly_chart(fig_4)  # 显示成对散点图

    
    
    
