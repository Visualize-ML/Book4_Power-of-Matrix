###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2025
###############

import streamlit as st  # 导入 Streamlit 库，用于创建交互式 Web 应用
import plotly.express as px  # 导入 Plotly Express，用于绘制交互式图表

import numpy as np  # 导入 NumPy，用于数值计算
import pandas as pd  # 导入 Pandas，用于数据处理
from sklearn.datasets import load_iris  # 从 scikit-learn 导入 Iris 数据集

#%%

# 加载 Iris 数据集
iris = load_iris()  # 加载 Iris 数据集
X = iris.data  # 提取特征数据
y = iris.target  # 提取目标标签

# 定义特征名称
feature_names = ['Sepal length, x1', 'Sepal width, x2',
                 'Petal length, x3', 'Petal width, x4']  # 定义特征名称

# 将 NumPy 数组 X 转换为 Pandas DataFrame
X_df = pd.DataFrame(X, columns=feature_names)  # 创建 DataFrame 并指定列名为特征名称

# 在侧边栏中展示矩阵分解公式
with st.sidebar:
    st.latex('X = USV^T')  # 展示 SVD 的公式
    st.latex('X = \sum_{j=1}^{D} s_j u_j v_j^T')  # 展示矩阵分解的展开公式
    st.latex('X \simeq \sum_{j=1}^{p} s_j u_j v_j^T')  # 展示矩阵近似公式

#%% 原始数据 X

X = X_df.to_numpy()  # 将 DataFrame 转换回 NumPy 数组

# 计算矩阵 X 的奇异值分解
U, S, V_T = np.linalg.svd(X, full_matrices=False)  # 进行 SVD 分解
S = np.diag(S)  # 将奇异值转换为对角矩阵
V = V_T.T  # 转置右奇异矩阵以获得列向量形式

# 在侧边栏中添加滑块，用于选择近似矩阵的成分数 p
with st.sidebar:
    p = st.slider('Choose p, number of component to approximate X:',  # 滑块标题
                  1, 4, step=1)  # 滑块范围为 1 到 4，步进为 1

#%% 近似矩阵 X

# 使用前 p 个奇异值、左奇异向量和右奇异向量近似原始矩阵 X
X_apprx = U[:, 0:p] @ S[0:p, 0:p] @ V[:, 0:p].T  # 根据前 p 个成分计算近似矩阵
X_apprx_df = pd.DataFrame(X_apprx, columns=feature_names)  # 将近似矩阵转换为 DataFrame

# 计算误差矩阵
Error_df = X_df - X_apprx_df  # 原始矩阵与近似矩阵的差

#%% 可视化原始矩阵、近似矩阵和误差矩阵

# 使用 Streamlit 的列布局
col1, col2, col3 = st.columns(3)  # 创建三列布局

# 在第一列中显示原始矩阵 X
with col1:
    st.latex('X')  # 显示原始矩阵的 LaTeX 表示
    fig_1 = px.imshow(X_df,  # 绘制热图表示原始矩阵
                      color_continuous_scale='RdYlBu_r',  # 使用红黄蓝色带
                      range_color=[0, 8])  # 设置颜色范围

    fig_1.layout.height = 500  # 设置图像高度为 500 像素
    fig_1.layout.width = 300  # 设置图像宽度为 300 像素
    fig_1.update_layout(coloraxis_showscale=False)  # 隐藏颜色条
    st.plotly_chart(fig_1)  # 在 Streamlit 页面中显示图表

# 在第二列中显示近似矩阵 X_apprx
with col2:
    st.latex('\hat{X}')  # 显示近似矩阵的 LaTeX 表示
    fig_2 = px.imshow(X_apprx_df,  # 绘制热图表示近似矩阵
                      color_continuous_scale='RdYlBu_r',  # 使用红黄蓝色带
                      range_color=[0, 8])  # 设置颜色范围

    fig_2.layout.height = 500  # 设置图像高度为 500 像素
    fig_2.layout.width = 300  # 设置图像宽度为 300 像素
    fig_2.update_layout(coloraxis_showscale=False)  # 隐藏颜色条
    st.plotly_chart(fig_2)  # 在 Streamlit 页面中显示图表

# 在第三列中显示误差矩阵 X - X_apprx
with col3:
    st.latex('X - \hat{X}')  # 显示误差矩阵的 LaTeX 表示
    fig_3 = px.imshow(Error_df,  # 绘制热图表示误差矩阵
                      color_continuous_scale='RdYlBu_r',  # 使用红黄蓝色带
                      range_color=[0, 8])  # 设置颜色范围

    fig_3.layout.height = 500  # 设置图像高度为 500 像素
    fig_3.layout.width = 300  # 设置图像宽度为 300 像素
    fig_3.update_layout(coloraxis_showscale=False)  # 隐藏颜色条
    st.plotly_chart(fig_3)  # 在 Streamlit 页面中显示图表


