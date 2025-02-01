
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2025
###############

import streamlit as st  # 导入 Streamlit，用于创建交互式 Web 应用
import plotly.graph_objects as go  # 导入 Plotly 图形对象，用于绘图
import sympy  # 导入 SymPy，用于符号运算
import numpy as np  # 导入 NumPy，用于数值计算
from scipy.stats import multivariate_normal  # 从 SciPy 导入多元正态分布

# 定义一个函数，将 NumPy 数组转换为 LaTeX bmatrix 格式
def bmatrix(a):
    """返回一个 LaTeX 矩阵表示"""
    if len(a.shape) > 2:  # 如果数组维度大于2，抛出异常
        raise ValueError('bmatrix 函数最多支持二维矩阵')  
    lines = str(a).replace('[', '').replace(']', '').splitlines()  # 将数组转换为字符串并去掉方括号
    rv = [r'\begin{bmatrix}']  # LaTeX 矩阵起始
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]  # 按行格式化
    rv += [r'\end{bmatrix}']  # LaTeX 矩阵结束
    return '\n'.join(rv)  # 返回拼接后的字符串

# 在侧边栏中创建滑块，用于调整协方差矩阵的参数
with st.sidebar:
    # 显示协方差矩阵的 LaTeX 表示
    st.latex(r'''
             \Sigma = \begin{bmatrix}
    \sigma_1^2 & 
    \rho \sigma_1 \sigma_2 \\
    \rho \sigma_1 \sigma_2 & 
    \sigma_2^2
    \end{bmatrix}''')
    
    # 定义协方差矩阵的元素
    st.write('$\sigma_1$')  # 显示 σ₁
    sigma_1 = st.slider('sigma_1', 1.0, 2.0, step=0.1)  # 创建滑块，用于调整 σ₁
    st.write('$\sigma_2$')  # 显示 σ₂
    sigma_2 = st.slider('sigma_2', 1.0, 2.0, step=0.1)  # 创建滑块，用于调整 σ₂
    st.write('$\\rho$')  # 显示相关系数 ρ
    rho_12 = st.slider('rho', -0.9, 0.9, step=0.1)  # 创建滑块，用于调整 ρ

#%%

# 显示正态分布的概率密度函数公式（1D 和 2D）
st.latex(r'''
   f(x) = \frac{1}{\sqrt{2\pi} \sigma} 
          \exp\left( -\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^{\!2}\,\right)
          ''')  # 1D 正态分布公式
st.latex(r'''
   f(x) = \frac{1}{\left( 2 \pi \right)^{\frac{D}{2}} 
          \begin{vmatrix}
          \Sigma 
          \end{vmatrix}^{\frac{1}{2}}} 
          \exp\left( 
         -\frac{1}{2}
          \left( x - \mu \right)^{T} \Sigma^{-1} \left( x - \mu \right)
          \right)
          ''')  # 2D 正态分布公式

#%% 定义网格和协方差矩阵

# 定义 x1 和 x2 的值域
x1 = np.linspace(-3, 3, 101)  # 在 [-3, 3] 上生成 101 个点
x2 = np.linspace(-3, 3, 101)  # 同样生成 x2 的点
xx1, xx2 = np.meshgrid(x1, x2)  # 创建网格点
pos = np.dstack((xx1, xx2))  # 将网格点堆叠为多维数组

# 定义协方差矩阵
Sigma = [[sigma_1**2, rho_12 * sigma_1 * sigma_2],  # 第一行
         [rho_12 * sigma_1 * sigma_2, sigma_2**2]]  # 第二行

# 创建多元正态分布对象
rv = multivariate_normal([0, 0], Sigma)  # 均值为 [0, 0]，协方差矩阵为 Sigma
PDF_zz = rv.pdf(pos)  # 计算网格点上的概率密度值

#%%

# 将协方差矩阵转换为 NumPy 数组
Sigma = np.array(Sigma)

# 计算协方差矩阵的特征值和特征向量
D, V = np.linalg.eig(Sigma)  # 特征分解
D = np.diag(D)  # 将特征值转化为对角矩阵

# 显示协方差矩阵和分解结果的 LaTeX 表示
st.latex(r'''\Sigma = \begin{bmatrix}%s & %s\\%s & %s\end{bmatrix}''' 
         % (sigma_1**2, 
            rho_12 * sigma_1 * sigma_2, 
            rho_12 * sigma_1 * sigma_2, 
            sigma_2**2))  # 显示协方差矩阵
st.latex(r'''\Sigma = V \Lambda V^{T}''')  # 显示特征分解公式
st.latex(bmatrix(Sigma) + '=' + 
         bmatrix(np.around(V, decimals=3)) + '@' + 
         bmatrix(np.around(D, decimals=3)) + '@' + 
         bmatrix(np.around(V.T, decimals=3)))  # 显示分解的详细过程

#%% 绘制 3D 表面图

# 创建 3D 表面图
fig_surface = go.Figure(go.Surface(
    x=x1,  # x 轴为 x1 的值
    y=x2,  # y 轴为 x2 的值
    z=PDF_zz,  # z 轴为概率密度值
    colorscale='RdYlBu_r'))  # 使用红黄蓝颜色映射
fig_surface.update_layout(
    autosize=False,  # 禁用自动调整尺寸
    width=500,  # 图表宽度
    height=500)  # 图表高度
st.plotly_chart(fig_surface)  # 在 Streamlit 页面上显示 3D 表面图

#%% 绘制 2D 等高线图

# 创建 2D 等高线图
fig_contour = go.Figure(
    go.Contour(
        z=PDF_zz,  # 等高线图的高度值
        x=x1,  # x 轴为 x1 的值
        y=x2,  # y 轴为 x2 的值
        colorscale='RdYlBu_r'  # 使用红黄蓝颜色映射
    ))

# 设置等高线图的布局
fig_contour.update_layout(
    autosize=False,  # 禁用自动调整尺寸
    width=500,  # 图表宽度
    height=500)  # 图表高度

# 在 Streamlit 页面上显示 2D 等高线图
st.plotly_chart(fig_contour)
  
        
        
        
    