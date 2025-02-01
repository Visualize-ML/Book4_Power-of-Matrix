
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2025
###############

## 导入必要的库
import streamlit as st  # 引入Streamlit库，用于创建交互式Web应用
import plotly.express as px  # 引入Plotly Express库，用于绘图

## 加载鸢尾花数据
df = px.data.iris()  # 使用Plotly自带的数据集加载鸢尾花数据

features = df.columns.to_list()[:-2]  # 获取数据集中特征列的名称（排除最后两列）
with st.sidebar:  # 创建侧边栏
    st.write('2D scatter plot')  # 显示文本“2D散点图”
    x_feature = st.radio('Horizontal axis',  # 创建单选按钮，选择X轴的特征
                         features)  # 单选按钮的选项为特征列名称
    y_feature = st.radio('Vertical axis',  # 创建单选按钮，选择Y轴的特征
                         features)  # 单选按钮的选项为特征列名称

## 原始数据展示
with st.expander('Original data'):  # 创建可展开的部分，标题为“原始数据”
    st.write(df)  # 显示数据框df的内容

## 热力图展示
with st.expander('Heatmap'):  # 创建可展开的部分，标题为“热力图”
    fig_1 = px.imshow(df.iloc[:, 0:4],  # 使用前四列数据绘制热力图
                      color_continuous_scale='RdYlBu_r')  # 选择颜色映射为红黄蓝反转
    st.plotly_chart(fig_1)  # 在Streamlit中显示热力图

## 二维散点图展示
with st.expander('2D scatter plot'):  # 创建可展开的部分，标题为“二维散点图”
    fig_2 = px.scatter(df, x=x_feature, y=y_feature, color="species")  # 根据用户选择的特征绘制二维散点图，按种类着色
    st.plotly_chart(fig_2)  # 在Streamlit中显示散点图

## 三维散点图展示
with st.expander('3D scatter plot'):  # 创建可展开的部分，标题为“三维散点图”
    fig_3 = px.scatter_3d(df,  # 绘制三维散点图
                          x='sepal_length',  # X轴为花萼长度
                          y='sepal_width',  # Y轴为花萼宽度
                          z='petal_width',  # Z轴为花瓣宽度
                          color='species')  # 按种类着色
    st.plotly_chart(fig_3)  # 在Streamlit中显示三维散点图

## 配对散点图展示
with st.expander('Pairwise scatter plot'):  # 创建可展开的部分，标题为“配对散点图”
    fig_4 = px.scatter_matrix(df,  # 绘制配对散点图
                              dimensions=["sepal_width", 
                                          "sepal_length", 
                                          "petal_width", 
                                          "petal_length"],  # 指定绘图的维度
                              color="species")  # 按种类着色
    st.plotly_chart(fig_4)  # 在Streamlit中显示配对散点图

    
    
    
