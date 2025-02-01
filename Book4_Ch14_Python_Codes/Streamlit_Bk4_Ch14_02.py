
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2025
###############

import numpy as np  # 导入NumPy库，用于数值计算
import streamlit as st  # 导入Streamlit库，用于创建交互式Web应用
import time  # 导入time模块，用于添加延时效果

# 定义状态转移矩阵 A
A = np.matrix([[0.7, 0.2],  # 第一行表示从鸡到鸡的概率为0.7，从鸡到兔子的概率为0.3
               [0.3, 0.8]])  # 第二行表示从兔子到鸡的概率为0.2，从兔子到兔子的概率为0.8

# 在侧边栏中创建交互组件
with st.sidebar:
    # 创建滑块，用于用户设置初始状态中鸡的比例
    pi_0_chicken = st.slider('Ratio of chicken:',  # 滑块标题
                             0.0, 1.0, step=0.1)  # 范围从0到1，每次步进0.1
    pi_0_rabbit = 1 - pi_0_chicken  # 计算兔子的比例，使鸡和兔子的比例和为1
    st.write('Ratio of rabbit: ' + str(round(pi_0_rabbit, 1)))  # 显示兔子的比例，保留1位小数

    # 创建滑块，用于用户设置模拟的天数
    num_iterations = st.slider('Number of nights:',  # 滑块标题
                                20, 100, step=5)  # 范围从20到100，每次步进5

# 创建进度条和状态文本，用于显示迭代进度
progress_bar = st.sidebar.progress(0)  # 初始化进度条为0%
status_text = st.sidebar.empty()  # 创建一个空白的文本区域，用于显示进度百分比

# 初始化状态向量，将用户输入的初始比例存入数组
last_rows = np.array([[pi_0_chicken, pi_0_rabbit]])  # 初始状态为一个行向量，包含鸡和兔子的比例

# 创建一个折线图，用于实时展示状态变化
chart = st.line_chart(last_rows)  # 初始化折线图，并将初始状态绘制到图上

# 开始迭代模拟状态转移
for i in range(1, num_iterations):  # 循环从第1天到用户设置的总天数
    last_status = last_rows[-1, :]  # 获取当前的状态向量（最后一行）
    new_rows = last_status @ A.T  # 使用矩阵乘法计算下一个状态，转移矩阵取转置
    percent = (i + 1) * 100 / num_iterations  # 计算当前完成的百分比

    # 更新进度条和状态文本
    status_text.text("%i%% Complete" % percent)  # 显示当前完成的百分比
    chart.add_rows(new_rows)  # 将新状态添加到折线图
    progress_bar.progress(i)  # 更新进度条的值
    last_rows = new_rows  # 更新最后的状态向量为当前计算的状态
    time.sleep(0.1)  # 延时0.1秒，以便观察每次状态更新

# 清空进度条，表示模拟完成
progress_bar.empty()  # 移除进度条

