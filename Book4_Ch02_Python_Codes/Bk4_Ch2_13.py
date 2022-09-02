
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

# Bk4_Ch2_13.py

import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns

def plot_heatmap(x,title):

    fig, ax = plt.subplots()
    ax = sns.heatmap(x,
                     cmap='RdYlBu_r',
                     cbar_kws={"orientation": "horizontal"}, vmin=-1, vmax=1)
    ax.set_aspect("equal")
    plt.title(title)

a = np.array([[0.5],[-0.7],[1],[0.25],[-0.6],[-1]])
b = np.array([[-0.8],[0.5],[-0.6],[0.9]])

a_outer_b = np.outer(a, b)
a_outer_a = np.outer(a, a)
b_outer_b = np.outer(b, b)

# Visualizations
plot_heatmap(a,'a')

plot_heatmap(b,'b')

plot_heatmap(a_outer_b,'a outer b')

plot_heatmap(a_outer_a,'a outer a')

plot_heatmap(b_outer_b,'b outer b')
