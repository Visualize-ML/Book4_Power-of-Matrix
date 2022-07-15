
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############


# Bk4_Ch6_02.py

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

# Generate matrices A and B
A = np.random.random_integers(0,40,size=(6,4))
A = A/20 - 1

B = np.random.random_integers(0,40,size=(4,3))
B = B/20 - 1

# visualize matrix A and B
plot_heatmap(A,'A')

plot_heatmap(B,'B')

# visualize A@B
C = A@B
plot_heatmap(C,'C = AB')

C_rep = np.zeros_like(C)

# reproduce C

for i in np.arange(4):
    C_i = A[:,[i]]@B[[i],:];
    title = 'C' + str(i + 1)
    plot_heatmap(C_i,title)
    
    C_rep = C_rep + C_i

# Visualize reproduced C
plot_heatmap(C_rep,'C reproduced')

