
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

# Bk4_Ch8_01.py

import matplotlib.pyplot as plt 
import numpy as np

def plot_shape(X,copy = False):
    if copy:
        fill_color = np.array([255,236,255])/255
        edge_color = np.array([255,0,0])/255
    else:
        fill_color = np.array([219,238,243])/255
        edge_color = np.array([0,153,255])/255
    
    plt.fill(X[:,0], X[:,1],
             color = fill_color,
             edgecolor = edge_color)
    
    plt.plot(X[:,0], X[:,1],marker = 'x',
             markeredgecolor = edge_color*0.5,
             linestyle = 'None') 

X = np.array([[1,1],
              [0,-1],
              [-1,-1],
              [-1,1]])

# visualizations

fig, ax = plt.subplots()

plot_shape(X)      # plot original

# translation
t1 = np.array([3,2]);
Z = X + t1
plot_shape(Z,True) # plot copy

t2 = np.array([-3,-2]);
Z = X + t2
plot_shape(Z,True) # plot copy

t3 = np.array([-2,3]);
Z = X + t3
plot_shape(Z,True) # plot copy

t4 = np.array([3,-3]);
Z = X + t4
plot_shape(Z,True) # plot copy

# Decorations
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
plt.axis('equal')
plt.axis('square')
plt.axhline(y=0, color='k', linewidth = 0.25)
plt.axvline(x=0, color='k', linewidth = 0.25)
plt.xticks(np.arange(-5, 6))
plt.yticks(np.arange(-5, 6))
ax.set_xlim(-5,5)
ax.set_ylim(-5,5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
