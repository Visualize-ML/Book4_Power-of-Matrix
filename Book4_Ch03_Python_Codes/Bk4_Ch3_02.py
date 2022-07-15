
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

# Bk4_Ch3_02.py

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

u = [0,0,4, 3]
v = [0,0,-2,4]
u_bis = [4,3,v[2],v[3]]
w = [0,0,2,7]

fig, ax = plt.subplots()

plt.quiver([u[0], u_bis[0], w[0]],
           [u[1], u_bis[1], w[1]],
           [u[2], u_bis[2], w[2]],
           [u[3], u_bis[3], w[3]],
           angles='xy', scale_units='xy', 
           scale=1, color=sns.color_palette())

plt.axvline(x=0, color='grey')
plt.axhline(y=0, color='grey')

plt.text(3, 1, r'$||\vec{u}||_2$', 
         color=sns.color_palette()[0], size=12,
         ha='center',va='center')

plt.text(3, 6, r'$||\vec{v}||_2$', 
         color=sns.color_palette()[1], size=12,
         ha='center',va='center')

plt.text(0, 4, r'$||\vec{u}+\vec{v}||_2$', 
         color=sns.color_palette()[2], size=12,
         ha='center',va='center')

plt.ylabel('$x_2$')
plt.xlabel('$x_1$')
plt.axis('scaled')
ax.set_xticks(np.arange(-2,8 + 1))
ax.set_yticks(np.arange(-2,8 + 1))
ax.set_xlim(-2, 8)
ax.set_ylim(-2, 8)
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])

# reference: Essential Math for Data Science
