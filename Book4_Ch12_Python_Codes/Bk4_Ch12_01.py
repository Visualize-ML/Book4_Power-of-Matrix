
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

# Bk4_Ch12_01.py

import numpy as np
from matplotlib import pyplot as plt

x1 = np.arange(-2,2,0.05)
x2 = np.arange(-2,2,0.05)

xx1_fine, xx2_fine = np.meshgrid(x1,x2)

a = 0; b = -1; c = 0;

yy_fine = a*xx1_fine**2 + 2*b*xx1_fine*xx2_fine + c*xx2_fine**2

# 3D visualization

fig, ax = plt.subplots()
ax = plt.axes(projection='3d')

ax.plot_wireframe(xx1_fine,xx2_fine,yy_fine,
                  color = [0.8,0.8,0.8],
                  linewidth = 0.25)

ax.contour3D(xx1_fine,xx2_fine,yy_fine,15,
             cmap = 'RdYlBu_r')

ax.view_init(elev=30, azim=60)
ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
ax.zaxis.set_ticks([])
plt.tight_layout()
ax.set_proj_type('ortho')
plt.show()

# 2D visualization

fig, ax = plt.subplots()

ax.contourf(xx1_fine,xx2_fine,yy_fine,15,
           cmap = 'RdYlBu_r')

ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
ax.set_aspect('equal')

plt.show()
