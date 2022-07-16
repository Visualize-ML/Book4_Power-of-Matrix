
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

# Bk4_Ch12_02.py

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math as m

cos_theta_12 = np.cos(m.radians(135))
cos_theta_13 = np.cos(m.radians(60))
cos_theta_23 = np.cos(m.radians(120))

P = np.array([[1, cos_theta_12, cos_theta_13],
              [cos_theta_12, 1, cos_theta_23],
              [cos_theta_13, cos_theta_23, 1]])

L = np.linalg.cholesky(P)
R = L.T


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

plt.plot(0,0,0,color = 'r', marker = 'x',
         markersize = 12)

colors = ['b', 'r', 'g']
for i in np.arange(0,3):
    
    vector = R[:,i]
    v = np.array([vector[0],vector[1],vector[2]])
    vlength=np.linalg.norm(v)
    ax.quiver(0,0,0,vector[0],vector[1],vector[2],
            length=vlength, color = colors[i])

ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])
ax.view_init(35, 60)
ax.set_proj_type('ortho')
ax.set_box_aspect([1,1,1])
