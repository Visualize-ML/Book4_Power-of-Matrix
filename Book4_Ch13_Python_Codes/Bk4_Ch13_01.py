
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

# Bk4_Ch13_01.py

import numpy as np
import matplotlib.pyplot as plt

A = np.array([[1.25,  -0.75],
              [-0.75, 1.25]])

xx1, xx2 = np.meshgrid(np.linspace(-8, 8, 9), np.linspace(-8, 8, 9))
num_vecs = np.prod(xx1.shape);

thetas = np.linspace(0, 2*np.pi, num_vecs)

thetas = np.reshape(thetas, (-1, 9))
thetas = np.flipud(thetas);

uu = np.cos(thetas);
vv = np.sin(thetas);

fig, ax = plt.subplots()

ax.quiver(xx1,xx2,uu,vv,
          angles='xy', scale_units='xy',scale=1, 
          edgecolor='none', facecolor= 'b')

plt.ylabel('$x_2$')
plt.xlabel('$x_1$')
plt.axis('scaled')
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
ax.set_xticks(np.linspace(-10,10,11));
ax.set_yticks(np.linspace(-10,10,11));
plt.show()

# Matrix multiplication
V = np.array([uu.flatten(),vv.flatten()]).T;
W = V@A;

uu_new = np.reshape(W[:,0],(-1, 9));
vv_new = np.reshape(W[:,1],(-1, 9));

fig, ax = plt.subplots()

ax.quiver(xx1,xx2,uu,vv,
          angles='xy', scale_units='xy',scale=1, 
          edgecolor='none', facecolor= 'b')

ax.quiver(xx1,xx2,uu_new,vv_new,
          angles='xy', scale_units='xy',scale=1, 
          edgecolor='none', facecolor= 'r')

plt.ylabel('$x_2$')
plt.xlabel('$x_1$')
plt.axis('scaled')
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
ax.set_xticks(np.linspace(-10,10,11));
ax.set_yticks(np.linspace(-10,10,11));
plt.show()


fig, ax = plt.subplots()
ax.quiver(xx1*0,xx2*0,uu,vv,
          angles='xy', scale_units='xy',scale=1, 
          edgecolor='none', facecolor= 'b')

ax.quiver(xx1*0,xx2*0,uu_new,vv_new,
          angles='xy', scale_units='xy',scale=1,
          edgecolor='none', facecolor= 'r')

plt.ylabel('$x_2$')
plt.xlabel('$x_1$')
plt.axis('scaled')
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
ax.set_xticks(np.linspace(-2,2,5));
ax.set_yticks(np.linspace(-2,2,5));
plt.show()
