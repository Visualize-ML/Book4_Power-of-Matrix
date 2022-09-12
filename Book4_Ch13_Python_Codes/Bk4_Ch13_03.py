
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

# Bk4_Ch13_03.py

import numpy as np
import matplotlib.pyplot as plt

theta = np.deg2rad(30)

r = 0.8 # 1.2, scaling factor

R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

S = np.array([[r, 0],
              [0, r]])

A = R@S

# A = np.array([[1, -1],
#               [1, 1]])

Lamb, V = np.linalg.eig(A)

theta_array = np.arange(0,np.pi*2,np.pi*2/18)


colors = plt.cm.rainbow(np.linspace(0,1,len(theta_array)))


fig, ax = plt.subplots()

for j, theat_i in enumerate(theta_array):
    
    # initial point
    x = np.array([[5*np.cos(theat_i)],
                  [5*np.sin(theat_i)]])

    plt.plot(x[0],x[1], 
             marker = 'x',color = colors_j,
             markersize = 15)
    # plot the initial point
    
    x_array = x

    for i in np.arange(20):

        x = A@x
        x_array = np.column_stack((x_array,x))


    colors_j = colors[j,:]
    plt.plot(x_array[0,:],x_array[1,:], 
             marker = '.',color = colors_j)


plt.axis('scaled')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.axvline(x=0,color = 'k')
ax.axhline(y=0,color = 'k')
