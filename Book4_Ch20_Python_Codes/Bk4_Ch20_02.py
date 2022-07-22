
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

# Bk4_Ch20_02.py

import numpy as np
import matplotlib.pyplot as plt

alphas = np.linspace(0, 2*np.pi, 100)

# unit circle
r = np.sqrt(1.0)

z1 = r*1/np.cos(alphas)
z2 = r*np.tan(alphas)

Z = np.array([z1, z2]).T # data of unit circle

# scale
S = np.array([[1, 0],
              [0, 1]])

thetas = np.array([0, 30, 45, 60, 90, 120])

for theta in thetas:

    # rotate
    print('==== Rotate ====')
    print(theta)
    theta = theta/180*np.pi
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])

    X = Z@S@R.T;
    
    x1 = X[:,0]
    x2 = X[:,1]
    
    fig, ax = plt.subplots(1)
    ax.plot(z1, z2, 'b') # plot the unit circle
    ax.plot(x1, x2, 'r') # plot the transformed shape

    plt.axvline(x=0, color= 'k', zorder=0)
    plt.axhline(y=0, color= 'k', zorder=0)
    ax.set_aspect(1)
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    plt.grid(linestyle='--')
    plt.show()
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
