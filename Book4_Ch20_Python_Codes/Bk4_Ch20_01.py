
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

# Bk4_Ch20_01.py

import numpy as np
import matplotlib.pyplot as plt

alphas = np.linspace(0, 2*np.pi, 100)

# unit circle
r = np.sqrt(1.0)

z1 = r*np.cos(alphas)
z2 = r*np.sin(alphas)

Z = np.array([z1, z2]).T # data of unit circle

# scale
S = np.array([[2, 0],
              [0, 0.5]])

thetas = np.array([0, 30, 45, 60, 90, 120])

for theta in thetas:

    # rotate
    print('==== Rotate ====')
    print(theta)
    theta = theta/180*np.pi
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    
    # translate
    c = np.array([2, 1])
    X = Z@S@R.T + c;
    
    Q = R@np.linalg.inv(S)@np.linalg.inv(S)@R.T
    print('==== Q ====')
    print(Q)
    LAMBDA, V = np.linalg.eig(Q)
    print('==== LAMBDA ====')
    print(LAMBDA)
    print('==== V ====')
    print(V)
    
    x1 = X[:,0]
    x2 = X[:,1]
    
    fig, ax = plt.subplots(1)
    ax.plot(z1, z2, 'b') # plot the unit circle
    ax.plot(x1, x2, 'r') # plot the transformed shape
    ax.plot(c[0],c[1],'xk') # plot the center
    
    ax.quiver(0,0,1,0,color = 'b',angles='xy', scale_units='xy',scale=1)
    ax.quiver(0,0,0,1,color = 'b',angles='xy', scale_units='xy',scale=1)
    ax.quiver(0,0,-1,0,color = 'b',angles='xy', scale_units='xy',scale=1)
    ax.quiver(0,0,0,-1,color = 'b',angles='xy', scale_units='xy',scale=1)
    
    ax.quiver(0,0,c[0],c[1],color = 'k',angles='xy', scale_units='xy',scale=1)
    
    ax.quiver(c[0],c[1],
              V[0,0]/np.sqrt(LAMBDA[0]),
              V[1,0]/np.sqrt(LAMBDA[0]),color = 'r',
              angles='xy', scale_units='xy',scale=1)
    
    ax.quiver(c[0],c[1],
              V[0,1]/np.sqrt(LAMBDA[1]),
              V[1,1]/np.sqrt(LAMBDA[1]),color = 'r',
              angles='xy', scale_units='xy',scale=1)

    ax.quiver(c[0],c[1],
              -V[0,0]/np.sqrt(LAMBDA[0]),
              -V[1,0]/np.sqrt(LAMBDA[0]),color = 'r',
              angles='xy', scale_units='xy',scale=1)
    
    ax.quiver(c[0],c[1],
              -V[0,1]/np.sqrt(LAMBDA[1]),
              -V[1,1]/np.sqrt(LAMBDA[1]),color = 'r',
              angles='xy', scale_units='xy',scale=1)
    
    plt.axvline(x=0, color= 'k', zorder=0)
    plt.axhline(y=0, color= 'k', zorder=0)
    
    
    ax.set_aspect(1)
    plt.xlim(-2,4)
    plt.ylim(-2,4)
    plt.grid(linestyle='--')
    plt.show()
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
