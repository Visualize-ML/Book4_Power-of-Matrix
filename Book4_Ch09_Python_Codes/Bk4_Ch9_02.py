
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

# Bk4_Ch9_02.py

import numpy as np
import matplotlib.pyplot as plt

thetas = np.array([0, 15, 30, 45, 60, 75, 90, 120, 135])

x = np.array([[4],
              [3]])

i = 1
fig = plt.figure()

for theta in thetas:
    
    theta = theta/180*np.pi
    ax = fig.add_subplot(3, 3, i)

    v1 = np.array([[np.cos(theta)], 
                   [np.sin(theta)]])

    proj = v1.T@x
    print(proj)
    plt.plot([-v1[0]*6, v1[0]*6], [-v1[1]*6, v1[1]*6])
    plt.plot([x[0], v1[0]*proj], [x[1], v1[1]*proj], 
             color = 'k', linestyle = '--')
    plt.plot(v1[0]*proj, v1[1]*proj, color = 'k', marker = 'x')
    
    plt.quiver (0, 0, v1[0], v1[1],
                angles='xy', scale_units='xy',
                scale=1, color = 'b')

    v2 = np.array([[-np.sin(theta)], 
                  [np.cos(theta)]])
    
    proj = v2.T@x
    print(proj)
    plt.plot([-v2[0]*6, v2[0]*6], [-v2[1]*6, v2[1]*6])
    plt.plot([x[0], v2[0]*proj], [x[1], v2[1]*proj], 
             color = 'k', linestyle = '--')
    plt.plot(v2[0]*proj, v2[1]*proj, color = 'k', marker = 'x')
    
    plt.quiver (0, 0, v2[0], v2[1],
                angles='xy', scale_units='xy',
                scale=1,color = 'r')
    
    plt.axhline(y = 0, color = 'k')
    plt.axvline(x = 0, color = 'k')
    plt.plot(x[0],x[1], marker = 'x', color = 'r')
    plt.quiver(0, 0, x[0],x[1], 
               angles='xy', scale_units='xy',
               scale=1, color = 'k')
    
    plt.axis('scaled')
    ax.grid(linestyle='--', linewidth=0.25, color=[0.75,0.75,0.75])
    plt.xlim([-6, 6])
    plt.ylim([-6, 6])
    plt.xticks(np.linspace(-6,6,13))
    plt.yticks(np.linspace(-6,6,13))
    
    i = i + 1
