
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

# Bk4_Ch9_01.py

import numpy as np
import matplotlib.pyplot as plt

thetas = np.linspace(0,np.pi,25)

x = np.array([[4],
              [3]])

fig, axes = plt.subplots()

for theta in thetas:
    
    v = np.array([[np.cos(theta)], 
                  [np.sin(theta)]])
    
    proj = v.T@x
    print(proj)
    plt.plot([-v[0]*6, v[0]*6], [-v[1]*6, v[1]*6])
    plt.plot([x[0], v[0]*proj], [x[1], v[1]*proj], color = 'k')
    plt.plot(v[0]*proj, v[1]*proj, color = 'k', marker = 'x')
    
    plt.quiver (0, 0, v[0], v[1],
                angles='xy', scale_units='xy',scale=1)

plt.plot(x[0],x[1], marker = 'x', color = 'r')
plt.axis('scaled')
