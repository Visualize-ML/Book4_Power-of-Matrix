
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

# Bk4_Ch20_03.py

import numpy as np
import matplotlib.pyplot as plt

a = 1.5
b = 1

x1 = np.linspace(-3,3,200)
x2 = np.linspace(-3,3,200)
xx1,xx2 = np.meshgrid(x1,x2)

fig, ax = plt.subplots()

theta_array = np.linspace(0,2*np.pi,100)

plt.plot(a*np.cos(b*np.sin(theta)),b*np.sin(b*np.sin(theta)),color = 'k')

colors = plt.cm.RdYlBu(np.linspace(0,1,len(theta_array)))

for i in range(len(theta_array)):
    
    theta = theta_array[i]
    
    p1 = a*np.cos(theta)
    p2 = b*np.sin(theta)
    
    tangent = p1*xx1/a**2 + p2*xx2/b**2 - p1**2/a**2 - p2**2/b**2
    
    colors_i = colors[int(i),:]
    
    ax.contour(xx1,xx2,tangent, levels = [0], colors = [colors_i])

plt.axis('scaled')
ax.set_xlim(-3,3)
ax.set_ylim(-3,3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.axvline(x=0,color = 'k')
ax.axhline(y=0,color = 'k')
