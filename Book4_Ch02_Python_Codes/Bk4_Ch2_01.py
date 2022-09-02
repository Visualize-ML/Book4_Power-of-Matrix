
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

# Bk4_Ch2_01.py

import numpy as np
import matplotlib.pyplot as plt

def draw_vector(vector,RBG): 
    array = np.array([[0, 0, vector[0], vector[1]]])
    X, Y, U, V = zip(*array)
    plt.quiver(X, Y, U, V,angles='xy', scale_units='xy',scale=1,color = RBG)

fig, ax = plt.subplots()

draw_vector([4,3],np.array([0,112,192])/255)
draw_vector([-3,4],np.array([255,0,0])/255)

plt.ylabel('$x_2$')
plt.xlabel('$x_1$')
plt.axis('scaled')
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
plt.show()
