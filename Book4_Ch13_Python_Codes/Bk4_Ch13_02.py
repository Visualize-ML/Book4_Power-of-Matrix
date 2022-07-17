
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

# Bk4_Ch13_02.py

import numpy as np
import matplotlib.pyplot as plt

def visualize(X_circle,X_vec,title_txt):
    
    fig, ax = plt.subplots()
    
    plt.plot(X_circle[0,:], X_circle[1,:],'k',
             linestyle = '--',
             linewidth = 0.5)
    
    plt.quiver(0,0,X_vec[0,0],X_vec[1,0],
              angles='xy', scale_units='xy',scale=1, 
              color = [0, 0.4392, 0.7529])
    
    plt.quiver(0,0,X_vec[0,1],X_vec[1,1],
              angles='xy', scale_units='xy',scale=1, 
              color = [1,0,0])
    
    plt.axvline(x=0, color= 'k', zorder=0)
    plt.axhline(y=0, color= 'k', zorder=0)
    
    plt.ylabel('$x_2$')
    plt.xlabel('$x_1$')
    
    ax.set_aspect(1)
    ax.set_xlim([-2.5, 2.5])
    ax.set_ylim([-2.5, 2.5])
    ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
    ax.set_xticks(np.linspace(-2,2,5));
    ax.set_yticks(np.linspace(-2,2,5));
    plt.title(title_txt)
    plt.show()


theta = np.linspace(0, 2*np.pi, 100)

circle_x1 = np.cos(theta)
circle_x2 = np.sin(theta)

V_vec = np.array([[np.sqrt(2)/2, -np.sqrt(2)/2],
                  [np.sqrt(2)/2,  np.sqrt(2)/2]])

X_circle = np.array([circle_x1, circle_x2])

# plot original circle and two vectors
visualize(X_circle,V_vec,'Original')

A = np.array([[1.25, -0.75],
              [-0.75, 1.25]])

# plot the transformation of A

visualize(A@X_circle, A@V_vec,'$A$')


#%% Eigen deomposition

# A = V @ D @ V.T

lambdas, V = np.linalg.eig(A)

D = np.diag(np.flip(lambdas))
V = V.T # reverse the order

print('=== LAMBDA ===')
print(D)
print('=== V ===')
print(V)

# plot the transformation of V.T

visualize(V.T@X_circle, V.T@V_vec,'$V^T$')

# plot the transformation of D @ V.T

visualize(D@V.T@X_circle, D@V.T@V_vec,'$\u039BV^T$')

# plot the transformation of V @ D @ V.T

visualize(V@D@V.T@X_circle, V@D@V.T@V_vec,'$V\u039BV^T$')

# plot the transformation of A

visualize(A@X_circle, A@V_vec,'$A$')

