
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

# Bk4_Ch15_01.py

import numpy as np
import matplotlib.pyplot as plt

def visualize(X_circle,X_vec,title_txt):
    
    fig, ax = plt.subplots()
    
    plt.plot(X_circle[:,0], X_circle[:,1],'k',
             linestyle = '--',
             linewidth = 0.5)
    
    plt.quiver(0,0,X_vec[0,0],X_vec[0,1],
              angles='xy', scale_units='xy',scale=1, 
              color = [0, 0.4392, 0.7529])
    
    plt.quiver(0,0,X_vec[1,0],X_vec[1,1],
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

X_vec = np.array([[1,0],
                 [0,1]])

X_circle = np.array([circle_x1, circle_x2]).T

# plot original circle and two vectors
visualize(X_circle,X_vec,'Original')

A = np.array([[1.6250, 0.6495],
              [0.6495, 0.8750]])

# plot the transformation of A

visualize(X_circle@A.T, X_vec@A.T,'$A$')


#%% SVD
# A = U @ S @ V.T

U, S, V = np.linalg.svd(A)
S = np.diag(S)
V[:,0] = -V[:,0] # reverse sign of first vector of V
U[:,0] = -U[:,0] # reverse sign of first vector of U

print('=== U ===')
print(U)
print('=== S ===')
print(S)
print('=== V ===')
print(V)

# plot the transformation of V

visualize(X_circle@V, X_vec@V,'$V^T$')

# plot the transformation of V @ S

visualize(X_circle@V@S, X_vec@V@S,'$SV^T$')

# plot the transformation of V @ S @ U.T

visualize(X_circle@V@S@U.T, X_vec@V@S@U.T,'$USV^T$')

e1 = np.array([[1],
               [0]])

e2 = np.array([[0],
               [1]])

# Calculate step by step from e1 and e2
VT_e1 = V.T@e1
VT_e2 = V.T@e2

S_VT_e1 = S@VT_e1
S_VT_e2 = S@VT_e2

U_S_VT_e1 = U@S_VT_e1
U_S_VT_e2 = U@S_VT_e2
