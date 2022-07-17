
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

# Bk4_Ch14_02.py

import numpy as np
import matplotlib.pyplot as plt

# transition matrix
T = np.matrix([[0.7, 0.2],
               [0.3, 0.8]])

# steady state
sstate = np.linalg.eig(T)[1][:,1]
sstate = sstate/sstate.sum()
print(sstate)

# initial states
initial_x_array = np.array([[1, 0, 0.5, 0.4],  # Chicken
                            [0, 1, 0.5, 0.6]]) # Rabbit

num_iterations = 10;

for i in np.arange(0,4):
    
    initial_x = initial_x_array[:,i][:, None]
    
    x_i = np.zeros_like(initial_x)
    x_i = initial_x
    X =   initial_x.T;
    
    # matrix power through iterations
    
    for x in np.arange(0,num_iterations):
        x_i = T@x_i;
        X = np.concatenate([X, x_i.T],axis = 0)
    
    fig, ax = plt.subplots()
    
    itr = np.arange(0,num_iterations+1);
    plt.plot(itr,X[:,0],marker = 'x',color = (1,0,0))
    plt.plot(itr,X[:,1],marker = 'x',color = (0,0.6,1))
    
    ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
    ax.set_xlim(0, num_iterations)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Iteration, k')
    ax.set_ylabel('State')
