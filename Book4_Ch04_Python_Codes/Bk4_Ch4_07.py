
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

# Bk4_Ch4_07.py

import numpy as np

A = np.array([[1, 2]])

B = np.array([[5, 6], 
              [8, 9]])

print(A*B)

A = np.array([[1, 2]])

B = np.matrix([[5, 6], 
              [8, 9]])

print(A*B)

A = np.matrix([[1, 2]])

B = np.matrix([[5, 6], 
              [8, 9]])

print(A*B)
