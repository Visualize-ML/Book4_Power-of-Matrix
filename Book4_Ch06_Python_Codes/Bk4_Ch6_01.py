
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############


# Bk4_Ch6_01.py

import numpy as np

A = np.array([[1, 2, 3, 0,  0],
              [4, 5, 6, 0,  0],
              [0, 0, 0, -1, 0],
              [0, 0 ,0, 0,  1]])

# NumPy array slicing

A_1_1 = A[0:2,0:3]

A_1_2 = A[0:2,3:]
# A_1_2 = A[0:2,-2:]
A_2_1 = A[2:,0:3]
# A_2_1 = A[-2:,0:3]
A_2_2 = A[2:,3:]
# A_2_2 = A[-2:,-2:]

# Assemble a matrix from nested lists of blocks

A_ = np.block([[A_1_1, A_1_2],
               [A_2_1, A_2_2]])
