
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

# Bk4_Ch2_11.py

import numpy as np
a = np.array([-2, 1, 1])
b = np.array([1, -2, -1])
# a = [-2, 1, 1]
# b = [1, -2, -1]

# calculate cross product of row vectors
a_cross_b = np.cross(a, b)

a_col = np.array([[-2], [1], [1]])
b_col = np.array([[1], [-2], [-1]])

# calculate cross product of column vectors
a_cross_b_col = np.cross(a_col,b_col,axis=0)
