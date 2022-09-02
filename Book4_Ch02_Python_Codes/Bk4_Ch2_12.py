
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

# Bk4_Ch2_12.py

import numpy as np
a = np.array([-2, 1, 1])
b = np.array([1, -2, -1])
# a = [-2, 1, 1]
# b = [1, -2, -1]


# calculate element-wise product of row vectors
a_times_b = np.multiply(a, b)
a_times_b_2 = a*b

a_col = np.array([[-2], [1], [1]])
b_col = np.array([[1], [-2], [-1]])

# calculate element-wise product of column vectors
a_times_b_col = np.multiply(a_col,b_col)
a_times_b_col_2 = a_col*b_col
