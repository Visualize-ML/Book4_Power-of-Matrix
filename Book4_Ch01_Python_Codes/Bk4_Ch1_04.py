
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

# Bk4_Ch1_04.py

import numpy as np

# define two column vectors
a = np.array([[-2], [5]])
b = np.array([[5], [-1]])

# calculate vector addition
a_plus_b = a + b
a_plus_b_2 = np.add(a,b)

# calculate vector subtraction
a_minus_b = a - b
a_minus_b_2 = np.subtract(a,b)

b_minus_a = b - a
b_minus_a_2 = np.subtract(b,a)
