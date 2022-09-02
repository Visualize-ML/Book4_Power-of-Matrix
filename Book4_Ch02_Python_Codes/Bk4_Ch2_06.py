
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

# Bk4_Ch2_06.py

import numpy as np

a = np.array([[4, 3]])
b = np.array([[5, -2]])

a_dot_b = np.inner(a, b)

a_2 = np.array([[4], [3]])
b_2 = np.array([[5], [-2]])
a_dot_b_2 = a_2.T@b_2
