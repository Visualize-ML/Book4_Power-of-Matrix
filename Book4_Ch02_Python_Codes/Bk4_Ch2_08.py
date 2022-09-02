
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

# Bk4_Ch2_08.py

import numpy as np
a = np.array([[1,2],
              [3,4]])

b = np.array([[3,4],
              [5,6]])

a_dot_b = np.vdot(a,b)
# [1,2,3,4]*[3,4,5,6].T
