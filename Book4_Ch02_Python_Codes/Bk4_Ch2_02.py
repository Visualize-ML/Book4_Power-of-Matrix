
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

# Bk4_Ch2_02.py

import numpy as np

# define two column vectors
a = np.array([[4], [3]])
b = np.array([[-3], [4]])

# calculate L2 norm
a_L2_norm = np.linalg.norm(a)
b_L2_norm = np.linalg.norm(b)
