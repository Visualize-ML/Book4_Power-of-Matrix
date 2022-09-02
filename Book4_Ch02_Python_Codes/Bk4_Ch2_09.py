
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

# Bk4_Ch2_09.py

import numpy as np

a, b = np.array([[4], [3]]), np.array([[5], [-2]])

# calculate cosine theta
cos_theta = (a.T @ b) / (np.linalg.norm(a,2) * np.linalg.norm(b,2))

# calculate theta in radian
cos_radian = np.arccos(cos_theta)

# convert radian to degree
cos_degree = cos_radian * ((180)/np.pi)
