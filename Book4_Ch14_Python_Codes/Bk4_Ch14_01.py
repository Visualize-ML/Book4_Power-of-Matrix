
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

# Bk4_Ch14_01.py

import numpy as np

A = np.matrix([[1.25, -0.75],
               [-0.75, 1.25]])

LAMBDA, V = np.linalg.eig(A)

B = V@np.diag(np.sqrt(LAMBDA))@np.linalg.inv(V)

A_reproduced = B@B

print(A_reproduced)
