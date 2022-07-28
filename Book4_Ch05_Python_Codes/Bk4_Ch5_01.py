
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

# Bk4_Ch5_01.py

import numpy as np

a = np.array([[1],
              [2],
              [3]])

a_1D = np.array([1,2,3])

b = np.array([[-4],
              [-5],
              [-6]])

b_1D = np.array([-4,-5,-6])

#%% sum of the elements in a

print(np.einsum('ij->',a))
print(np.einsum('i->',a_1D))

#%% element-wise multiplication of a and b

print(np.einsum('ij,ij->ij',a,b))
print(np.einsum('i,i->i',a_1D,b_1D))

#%% inner product of a and b

print(np.einsum('ij,ij->',a,b))
print(np.einsum('i,i->',a_1D,b_1D))

#%% outer product of a and itself

print(np.einsum('ij,ji->ij',a,a))
print(np.einsum('i,j->ij',a_1D,a_1D))


#%% outer product of a and b

print(np.einsum('ij,ji->ij',a,b))
print(np.einsum('i,j->ij',a_1D,b_1D))

#%%

# A is a square matrix
A = np.array([[1,2,3],
              [4,5,6],
              [7,8,9]])

B = np.array([[-1,-4,-7],
              [-2,-5,-8],
              [-3,-6,-9]])

#%% transpose of A

print(np.einsum('ji',A))

#%% sum of all values in A

print(np.einsum('ij->',A))

#%% sum across rows

print(np.einsum('ij->j',A))

#%% sum across columns

print(np.einsum('ij->i',A))

#%% extract main diagonal of A

print(np.einsum('ii->i',A))

#%% calculate the trace of A

print(np.einsum('ii->',A))

#%% matrix multiplication of A and B

print(np.einsum('ij,jk->ik', A, B))

#%% sum of all elements in the matrix multiplication of A and B

print(np.einsum('ij,jk->', A, B))

#%% first matrix multiplication, then transpose

print(np.einsum('ij,jk->ki', A, B))

#%% element-wise multiplication of A and B

print(np.einsum('ij,ij->ij', A, B))
