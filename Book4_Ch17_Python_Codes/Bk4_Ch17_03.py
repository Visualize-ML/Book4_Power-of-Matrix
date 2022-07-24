
###############
# Authored by Weisheng Jiang
# Book 4  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

# Bk4_Ch17_03.py

import sympy
from sympy import Matrix, Transpose
import numpy as np
from sympy.functions import exp
import matplotlib.pyplot as plt


def mesh_circ(c1, c2, r, num):
    
    theta = np.arange(0,2*np.pi+np.pi/num,np.pi/num)
    r     = np.arange(0,r,r/num)
    theta,r = np.meshgrid(theta,r)
    xx1 = np.cos(theta)*r + c1
    xx2 = np.sin(theta)*r + c2
    
    return xx1, xx2


#define symbolic vars, function
x1,x2,p1,p2 = sympy.symbols('x1 x2 p1 p2')

f_x = -4*x1**2 -4*x2**2
f_p = -4*p1**2 -4*p2**2

print(f_x)

#take the gradient symbolically
grad_f = [sympy.diff(f_p,var) for var in (p1,p2)]
print(grad_f)

f_x_fcn = sympy.lambdify([x1,x2],f_x)

t_x = Matrix(grad_f).T*Matrix([[x1 - p1], [x2 - p2]]) + Matrix([f_p])
print(t_x)

t_x_fcn = sympy.lambdify([x1,x2,p1,p2],t_x)

#turn into a bivariate lambda for numpy
grad_fcn = sympy.lambdify([x1,x2],grad_f)

xx1, xx2 = mesh_circ(0, 0, 3, 20)

# expansion point
p1 = -1.5
p2 = 0
py = f_x_fcn(p1,p2)

# coarse mesh
xx1_, xx2_ = np.meshgrid(np.linspace(p1-1, p1+1, 10),
                         np.linspace(p2-1, p2+1, 10))

# quadratic surface
ff_x = f_x_fcn(xx1,xx2)

# tangent plane
tt_x = t_x_fcn(xx1_, xx2_, p1, p2)

# 3D visualization
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_wireframe(xx1, xx2, ff_x, rstride=1, 
                  cstride=1, color = [0.5,0.5,0.5],
                  linewidth = 0.2)

ax.plot_wireframe(xx1_, xx2_, np.squeeze(tt_x), rstride=1, 
                  cstride=1, color = [1,0,0],
                  linewidth = 0.2)
ax.plot(p1,p2,py,'x')

ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
ax.zaxis.set_ticks([])
plt.xlim(-3,3)
plt.ylim(-3,3)
ax.view_init(30, -125)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$f(x_1,x_2)$')
