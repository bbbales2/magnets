import numpy
import pickle
import os
import matplotlib.pyplot as plt
import scipy.interpolate

os.chdir('/home/bbales2/magnets/')

fname = 'gd'

f = open('mhi_250x250_{0}'.format(fname))
m = pickle.load(f)
f.close()

from fenics import *

# Scaled variables
H = 1.0
W = 1.0

alpha = 1.0

# Create mesh and define function space
mesh = RectangleMesh(Point(0.0, 0.0), Point(H, W), 40, 40)
V = FunctionSpace(mesh, "CG", 2)
# Define boundary condition
tol = 1E-10

def clamped_boundary(x, on_boundary):
    return on_boundary and (x[0] < tol)

bc = DirichletBC(V, Constant(0), clamped_boundary)

y = numpy.linspace(0, 1.0, m.shape[0])
x = numpy.linspace(0, 1.0, m.shape[1])

Mhatf = scipy.interpolate.interp2d(x, y, m)

class Mhat(Expression):
    def eval(self, value, x):
        value[0] = Mhatf(x[1], x[0])

mh = Mhat(element = V.ufl_element())

# Define variational problem
u = TrialFunction(V)
d = u.geometric_dimension() # space dimension
#
v = TestFunction(V)
a = 2.0 * v.dx(0) * u.dx(0) * dx + alpha * v.dx(0).dx(0) * u.dx(0).dx(0) * dx + alpha * v.dx(1).dx(1) * u.dx(1).dx(1) * dx + 2 * alpha * v.dx(1).dx(0) * u.dx(1).dx(0) * dx
L = 2.0 * v.dx(0) * mh * dx
u = Function(V)
solve(a == L, u, bc)

plot(u, title = "Phi")

plot(mh, mesh = mesh, title = "M (input)")#, mode = "displacement")

plot(u.dx(0), title = "M (computed)")#, mode = "displacement")

plot(u.dx(0) - mh, title = "M (computed - input)")#, mode = "displacement")

plot(u.dx(1), title = "S")#, mode = "displacement")

plot(u.dx(0).dx(0), title = "dydy")#, mode = "displacement")
plot(u.dx(0).dx(1), title = "dydx")#, mode = "displacement")
plot(u.dx(1).dx(1), title = "dxdx")#, mode = "displacement")
interactive()
