import numpy
import pickle
import os
import matplotlib.pyplot as plt
import scipy.interpolate

os.chdir('/home/bbales2/magnets/')

fname = 'fege'

f = open('mhi_100x100_{0}'.format(fname))
m = pickle.load(f)
f.close()

from fenics import *

# Scaled variables
H = 1.0
W = 1.0

alpha = 100.0

# Create mesh and define function space
mesh = RectangleMesh(Point(0.0, 0.0), Point(H, W), 40, 40)
V = FunctionSpace(mesh, "P", 2)
#R = FiniteElement(mesh, "R", 0)
#V = FiniteElement("P", mesh.ufl_cell(), 2)
#R = FiniteElement("R", mesh.ufl_cell(), 0)
#W = FunctionSpace(mesh, V * R)
# Define boundary condition
tol = 1E-10

def clamped_boundary(x, on_boundary):
    return on_boundary and (x[0] < tol)# or x[1] > 1 - tol)

bc = DirichletBC(V, Constant(0), clamped_boundary)

y = numpy.linspace(0, 1.0, m.shape[0])
x = numpy.linspace(0, 1.0, m.shape[1])

Mhatf = scipy.interpolate.interp2d(x, y, m)

class Mhat(Expression):
    def eval(self, value, x):
        value[0] = Mhatf(x[1], x[0])#m[min(m.shape[0] - 1, int(x[0] * m.shape[0])), min(m.shape[1] - 1, int(x[1] * m.shape[1]))]

mh = Mhat(element = V.ufl_element())

# Define variational problem
u = TrialFunction(V)
d = u.geometric_dimension() # space dimension

v = TestFunction(V)
a = 2.0 * v.dx(0) * u.dx(0) * dx + alpha * v.dx(1).dx(1) * u.dx(1).dx(1) * dx + alpha * v.dx(0).dx(0) * u.dx(0).dx(0) * dx + 2 * alpha * v.dx(1).dx(0) * u.dx(1).dx(0) * dx# + c * v * dx + d * u * dx
L = 2.0 * v.dx(0) * mh * dx# + v * dx
u = Function(V)
solve(a == L, u, bc)

#mh, _ = Mhat(element = W.ufl_element())

# Define variational problem
#u, c = TrialFunction(W)
#d = u.geometric_dimension() # space dimension

#v, d = TestFunction(W)
#a = 2.0 * v.dx(0) * u.dx(0) * dx + alpha * v.dx(1).dx(1) * u.dx(1).dx(1) * dx + alpha * v.dx(0).dx(0) * u.dx(0).dx(0) * dx + 2 * alpha * v.dx(1).dx(0) * u.dx(1).dx(0) * dx# + c * v * dx + d * u * dx
#L = 2.0 * v.dx(0) * mh * dx
# Compute solution
#w = Function(W)
#solve(a == L, w)

#(u, c) = w.split()

plot(u, title = "Phi")

plot(mh, mesh = mesh, title = "M (input)")#, mode = "displacement")

plot(u.dx(0), title = "M (computed)")#, mode = "displacement")

plot(u.dx(0) - mh, title = "M (computed - input)")#, mode = "displacement")

plot(u.dx(1), title = "S")#, mode = "displacement")
interactive()
