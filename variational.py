#%%

import numpy
import pickle
import os
import matplotlib.pyplot as plt

os.chdir('/home/bbales2/magnets2/')

f = open('mhi_100x100_fege')
m = pickle.load(f)
f.close()

#m = numpy.flipud(m)

m = m[::1, ::1]

plt.imshow(m)
plt.colorbar()
plt.show()

I, J = m.shape
ys = numpy.linspace(0, 1.0, I)
xs = numpy.linspace(0, 1.0, J)

dx = xs[1] - xs[0]
dy = ys[1] - ys[0]

mi = numpy.cumsum(m, axis = 0) * dy

#%%

a = 0.0001

A = numpy.zeros((mi.shape[0], mi.shape[1], mi.shape[0], mi.shape[1]))
b = numpy.zeros(mi.shape)

for i in range(I):
    for j in range(J):
        A[i, j, i, j] += 1

        if i == 0:
            pass
            #A[i, j, i + 1, j] += -0.5 * a / dx**2
            #A[i, j, i, j] -= -0.5 * a / dx**2
        elif i == I - 1:
            pass
            #A[i, j, i - 1, j] += -0.5 * a / dx**2
            #A[i, j, i, j] -= -0.5 * a / dx**2
        else:
            A[i, j, i + 1, j] += -0.5 * a / dx**2
            A[i, j, i, j] -= -0.5 * 2 * a / dx**2
            A[i, j, i - 1, j] += -0.5 * a / dx**2

        if j == 0:
            A[i, j, i, j + 1] += -a / dx**2
            A[i, j, i, j] -= -a / dx**2
        elif j == J - 1:
            A[i, j, i, j - 1] += -a / dx**2
            A[i, j, i, j] -= -a / dx**2
        else:
            A[i, j, i, j + 1] += -a / dx**2
            A[i, j, i, j] -= -2 * a / dx**2
            A[i, j, i, j - 1] += -a / dx**2

        b[i, j] = 0.5 * mi[i, j]

A = numpy.reshape(A, (A.shape[0] * A.shape[1], A.shape[0] * A.shape[1]))

phi = numpy.linalg.solve(A, b.flatten()).reshape(m.shape)

plt.imshow(phi, interpolation = 'NONE')
plt.title('$\Phi$')
plt.colorbar()
plt.show()

m__ = numpy.gradient(phi, axis = 0)

plt.imshow(m__, interpolation = 'NONE')
plt.title('$M$')
plt.colorbar()
plt.show()

s__ = numpy.gradient(phi, axis = 1)

plt.imshow(s__, interpolation = 'NONE')
plt.title('$\Delta S$')
plt.colorbar()
plt.show()

#%%