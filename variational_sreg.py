#%%

import numpy
import pickle
import os
import matplotlib.pyplot as plt

os.chdir('/home/bbales2/magnets/')

fname = 'gd'

f = open('mhi_100x100_{0}'.format(fname))
m = pickle.load(f)
f.close()

#m = numpy.flipud(m)

m = m[::, ::]

plt.imshow(numpy.flipud(m), extent = [0.0, 1.0, 0.0, 1.0])
plt.colorbar()
plt.show()

I, J = m.shape
ys = numpy.linspace(0, 1.0, I)
xs = numpy.linspace(0, 1.0, J)

dx = xs[1] - xs[0]
dy = ys[1] - ys[0]

#mi = numpy.cumsum(m, axis = 0) * dy

#md = numpy.pad(m[1:] - m[:-1], ((1, 0), (0, 0)), mode = 'constant') / dy
#md = numpy.gradient(m, dy, axis = 0)

#plt.imshow(md, interpolation = 'NONE')
#plt.title("$m'$")
#plt.show()
#%%

a = 0.00001

A = numpy.zeros((m.shape[0], m.shape[1], m.shape[0], m.shape[1]))
b = numpy.zeros(m.shape)

for i in range(I):
    for j in range(J):
        #A[i, j, i, j] += 1

        if i == 0:
            A[i, j, i, j] -= 1 / (dy**2)
            A[i, j, i + 1, j] += 2 / (dy**2)
            #b[i, j] += m[i, j] / (2.0 * dy)
            #A[i, j, i, j] -= 1 / (2.0 * dy**2)
            #A[i, j, i + 1, j] += 1 / (2.0 * dy**2)
            #b[i, j] += m[i, j] / (2.0 * dy)
        elif i == I - 1:
            #A[i, j, i - 1, j] += 1 / dy**2
            #A[i, j, i, j] -= 2 / dy**2
            A[i, j, i - 1, j] += 1 / (2.0 * dy**2)
            A[i, j, i, j] -= 1 / (2.0 * dy**2)
            b[i, j] -= m[i, j] / (2.0 * dy)
        else:
            A[i, j, i - 1, j] += 1 / dy**2
            A[i, j, i, j] -= 2 / dy**2
            A[i, j, i + 1, j] += 1 / dy**2

        #if i < I - 1:

#        if j > 0:
#            A[i, j, i, j - 2] += -0.5 * a / dx**4
#
#        if j > 1:
#            A[i, j, i, j - 1] -= 4 * -0.5 * a / dx**4
#
#        A[i, j, i, j] += -0.5 * 6 * a / dx**4
#
#        if j < J - 1:
#            A[i, j, i, j + 1] -= 4 * -0.5 * a / dx**4
#
#        if j < J - 2:
#            A[i, j, i, j + 2] += -0.5 * a / dx**4

# I think this is correct but stuff works better with it commented
#        if j == 1:
#            A[i, j, i, j - 1] -= 2 * -0.5 * a / (dx**4)
#            A[i, j, i, j] += 5.0 * -0.5 * a / (dx**4)
#            A[i, j, i, j + 2] -= 4.0 * -0.5 * a / (dx**4)
#            A[i, j, i, j + 3] += 1.0 * -0.5 * a / (dx**4)
#        elif j == J - 2:
#            A[i, j, i, j - 3] += 1.0 * -0.5 * a / (dx**4)
#            A[i, j, i, j - 2] -= 4.0 * -0.5 * a / (dx**4)
#            A[i, j, i, j] += 5.0 * -0.5 * a / (dx**4)
#            A[i, j, i, j + 1] -= 2.0 * -0.5 * a / (dx**4)
# I don't think this is correct
#        if j == 1:
#            A[i, j, i, j - 1] -= 0.5 * -0.5 * a / (2.0 * dx**4)
#            A[i, j, i, j] += 1.0 * -0.5 * a / (2.0 * dx**4)
#            A[i, j, i, j + 2] -= 1.0 * -0.5 * a / (2.0 * dx**4)
#            A[i, j, i, j + 3] += 0.5 * -0.5 * a / (2.0 * dx**4)
#        elif j == J - 2:
#            A[i, j, i, j - 3] += 0.5 * -0.5 * a / (2.0 * dx**4)
#            A[i, j, i, j - 2] -= 1.0 * -0.5 * a / (2.0 * dx**4)
#            A[i, j, i, j] += 1.0 * -0.5 * a / (2.0 * dx**4)
#            A[i, j, i, j + 1] -= 0.5 * -0.5 * a / (2.0 * dx**4)
        if j > 1 and j < J - 2:
            A[i, j, i, j - 2] += -0.5 * a / dx**4
            A[i, j, i, j - 1] -= 4 * -0.5 * a / dx**4
            A[i, j, i, j] += -0.5 * 6 * a / dx**4
            A[i, j, i, j + 1] -= 4 * -0.5 * a / dx**4
            A[i, j, i, j + 2] += -0.5 * a / dx**4
#
#        if i > 0:
#            if j > 0:
#                A[i, j, i - 1, j - 1] += -a / (dy**2 * dx**2)
#                A[i, j, i - 1, j] -= -a / (dy**2 * dx**2)
#                A[i, j, i, j - 1] -= -a / (dy**2 * dx**2)
#                A[i, j, i, j] += -a / (dy**2 * dx**2)
#
#        if i > 0:
#            if j < J - 1:
#                A[i, j, i - 1, j] += -a / (dy**2 * dx**2)
#                A[i, j, i - 1, j + 1] -= -a / (dy**2 * dx**2)
#                A[i, j, i, j] -= -a / (dy**2 * dx**2)
#                A[i, j, i, j + 1] += -a / (dy**2 * dx**2)
#
#        if i < I - 1:
#            if j > 0:
#                A[i, j, i, j - 1] += -a / (dy**2 * dx**2)
#                A[i, j, i, j] -= -a / (dy**2 * dx**2)
#                A[i, j, i + 1, j - 1] -= -a / (dy**2 * dx**2)
#                A[i, j, i + 1, j] += -a / (dy**2 * dx**2)
#
#        if i < I - 1:
#            if j < J - 1:
#                A[i, j, i, j] += -a / (dy**2 * dx**2)
#                A[i, j, i, j + 1] -= -a / (dy**2 * dx**2)
#                A[i, j, i + 1, j] -= -a / (dy**2 * dx**2)
#                A[i, j, i + 1, j + 1] += -a / (dy**2 * dx**2)


        if j == 0 or j == J - 1:
            pass
        elif i == 0:
            A[i, j, i, j - 1] -= 1 * -a / (dy**2 * dx**2)

            A[i, j, i, j] += 2 * -a / (dy**2 * dx**2)

            A[i, j, i, j + 1] -= 1 * -a / (dy**2 * dx**2)

            A[i, j, i + 1, j - 1] += -a / (dy**2 * dx**2)

            A[i, j, i + 1, j] -= 2 * -a / (dy**2 * dx**2)

            A[i, j, i + 1, j + 1] += -a / (dy**2 * dx**2)



#        elif i == I - 1:
#            if i > 0 and j > 0:
#                A[i, j, i - 1, j - 1] += -a / (dy**2 * dx**2)
#
#            if i > 0:
#                A[i, j, i - 1, j] -= 2 * -a / (dy**2 * dx**2)
#
#            if i > 0 and j < J - 1:
#                A[i, j, i - 1, j + 1] += -a / (dy**2 * dx**2)
#
#            if j > 0:
#                A[i, j, i, j - 1] -= 1 * -a / (dy**2 * dx**2)
#
#            A[i, j, i, j] += 2 * -a / (dy**2 * dx**2)
#
#            if j < J - 1:
#                A[i, j, i, j + 1] -= 1 * -a / (dy**2 * dx**2)


        else:
            if i > 0 and j > 0:
                A[i, j, i - 1, j - 1] += -a / (dy**2 * dx**2)

            if i > 0:
                A[i, j, i - 1, j] -= 2 * -a / (dy**2 * dx**2)

            if i > 0 and j < J - 1:
                A[i, j, i - 1, j + 1] += -a / (dy**2 * dx**2)

            if j > 0:
                A[i, j, i, j - 1] -= 2 * -a / (dy**2 * dx**2)

            A[i, j, i, j] += 4 * -a / (dy**2 * dx**2)

            if j < J - 1:
                A[i, j, i, j + 1] -= 2 * -a / (dy**2 * dx**2)

            if i < I - 1 and j > 0:
                A[i, j, i + 1, j - 1] += -a / (dy**2 * dx**2)

            if i < I - 1:
                A[i, j, i + 1, j] -= 2 * -a / (dy**2 * dx**2)

            if i < I - 1 and j < J - 1:
                A[i, j, i + 1, j + 1] += -a / (dy**2 * dx**2)

        #if j == 0:
        #    A[i, j, i, j + 1] += -a / dx**2
        #    A[i, j, i, j] -= -a / dx**2
        #elif j == J - 1:
        #    A[i, j, i, j - 1] += -a / dx**2
        #    A[i, j, i, j] -= -a / dx**2
        #else:
        #    A[i, j, i, j + 1] += -a / dx**2
        #    A[i, j, i, j] -= -2 * a / dx**2
        #    A[i, j, i, j - 1] += -a / dx**2

        if i == 0:
            b[i, j] += (m[i + 1, j] - m[i, j]) / dy
        elif i == I - 1:
            b[i, j] += (m[i, j] - m[i - 1, j]) / dy
        else:
            b[i, j] += (m[i + 1, j] - m[i - 1, j]) / (2.0 * dy)
        #b[i, j] += 0.5 * md[i, j]

A = numpy.reshape(A, (A.shape[0] * A.shape[1], A.shape[0] * A.shape[1]))

phi = numpy.linalg.solve(A, b.flatten()).reshape(m.shape)

plt.imshow(numpy.flipud(phi), interpolation = 'NONE', extent = [0.0, 1.0, 0.0, 1.0])
plt.title('$\Phi$, {0}'.format(fname))
plt.colorbar()
plt.show()

m__ = numpy.gradient(phi, dy, axis = 0)

plt.imshow(numpy.flipud(m__), interpolation = 'NONE', extent = [0.0, 1.0, 0.0, 1.0])
plt.title('$M$, {0}'.format(fname))
plt.colorbar()
plt.show()

s__ = numpy.gradient(phi, dx, axis = 1)

plt.imshow(numpy.flipud(s__), interpolation = 'NONE', extent = [0.0, 1.0, 0.0, 1.0])
plt.title('$\Delta S$, {0}'.format(fname))
plt.colorbar()
plt.show()

#%%
