#%%

import numpy
import pickle
import os
import matplotlib.pyplot as plt

os.chdir('/home/bbales2/magnets2/')

f = open('mhi_100x100_gd')
m = pickle.load(f)
f.close()

#m = numpy.flipud(m)

m = m[::4, ::4]

plt.imshow(m)
plt.colorbar()
plt.show()

my = m[1:] - m[:-1]
#mz = (m[1:] + m[:-1]) / 2.0

plt.imshow(my)
plt.colorbar()
plt.show()

#my[0] += m[0]

plt.imshow(numpy.cumsum(my, axis = 0))
plt.colorbar()
plt.show()

J, I = my.shape
xs = numpy.linspace(0, 1.0, I)
ys = numpy.linspace(0, 1.0, J)

dx = xs[1] - xs[0]
dy = ys[1] - ys[0]

my /= dy
#%%
s = numpy.zeros(my.shape)

a = 0.001

A = numpy.zeros((my.shape[0], my.shape[1], my.shape[0], my.shape[1]))
b = numpy.zeros(my.shape)

for j in range(J):
    integral = 0.0

    for i in range(I):
        x = xs[i]

        if j == 0:
            A[j, i, j + 1, i] += a / dx**2
            A[j, i, j, i] -= a / dx**2
        elif j == J - 1:
            A[j, i, j - 1, i] += a / dx**2
            A[j, i, j, i] -= 2 * a / dx**2
        else:
            A[j, i, j + 1, i] += a / dx**2
            A[j, i, j, i] -= 2 * a / dx**2
            A[j, i, j - 1, i] += a / dx**2

        if i == 0:
            A[j, i, j, i + 1] += a / dx**2
            A[j, i, j, i] -= a / dx**2
        elif i == I - 1:
            A[j, i, j, i - 1] += a / dx**2
            A[j, i, j, i] -= a / dx**2
        else:
            A[j, i, j, i + 1] += a / dx**2
            A[j, i, j, i] -= 2 * a / dx**2
            A[j, i, j, i - 1] += a / dx**2

        for ii in range(i + 1):
            f = x# if ii < i else (x / 2.0)

            if j == 0:
                A[j, i, j + 1, ii] += f * dx / dy**2
                A[j, i, j, ii] -= f * dx / dy**2
            elif j == J - 1:
                A[j, i, j - 1, ii] += f * dx / dy**2
                A[j, i, j, ii] -= 2 * f * dx / dy**2
            else:
                A[j, i, j + 1, ii] += f * dx / dy**2
                A[j, i, j, ii] -= 2 * f * dx / dy**2
                A[j, i, j - 1, ii] += f * dx / dy**2

        b[j, i] = x * my[j, i]

A = numpy.reshape(A, (A.shape[0] * A.shape[1], A.shape[0] * A.shape[1]))

s = numpy.linalg.solve(A, b.flatten())

s = s.reshape(my.shape)

plt.imshow(s, interpolation = 'NONE')
plt.colorbar()
plt.show()

mh = numpy.zeros(my.shape)

mh = numpy.cumsum(s[1:] - s[:-1], axis = 1)

plt.imshow(mh, interpolation = 'NONE')
plt.colorbar()
plt.show()

#%%
mh = numpy.zeros(my.shape)
sy = numpy.zeros(my.shape)

for j in range(J):
    integral = 0.0
    
    for i in range(I):
        x = xs[i]
                
        if j == 0:
            sy[j, i] = s[j + 1, i] - s[j, i]
        elif j == J - 1:
            sy[j, i] = s[j, i] - s[j - 1, i]
        else:
            sy[j, i] = (s[j + 1, i] - s[j - 1, i]) / 2.0
            
        sy[j, i] /= dy
        
        total = 0.0
        for ii in range(i + 1):
            f = 1.0# if ii < i else (1 / 2.0)        
                
            total += f * sy[j, ii] * dx
            
        mh[j, i] = total
        
plt.imshow(mh, interpolation = 'NONE')
plt.colorbar()
plt.show()
#%%

sxx = numpy.zeros(my.shape)
syy = numpy.zeros(my.shape)

for j in range(J):
    integral = 0.0
    
    for i in range(I):
        x = xs[i]
                
        if j == 0:
            syy[j, i] = s[j + 1, i] - s[j, i]
        elif j == J - 1:
            syy[j, i] = -2 * s[j, i] + s[j - 1, i]
        else:
            syy[j, i] = s[j + 1, i] - 2 * s[j, i] + s[j - 1, i]
            
        syy[j, i] /= dy**2
            
        if i == 0:
            sxx[j, i] = s[j, i + 1] - s[j, i]
        elif i == I - 1:
            sxx[j, i] = -s[j, i] + s[j, i - 1]
        else:
            sxx[j, i] = s[j, i + 1] - 2 * s[j, i] + s[j, i - 1]
            
        sxx[j, i] /= dx**2

        total = 0.0
        for ii in range(i + 1):
            f = 1.0 * x# if ii < i else (x / 2.0)        
                
            total += f * syy[j, ii] * dx
            
        print a * sxx[j, i]
        print a * syy[j, i]
        print x * total
        print -x * my[j, i]
        print a * sxx[j, i] + a * syy[j, i] + x * total - x * my[j, i]
        print '--'
                    
        #b[j, i] = x * my[j, i]
#%%
myh = A.dot(s.flatten()).reshape(my.shape)
myt = my.copy()

for i in range(I):
    #myh[:, i] *= xs[i]
    myt[:, i] *= xs[i]

plt.imshow(myh)
plt.show()

plt.imshow(myt)
plt.show()