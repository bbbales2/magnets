#%%

import matplotlib.pyplot as plt
import numpy

N = 20
K = 30
M = 64

x = numpy.linspace(0, 1, N)
y = numpy.linspace(0, 1, K)

m = numpy.zeros((N, K))
for i in range(N):
    for j in range(K):
        m[i, j] = numpy.sin(2 * numpy.pi * 2 * x[i] + 2 * y[i]) + numpy.random.randn() / 5

plt.imshow(m, interpolation = 'NONE')
plt.show()

#%%
V = numpy.zeros((N, K, M))

for n in range(N):
    for k in range(K):
        for i in range(M):
            #if i == 0:
            #    V[n, k, i] = 1.0
            #else:
                ii = i / 8
                jj = i % 8

                V[n, k, i] = -(numpy.pi * (ii + 1)) * numpy.sin(numpy.pi * (ii + 1) * x[n]) * numpy.cos(numpy.pi * jj * y[k])

#%%
#plt.plot(x, numpy.real(V[:, 2]))
#plt.plot(x, numpy.imag(V[:, 2]))

Vh = V.reshape((-1, M))

a = numpy.linalg.solve(numpy.einsum('nkl, nkm', V, V), numpy.einsum('nkl, nk', V, m))

mh = numpy.einsum('ijk, k', V, a)

plt.plot(m[:, :])
plt.plot(mh[:, :])
plt.legend(['Data', 'Fit'])
plt.show()

plt.imshow(mh, interpolation = 'NONE')
plt.colorbar()
plt.show()

plt.imshow(m, interpolation = 'NONE')
plt.colorbar()
#plt.plot(numpy.imag(V.dot(a)))
plt.show()

plt.imshow(mh - m, interpolation = 'NONE')
plt.colorbar()
#plt.plot(numpy.imag(V.dot(a)))
plt.show()