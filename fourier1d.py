#%%

import matplotlib.pyplot as plt
import numpy

M = 40
K = 5

x = numpy.linspace(0, 1, M)

data = numpy.zeros((M))
for i in range(M):
    data[i] = numpy.sin(2 * numpy.pi * 2 * x[i] + 0.5) + numpy.random.randn() / 5

plt.plot(data, '*')
plt.show()

V = numpy.zeros((M, K), dtype = 'complex128')

for m in range(M):
    for k in range(1, K + 1):
        V[m, k - 1] = 2 * 1j * numpy.pi * k * numpy.exp(2 * 1j * numpy.pi * k * x[m])

#plt.plot(x, numpy.real(V[:, 2]))
#plt.plot(x, numpy.imag(V[:, 2]))
Vh = numpy.real(V) - numpy.imag(V)

r = numpy.linalg.solve(numpy.real(V).T.dot(numpy.real(V)), numpy.real(V).T.dot(data))
c = numpy.linalg.solve(-numpy.imag(V).T.dot(numpy.imag(V)), numpy.imag(V).T.dot(data))

f = numpy.real(V.dot(r + 1j * c))

plt.plot(x, f)
plt.plot(x, data, '*')
plt.show()

#%%
print Vh.T.dot(data)
print numpy.real(V).T.dot(data)
print numpy.imag(V).T.dot(data)