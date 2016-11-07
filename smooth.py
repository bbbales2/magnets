#%%

import numpy
import matplotlib.pyplot as plt

N = 20

hs = numpy.linspace(0, 1.0, N)
ts = numpy.linspace(0, 1.0, N)

hs2, ts2 = numpy.meshgrid(hs, ts, indexing = 'ij')

mh = numpy.sin(2 * numpy.pi * 2 * hs2) * numpy.cos(2 * numpy.pi * 1.1 * ts2) + numpy.random.randn(N, N) * 0.1

plt.imshow(mh, interpolation = 'NONE')
plt.show()

dh = hs[1] - hs[0]
dt = ts[1] - ts[0]

#%%

dmds = numpy.zeros((N, N, N + 1, N))

for i in range(N):
    for j in range(N):
        for ih in [i, i + 1]:
            for jh in range(1, j + 1):
                if ih == i:
                    dmds[i, j, ih, jh] = -dt / dh
                if ih == i + 1:
                    dmds[i, j, ih, jh] = dt / dh

#%%

def getm(s, m0):
    m = numpy.zeros((N, N))

    for i in range(N):
        m[i] = 0#m0[i]
        for j in range(N):
            for jh in range(1, j + 1):
                m[i, j] += (s[i + 1, jh] - s[i, jh]) * dt / dh

    return m
#%%
alpha = 0.0001

def loss(m, m0, s):
    #losst = 0.0#
    losst = numpy.sum((m - mh)**2)

    for i in range(N):
        if i < N - 1:
            losst += alpha * (m0[i + 1] - m0[i])**2 / dh ** 2

    for i in range(N):
        for j in range(N):
            #print 'hi', alpha
            losst += alpha * (s[i + 1, j] - s[i, j])**2 / (dh ** 2)

            if j < N - 1:
                losst += alpha * (s[i, j + 1] - s[i, j])**2 / (dt ** 2)

    return losst

def dlossdm0(m, m0):
    dloss = numpy.sum(2 * (m - mh), axis = 1)

    for i in range(N):
        if i < N - 1:
            dloss[i] += -2 * alpha * (m0[i + 1] - m0[i]) / dh**2
            dloss[i + 1] += 2 * alpha * (m0[i + 1] - m0[i]) / dh**2

    return dloss

def dlossds(m, s):
    #dlossds = numpy.zeros((N + 1, N))
    dloss = 2 * numpy.einsum('ijkl,ij', dmds, (m - mh))

    for i in range(N):
        for j in range(N):
            dloss[i, j] += -2 * alpha * (s[i + 1, j] - s[i, j]) / dh**2
            dloss[i + 1, j] += 2 * alpha * (s[i + 1, j] - s[i, j]) / dh**2

            if j < N - 1:
                dloss[i, j] += -2 * alpha * (s[i, j + 1] - s[i, j]) / dt**2
                dloss[i, j + 1] += 2 * alpha * (s[i, j + 1] - s[i, j]) / dt**2

    return dloss

#%%


s = numpy.random.randn(N + 1, N)
m0 = numpy.random.randn(N)
#%%
approx = numpy.zeros((N + 1, N))
approx1 = numpy.zeros(N)

exact = dlossds(getm(s, m0), s)
exact1 = dlossdm0(getm(s, m0), m0)
#%%
for i in range(N + 1):
    for j in range(N):
        sh = s.copy()

        sh[i, j] *= 1.0001

        approx[i, j] = (loss(getm(s, m0), m0, s) - loss(getm(sh, m0), m0, sh)) / (s[i, j] - sh[i, j])

        #print exact[i, j], approx[i, j]
#%%
for i in range(N):
    m0h = m0.copy()

    m0h[i] *= 1.0001

    approx1[i] = (loss(getm(s, m0), m0, s) - loss(getm(s, m0h), m0h, s)) / (m0[i] - m0h[i])
#print max(numpy.abs(approx - exact).flatten())
print approx1
print exact1

#%%
plt.imshow(approx - exact, interpolation = 'NONE')
plt.colorbar()
plt.show()
#%%
plt.imshow(approx, interpolation = 'NONE')
plt.colorbar()
plt.show()

plt.imshow(exact, interpolation = 'NONE')
plt.colorbar()
plt.show()

#%%

s = numpy.random.randn(N + 1, N)
s[:, 0] = 0
m0 = numpy.random.randn(N)

#%%
while 1:
    m = getm(s, m0)

    l = loss(m, m0, s)
    dlds = dlossds(m, s)
    dldm = dlossdm0(m, m0)

    norm = numpy.linalg.norm(dlds.flatten()) + numpy.linalg.norm(dldm.flatten())
    dlds /= norm
    dldm /= norm

    s -= dlds * 1.0e-1
    m0 -= dldm * 1.0e-1

    print l

#%%

import pickle

f = open('/home/bbales2/magnets/mhi')
mh = pickle.load(f)
f.close()

#%%

plt.imshow((s[1:] - s[:-1]) / dh, interpolation = 'NONE')
plt.title('ds/dh')
plt.colorbar()
plt.show()

plt.imshow(s, interpolation = 'NONE')
plt.title('s')
plt.colorbar()
plt.show()

plt.imshow(getm(s, m0), interpolation = 'NONE')
plt.title('m')
plt.colorbar()
plt.show()

plt.imshow(mh, interpolation = 'NONE')
plt.title('mh')
plt.colorbar()
plt.show()

plt.imshow(getm(s, m0) - mh, interpolation = 'NONE')
plt.title('error (m - mh)')
plt.colorbar()
plt.show()