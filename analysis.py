#%%

import numpy
import pandas
import os
import matplotlib.pyplot as plt

os.chdir("/home/bbales2/magnets")

df = pandas.read_pickle("Gd_alldata.p")

#%%

def split_MTs(df, col="Magnetic Field (T)", precision=3):
   '''Splits a df containing M(T) runs at different fields, return a pandas
   groupby object, with the df grouped into isofield data (individual M(T)'s)'''
   return df.groupby(lambda x: numpy.around(df[col][x], precision))

M = {}

temps = numpy.linspace(235, 370, 100)
Mhs = []
Ts = []

for T, data in split_MTs(df):
    idx = numpy.argmin(data['Temperature (K)'].values)

    #plt.plot(data['Temperature (K)'][idx:], data['normalized_moment_cgs'][idx:])


    temperatures = []
    mags = []

    #for i in range(data.index[0], idx):
    temperature = []
    mags = []

    for temp, mag in zip(data['Temperature (K)'][idx:], data['normalized_moment_cgs'][idx:]):

        #if temp < 230 or temp > 375:
        #    continue

        temperature.append(temp)
        mags.append(mag)

    idxs = numpy.argsort(temperature)

    temperature = numpy.array(temperature)[idxs]
    mags = numpy.array(mags)[idxs]

    magsi = numpy.interp(temps, temperature, mags)
    Mhs.append(magsi)
    Ts.append(T)

    M[T] = numpy.array((temperature, mags))

    plt.plot(temperature, mags)

plt.show()
#%%
import scipy.interpolate

idxs = numpy.argsort(Ts)
Ts = numpy.array(Ts)[idxs]
Mhs = numpy.array(Mhs)[idxs]

plt.imshow(Mhs, interpolation = 'NONE')
plt.show()

interp = scipy.interpolate.interp2d(temps, Ts, Mhs)

Ts2 = numpy.linspace(min(Ts), max(Ts), 50)
temps2 = numpy.linspace(min(temps), max(temps), 50)

xs = numpy.linspace(0, 1.0, 50, endpoint = False)
ys = numpy.linspace(0, 1.0, 50, endpoint = False)

Mhis = interp(temps2, Ts2)

plt.imshow(Mhis, interpolation = 'NONE')
plt.show()

Mhis = numpy.flipud(Mhis)
#%%

Mhis = numpy.zeros((50, 50))

for l in range(50):
    for m in range(50):
        Mhis[l, m] = numpy.imag(numpy.exp(2 * 1j * numpy.pi * (3 * ys[l] + 1 * xs[m])))# + numpy.random.randn()

#%%
plt.imshow(Mhis, interpolation = 'NONE')
plt.show()

J = 8
K = 8

V = numpy.zeros((50, 50, J * K * 2))

for l in range(50):
    for m in range(50):
        for j in range(0, J):
            for k in range(0, K):
                #V[l, m, j - 1, k - 1, 0] = (float(k) / j) * numpy.real(numpy.exp(2 * 1j * numpy.pi * (k * ys[l] + j * xs[m])))#
                #V[l, m, j - 1, k - 1, 1] = (float(k) / j) * numpy.imag(numpy.exp(2 * 1j * numpy.pi * (k * ys[l] + j * xs[m])))#
                V[l, m, j * K + k] = numpy.real(numpy.exp(2 * 1j * numpy.pi * (k * ys[l] + j * xs[m])))#
                V[l, m, j * K + k + J * K] = numpy.imag(numpy.exp(2 * 1j * numpy.pi * (k * ys[l] + j * xs[m])))#

#Vnorm = numpy.zeros((J, K, 2))
#for j in range(J):
#    for k in range(K):
#        for l in range(2):
#            Vnorm[j, k, l] = numpy.linalg.norm(V[:, :, j, k, l].flatten()) + 1e-9

#Q = numpy.einsum('abcde, abfgh', V, V).reshape((-1, 32))
#W = numpy.einsum('abcde, ab', V, Mhis).reshape((32))
Q = numpy.einsum('abc, abd', V, V)
W = numpy.einsum('abc, ab', V, Mhis)
Q[64, 64] = 1
a = numpy.linalg.solve(Q, W)

Mha = numpy.real(numpy.einsum('jkm, m', V, a))

plt.imshow(Mha, interpolation = 'NONE')
plt.colorbar()
plt.show()
#%%
dM = {}

xs = []
ys = []
zs = []

temps = numpy.linspace(235, 370, 100)

dmdts = []
Hs = []

for T in M:
    temperature, mags = M[T]
    dts = temperature[1:] - temperature[:-1]
    dms = mags[1:] - mags[:-1]

    dM[T] = ((temperature[1:] + temperature[:-1]) / 2.0, dms / dts)

    for temperature, dmdt in zip(*dM[T]):
        xs.append(temperature)
        ys.append(T)
        zs.append(dmdt)

    dmdt_hat = numpy.interp(temps, dM[T][0], dM[T][1])
    dmdts.append(dmdt_hat)
    Hs.append(T)

    plt.plot(temps, dmdt_hat)#*dM[T])

print Hs
idxs = numpy.argsort(Hs)

Hs = numpy.array(Hs)[idxs]
dmdts = numpy.array(dmdts)[idxs]

#%%
plt.imshow(dmdts, interpolation = 'NONE')
plt.colorbar()
plt.show()

dS = numpy.zeros((len(Hs), len(temps)))

for i in range(1, dmdts.shape[0]):
    dS[i] = dS[i - 1] + (Hs[i] - Hs[i - 1]) * dmdts[i - 1]

plt.imshow(dS, interpolation = 'NONE')
plt.show()

import scipy.interpolate

interp = scipy.interpolate.interp2d(temps, Hs, dS)
#%%

Htmp = numpy.linspace(min(Hs), max(Hs), 100)

plt.imshow(interp(temps, Htmp)[::-1], interpolation = 'NONE', extent = [min(temps), max(temps), min(Hs), max(Hs)], aspect = 20.0)
plt.xlabel('Temperature')
plt.ylabel('Magnetic field')
plt.title('$\Delta S_M$')
plt.colorbar()
plt.show()

#%%
dmdt = scipy.interpolate.interp2d(xs, ys, zs)

#%%
Temperatures = numpy.linspace(235, 370, 100)
Ts = numpy.linspace(min(M.keys()), max(M.keys()), 100)

#%%
vs = dmdt(Temperatures, Ts)

plt.imshow(vs)
plt.show()
