#!/usr/bin/env python

#%%
import numpy as np
import pandas as pd
import scipy.interpolate

def gen_legendre_poly(n, xs, dx = False, integrate = False):

    if n % 2 == 0:
        M = int(float(n)/2)
    else:
        M = int(float(n - 1)/2)

    z = np.zeros((xs.shape[0]))

    for i in range(M+1):
        num = np.math.factorial(2*n - 2*i)
        d1 = (2**n) * np.math.factorial(i)
        d2 = np.math.factorial(n - i)
        d3 = np.math.factorial(n - 2*i)

        for j in range(xs.shape[0]):
            if dx:
                z[j] += (-1)**i * (n - 2*i) * (num / float(d1 * d2 * d3)) * (xs[j] ** (n - 2*i - 1))
            elif integrate:
                z[j] += (-1)**i * (num / float(d1 * d2 * d3 * (n - 2*i + 1))) * (xs[j] ** (n - 2*i + 1))
            else:
                z[j] += (-1)**i * (num / float(d1 * d2 * d3)) * (xs[j] ** (n - 2*i))

    return z

def gen_fourier(n, xs, dx = False, integrate = False):

    z = np.zeros((xs.shape[0]))

    for i in range(n):
        for j in range(xs.shape[0]):
            if dx:
                z[j] += np.real(1j * 2 * np.pi * np.exp(2 * 1j * np.pi * i * xs[j]))
            elif integrate:
                z[j] += np.real((1/(2 * 1j * np.pi)) * np.exp(2 * 1j * np.pi * i * xs[j]))
            else:
                z[j] += np.real(np.exp(2 * 1j * np.pi * i * xs[j]))

    return z

def gen_legendre_poly_2d(n, m, xs, ys, dx = False, dy = False, intx = False, inty = False):

    xl = gen_legendre_poly(n, xs, dx = dx, integrate = intx)
    yl = gen_legendre_poly(m, ys, dx = dy, integrate = inty)

    z = np.zeros((xs.shape[0],ys.shape[0]))
    for l in range(xs.shape[0]):
        for k in range(ys.shape[0]):
            z[l,k] = xl[l] * yl[k]

    return z

def gen_fourier_2d(n, m, xs, ys, dx = False, dy = False, intx = False, inty = False):

    z = np.zeros((xs.shape[0],ys.shape[0]))
    for l in range(xs.shape[0]):
        for k in range(ys.shape[0]):
            z[l,k] = np.real(np.exp(2 * 1j * np.pi * (n * xs[l] + (m+1) * xs[k])))

    return z

def fit(M, I=4, J=4, dx = False, dy = False, intx = False, inty = False, gen_func=gen_legendre_poly_2d):

    x = np.linspace(-1,1,50)
    y = np.linspace(-1,1,50)

    V = np.zeros((50,50,I,J))

    if dx:
        i_bounds = range(1,I+1)
    else:
        i_bounds = range(0,I)

    if dy:
        j_bounds = range(1,J+1)
    else:
        j_bounds = range(0,J)

    for i in i_bounds:
        for j in j_bounds:
            pi = i - 1 if dx else i
            pj = j - 1 if dy else j
            #V[:,:,pi,pj] = gen_legendre_poly_2d(i,j,x,y, dx=dx, dy=dy, intx=intx, inty=inty)
            V[:,:,pi,pj] = gen_func(i,j,x,y,dx=dx,dy=dy,intx=intx,inty=inty)

    f, axarr = plt.subplots(I, J)
    for i in range(I):
        for j in range(J):
            axarr[i,j].imshow(V[:,:,i,j])
            axarr[i,j].set_axis_off()

    plt.show()

    Q = np.einsum('abcd, abfg', V, V).reshape((-1, I*J))
    W = np.einsum('abcd, ab', V, M).reshape((I*J))

    a = np.linalg.solve(Q, W)
    V = V.reshape((50,50,I*J))

    plt.imshow(np.einsum('abc,c', V, a))
    plt.show()

    Vn = np.zeros((50, 50, I, J))

    for i in i_bounds:
        for j in range(J):
            Vn[:,:,i-1,j] = gen_func(i,j,x,y)

    f, axarr = plt.subplots(I, J)
    for i in range(I):
        for j in range(J):
            axarr[i,j].imshow(Vn[:,:,i,j])
            axarr[i,j].set_axis_off()

    plt.show()

    Vn = Vn.reshape((50,50,I*J))

    plt.imshow(np.einsum('abc,c', Vn, a))
    plt.show()


def read_data():

    df = pd.read_pickle("/home/bbales2/magnets/Gd_alldata.p")

    def split_MTs(df, col="Magnetic Field (T)", precision=3):
        return df.groupby(lambda x: np.around(df[col][x], precision))

    temps = np.linspace(235, 370, 100)
    Mhs = []
    Ts = []

    M = {}

    for T, data in split_MTs(df):
        idx = np.argmin(data['Temperature (K)'].values)

        temperature = []
        mags = []

        for temp, mag in zip(data['Temperature (K)'][idx:], data['normalized_moment_cgs'][idx:]):

            temperature.append(temp)
            mags.append(mag)

        idxs = np.argsort(temperature)

        temperature = np.array(temperature)[idxs]
        mags = np.array(mags)[idxs]

        magsi = np.interp(temps, temperature, mags)
        Mhs.append(magsi)
        Ts.append(T)

        M[T] = np.array((temperature, mags))

    idx = np.argsort(Ts)
    Ts = np.array(Ts)[idx]
    Mhs = np.array(Mhs)[idx]

    interp = scipy.interpolate.interp2d(temps, Ts, Mhs)

    Ts2 = np.linspace(min(Ts), max(Ts), 50)
    temps2 = np.linspace(min(temps), max(temps), 50)

    Mhis = interp(temps2, Ts2)

    return Mhis
#%%
if __name__ == "__main__":

    import matplotlib.pyplot as plt

    #Mhis = read_data()

    #Mhis -= Mhis.mean()

    xs = np.linspace(-1,1,50)
    for i in range(6):
        z = gen_legendre_poly(i, xs, integrate = True)
        plt.plot(xs, z)

    plt.show()

    #fit(Mhis,dx=True,inty=True,gen_func=gen_legendre_poly_2d)
    #fit(Mhis,gen_func=gen_legendre_poly_2d, dx=True, inty=True)
    #fit(Mhis,gen_func=gen_fourier_2d)

#%%

import numpy

m = numpy.fft.fft2(Mhis)

alpha = 0.1
y = 1.0

w = 2.0 * numpy.pi * numpy.fft.fftfreq(Mhis.shape[0], 1.0)
wx, wy = numpy.meshgrid(w, w)

out = m * wx / (wy)

out[0, :] = 0

s = numpy.fft.ifft2(out)

plt.imshow(numpy.real(s))
plt.show()
#w2 = w * w
#wx2, wy2 = numpy.meshgrid(w2, w2)