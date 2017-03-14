#%%
import numpy
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pickle
import pandas as pd
import scipy.interpolate
import os

os.chdir('/home/bbales2/magnets/')

def split_MTs(df, col="Magnetic Field (T)", precision=3):
   return df.groupby(lambda x: numpy.around(df[col][x], precision))

def interpolate_data(df, N):
    M = {}
    Mhs = []

    hs = []

    temp_min = 1e6
    temp_max = 0
    for T, data in split_MTs(df):
        idx = numpy.argmin(data['Temperature (K)'].values)

        tmax = numpy.max(data['Temperature (K)'][idx:].values)
        tmin = numpy.min(data['Temperature (K)'][idx:].values)

        temp_max = max(tmax, temp_max)
        temp_min = min(tmin, temp_min)

    ts = numpy.linspace(temp_min, temp_max, N)
    # Read in the data
    for T, data in split_MTs(df):
        # Only read in the bottom half
        idx = numpy.argmin(data['Temperature (K)'].values)

        temperature = []
        mags = []
        for temp, mag in zip(data['Temperature (K)'][idx:], data['normalized_moment_cgs'][idx:]):
            temperature.append(temp)
            mags.append(mag)

        idxs = numpy.argsort(temperature)

        temperature = numpy.array(temperature)[idxs]
        mags = numpy.array(mags)[idxs]

        # 1D Interpolate in the direction of temperatures
        magsi = numpy.interp(ts, temperature, mags)
        Mhs.append(magsi)
        hs.append(T)

        M[T] = numpy.array((temperature, mags))

    idxs = numpy.argsort(hs)
    hs = numpy.array(hs)[idxs]
    Mhs = numpy.array(Mhs)[idxs]

    # Build a 2d interpolation of temps and magnetic fields
    #interp = scipy.interpolate.interp2d(ts, hs, Mhs)

    #hs2 = numpy.linspace(min(hs), max(hs), N)
    #temps2 = numpy.linspace(min(temps), max(temps), N)

    # Evaluate interpolation at the data points
    #Mhis = interp(ts, hs2)

    return ts, hs, Mhs#numpy.flipud(Mhis)

# Grid size
N = 50

# regularization parameter
alpha = 0.0001

# data file
f = "Gd_alldata.p"

# step size in H and T
xs = numpy.linspace(0, 1, N)
ys = numpy.linspace(0, 1, N)
dh = xs[1] - xs[0]
dt = ys[1] - ys[0]

# Read in data and build an interpolation
df = pd.read_pickle(f)
ts, hs, mh = interpolate_data(df, N)

ts -= ts.min()
ts /= ts.max()

hs -= hs.min()
hs /= hs.max()

X = numpy.zeros((len(hs), len(ts), 2))
for i in range(mh.shape[0]):
    for j in range(mh.shape[1]):
        X[i, j, :] = (hs[i], ts[j])

X = X.reshape((-1, 2))
y = mh.reshape((-1, 1))

#%%

import GPy
#%%

m_full = GPy.models.GPRegression(X.reshape((-1, 2)), y.reshape(-1, 1))
m_full.optimize('bfgs')
#%%
m_full.plot(projection = '3d', legend = False)
plt.show()
print m_full
#%%
out = m_full.posterior_samples(X)

#%%

from mpl_toolkits.mplot3d import Axes3D

for i in range(out.shape[1]):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X[:, 0], X[:, 1], out[:, i], '*')
    ax.plot(X[:, 0], X[:, 1], y.flatten(), '+')
    plt.show()