#%%

import numpy
import pandas
import os
import matplotlib.pyplot as plt

os.chdir("/home/bbales2/magnets")

df = pandas.read_pickle("FeGe_alldata.p")

NT = 200
NH = 200
temps = numpy.linspace(269, 281, NT)

def split_MTs(df, col="Magnetic Field (T)", precision=3):
   return df.groupby(lambda x: numpy.around(df[col][x], precision))

M = {}

Mhs = []
Ts = []

for T, data in split_MTs(df):
    idx = numpy.argmin(data['Temperature (K)'].values)

    temperatures = []
    mags = []

    temperature = []
    mags = []

    for temp, mag in zip(data['Temperature (K)'][idx:], data['normalized_moment_cgs'][idx:]):
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

import scipy.interpolate

idxs = numpy.argsort(Ts)
Ts = numpy.array(Ts)[idxs]
Mhs = numpy.array(Mhs)[idxs]

plt.imshow(Mhs, interpolation = 'NONE')
plt.show()

interp = scipy.interpolate.interp2d(temps, Ts, Mhs)

Ts2 = numpy.linspace(min(Ts), max(Ts), NH)
temps2 = numpy.linspace(min(temps), max(temps), NT)

Mhis = interp(temps2, Ts2)

plt.imshow(Mhis)
plt.show()