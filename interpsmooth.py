import numpy
import matplotlib.pyplot as plt
import pyximport
pyximport.install()
import smooth_funcs
from scipy.optimize import minimize
import pickle
import pandas as pd
import scipy.interpolate

def split_MTs(df, col="Magnetic Field (T)", precision=3):
   return df.groupby(lambda x: numpy.around(df[col][x], precision))

def interpolate_data(df, N):
    M = {}
    Mhs = []
    Ts = []
        
    temp_min = 1e6
    temp_max = 0
    for T, data in split_MTs(df):
        idx = numpy.argmin(data['Temperature (K)'].values)

        tmax = numpy.max(data['Temperature (K)'][idx:].values)
        tmin = numpy.min(data['Temperature (K)'][idx:].values)

        temp_max = max(tmax, temp_max)
        temp_min = min(tmin, temp_min)
    
    temps = numpy.linspace(temp_min, temp_max, N)
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
        magsi = numpy.interp(temps, temperature, mags)
        Mhs.append(magsi)
        Ts.append(T)

        M[T] = numpy.array((temperature, mags))
    
    idxs = numpy.argsort(Ts)
    Ts = numpy.array(Ts)[idxs]
    Mhs = numpy.array(Mhs)[idxs]
    
    # Build a 2d interpolation of temps and magnetic fields
    interp = scipy.interpolate.interp2d(temps, Ts, Mhs)

    Ts2 = numpy.linspace(min(Ts), max(Ts), N)
    #temps2 = numpy.linspace(min(temps), max(temps), N)
    
    # Evaluate interpolation at the data points
    Mhis = interp(temps, Ts2)

    return Mhis

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
mh = numpy.flipud(interpolate_data(df, N))

# Loss function friendly for scipy.minimize
def f_loss(sm, mh = mh, N = N, dt = dt, dh = dh):
    
    # Take stacked vector and reconstruct
    # s and m0
    st = sm.reshape((N+2,N))
    m0 = st[N+1,:]
    sh = st[:N+1,:]

    # Evaluate m and loss
    m = smooth_funcs.getm(N, dt, dh, sh, m0)
    l = smooth_funcs.loss(alpha, N, dt, dh, m, mh, m0, sh)
    return l

# Jacobian function friendly for scipy.minimize
def df_loss(sm, mh = mh, N = N, dt = dt, dh = dh):

    # Take stacked vector and reconstruct
    # s and m0
    st = sm.reshape((N+2,N))
    m0 = st[N+1,:]
    sh = st[:N+1,:]
    
    # Evaluate m, dl/ds, dl/dm0 and restack them
    m = smooth_funcs.getm(N, dt, dh, sh, m0)
    dlds = smooth_funcs.dlossds(alpha, N, dt, dh, m, mh, sh)
    dldm = smooth_funcs.dlossdm0(alpha, N, dt, dh, m, mh, m0)
    dl = numpy.vstack((dlds, dldm))
    return dl.flatten()

# initial guesses
s = numpy.random.randn(N+1,N)
s[:, 0] = 0
m0 = numpy.random.randn(N)

# Stack into a vector for scipy.minimize
sm = numpy.vstack((s, m0))

# Specifiy boundary conditions (bottom of S is 0)
bounds = [(None,None)] * sm.shape[0] * sm.shape[1]
for i in range(N):
    bounds[N * N + i] = (0,0)

res = minimize(f_loss, sm, method='L-BFGS-B', bounds = bounds, jac = df_loss, options={'disp':True})

# Reconstruct s and m0 from vector form
approx_t = res.x.reshape((N+2,N))
sapprox = approx_t[:N+1,:]
m0approx = approx_t[N+1,:]

# Construct m
m = smooth_funcs.getm(N, dt, dh, sapprox, m0approx)

# Plotting
plt.imshow(sapprox, interpolation = 'NONE')
plt.title('s')
plt.colorbar()
plt.show()

plt.imshow(smooth_funcs.getm(N, dt, dh, sapprox, m0approx), interpolation = 'NONE')
plt.title('m')
plt.colorbar()
plt.show()
