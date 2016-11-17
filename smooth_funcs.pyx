#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

cimport numpy
import numpy

cpdef numpy.ndarray[numpy.double_t, ndim = 2] getm(int N, double dt, double dh, numpy.ndarray[numpy.double_t, ndim = 2] s, numpy.ndarray[numpy.double_t, ndim = 1] m0):
    cdef numpy.ndarray[numpy.double_t, ndim = 2] m
    cdef int i, j, jh

    m = numpy.zeros((N, N))

    for i in range(N):
        m[i] = m0[i]
        for j in range(N):
            for jh in range(1, j + 1):
                m[i, j] += (s[i + 1, jh] - s[i, jh]) * dt / dh

    return m

cpdef loss(double alpha, int N, double dt, double dh, numpy.ndarray[numpy.double_t, ndim = 2] m, numpy.ndarray[numpy.double_t, ndim = 2] mh, numpy.ndarray[numpy.double_t, ndim = 1] m0, numpy.ndarray[numpy.double_t, ndim = 2] s):
    cdef double losst
    cdef int i, j, jh

    losst = 0.0

    for i in range(N):
        for j in range(N):
            losst += (m[i, j] - mh[i, j]) * (m[i, j] - mh[i, j])
    #losst = numpy.sum((m - mh)**2)

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

cpdef numpy.ndarray[numpy.double_t, ndim = 1] dlossdm0(double alpha, int N, double dt, double dh, numpy.ndarray[numpy.double_t, ndim = 2] m, numpy.ndarray[numpy.double_t, ndim = 2] mh, numpy.ndarray[numpy.double_t, ndim = 1] m0):
    cdef numpy.ndarray[numpy.double_t, ndim = 1] dloss
    cdef numpy.ndarray[numpy.double_t, ndim = 2] tmp
    cdef int i

    tmp = (m - mh)

    #dloss = numpy.sum(2 * (m - mh), axis = 1)
    dloss = numpy.zeros(N)
    for i in range(N):
        for j in range(N):
            dloss[i] += 2 * tmp[i, j]

    for i in range(N):
        if i < N - 1:
            dloss[i] += -2 * alpha * (m0[i + 1] - m0[i]) / dh**2
            dloss[i + 1] += 2 * alpha * (m0[i + 1] - m0[i]) / dh**2

    return dloss

cpdef numpy.ndarray[numpy.double_t, ndim = 2] jacm0(double alpha, int N, double dt, double dh):
    cdef numpy.ndarray[numpy.double_t, ndim = 2] out
    cdef int i

    out = numpy.zeros((N, N))
    for i in range(N):
        for j in range(N):
            out[i, i] += 2.0
        #for j in range(N):
        #    out[i, j] += 2 * dmds[i, j, i, 0]

    for i in range(N):
        if i < N - 1:
            out[i, i + 1] -= 2 * alpha / dh**2
            out[i, i] += 2 * alpha / dh**2

            out[i + 1, i + 1] += 2 * alpha / dh**2
            out[i + 1, i] -= 2 * alpha / dh**2

            #dloss[i] += -2 * alpha * (m0[i + 1] - m0[i]) / dh**2
            #dloss[i + 1] += 2 * alpha * (m0[i + 1] - m0[i]) / dh**2

    return out#dloss

cpdef numpy.ndarray[numpy.double_t, ndim = 2] dlossds(double alpha, int N, double dt, double dh, numpy.ndarray[numpy.double_t, ndim = 2] m, numpy.ndarray[numpy.double_t, ndim = 2] mh, numpy.ndarray[numpy.double_t, ndim = 2] s):
    cdef numpy.ndarray[numpy.double_t, ndim = 2] dloss, tmp
    cdef int i, j, jh

    dloss = numpy.zeros((N + 1, N))

    tmp = (m - mh)

    for i in range(0, N + 1):
        for j in range(N - 1, 0, -1):
            jh = j

            if j < N - 1:
                dloss[i, j] = dloss[i, j + 1]

            if i < N:
                dloss[i, j] += -2 * tmp[i, jh] * dt / dh

            if i > 0:
                dloss[i, j] += 2 * tmp[i - 1, jh] * dt / dh

    for i in range(N):
        for j in range(N):
            dloss[i, j] += -2 * alpha * (s[i + 1, j] - s[i, j]) / dh**2
            dloss[i + 1, j] += 2 * alpha * (s[i + 1, j] - s[i, j]) / dh**2

            if j < N - 1:
                dloss[i, j] += -2 * alpha * (s[i, j + 1] - s[i, j]) / dt**2
                dloss[i, j + 1] += 2 * alpha * (s[i, j + 1] - s[i, j]) / dt**2

    return dloss

cpdef numpy.ndarray[numpy.double_t, ndim = 4] jacs(double alpha, int N, double dt, double dh, numpy.ndarray[numpy.double_t, ndim = 4] dmds):
    cdef numpy.ndarray[numpy.double_t, ndim = 4] out
    cdef int i, j

    out = numpy.zeros((N + 1, N, N + 1, N))

    for i in range(0, N + 1):
        for j in range(N - 1, 0, -1):
            if j < N - 1:
                out[i, j] = out[i, j + 1]

            if i < N:
                out[i, j] += -2 * dmds[i, j] * dt / dh

            if i > 0:
                out[i, j] += 2 * dmds[i - 1, j] * dt / dh

    for i in range(N):
        for j in range(N):
            out[i, j, i + 1, j] -= 2 * alpha / dh**2
            out[i, j, i, j] += 2 * alpha / dh**2

            out[i + 1, j, i + 1, j] += 2 * alpha / dh**2
            out[i + 1, j, i, j] -= 2 * alpha / dh**2

            if j < N - 1:
                out[i, j, i, j + 1] -= 2 * alpha / dh**2
                out[i, j, i, j] += 2 * alpha / dh**2

                out[i, j + 1, i + 1, j] += 2 * alpha / dh**2
                out[i, j + 1, i, j] -= 2 * alpha / dh**2

            #dloss[i, j] += -2 * alpha * (s[i + 1, j] - s[i, j]) / dh**2
            #dloss[i + 1, j] += 2 * alpha * (s[i + 1, j] - s[i, j]) / dh**2

            #if j < N - 1:
            #    dloss[i, j] += -2 * alpha * (s[i, j + 1] - s[i, j]) / dt**2
            #    dloss[i, j + 1] += 2 * alpha * (s[i, j + 1] - s[i, j]) / dt**2

    return out
