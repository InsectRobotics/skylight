import numpy as np


def rad2compass(phi, length=8):
    alpha = np.arange(length) * 2 * np.pi / length
    I = np.cos(phi[..., np.newaxis] - alpha)
    return I


def compass2rad(I, length=8):
    alpha = np.arange(length) * 2 * np.pi / length
    x0, y0 = (I * np.cos(alpha)).sum(axis=-1), (I * np.sin(alpha)).sum(axis=-1)
    phi = np.arctan(y0 / x0) % (2 * np.pi)
    return phi


def compass2rad2(I, length=8):
    alpha = np.arange(length) * 2 * np.pi / length
    x0, y0 = (I * np.cos(alpha)).sum(axis=-1), (I * np.sin(alpha)).sum(axis=-1)
    phi = np.arctan2(y0, x0) % (2 * np.pi)
    return phi


def angdist(a, b):
    d = np.absolute(a - b)
    d -= (d > np.pi) * (2 * np.pi)
    return np.absolute(d)
