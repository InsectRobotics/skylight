import numpy as np


def rad2compass(phi, length=8):
    alpha = np.arange(length) * 2 * np.pi / length
    I = np.cos(phi[..., np.newaxis] - alpha)
    return I


def compass2rad(I, length=8):
    xy = compass2xy(I, length)
    return np.arctan(xy[..., 1] / xy[..., 0]) % (2. * np.pi)


def compass2rad2(I, length=8):
    xy = compass2xy(I, length)
    return np.arctan2(xy[..., 1], xy[..., 0]) % (2. * np.pi)


def compass2xy(I, length=8):
    alpha = np.arange(length) * 2. * np.pi / length
    x = np.sum(I * np.cos(alpha), axis=-1)[..., np.newaxis]
    y = np.sum(I * np.sin(alpha), axis=-1)[..., np.newaxis]
    return np.concatenate((x, y), axis=-1)


def angdist(a, b):
    d = np.absolute(a - b)
    d -= np.float32(d > np.pi) * (2. * np.pi)
    return np.absolute(d)
