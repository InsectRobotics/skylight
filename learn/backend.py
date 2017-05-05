import numpy as np
import tensorflow as tf


def rad2compass(phi, length=8):
    alpha = np.arange(length) * 2 * np.pi / length
    I = np.cos(phi[..., np.newaxis] - alpha)
    return I


def compass2rad(I, length=8):
    alpha = np.arange(length) * 2 * np.pi / length
    x, y = np.sum(I * np.cos(alpha), axis=-1), np.sum(I * np.sin(alpha), axis=-1)
    phi = np.arctan(y / x) % (2 * np.pi)
    return phi


def compass2rad2(I, length=8):
    alpha = np.arange(length) * 2 * np.pi / length
    x, y = np.sum(I * np.cos(alpha), axis=-1), np.sum(I * np.sin(alpha), axis=-1)
    phi = np.arctan2(y, x) % (2 * np.pi)
    return phi


def compass2rad_tf(I, length=8):
    alpha = np.arange(length) * 2 * np.pi / length
    x, y = np.sum(I * np.cos(alpha), axis=-1), np.sum(I * np.sin(alpha), axis=-1)
    phi = tf.atan(y / x) % (2 * np.pi)
    return phi


def angdist(a, b):
    d = np.absolute(a - b)
    d -= tf.cast(d > np.pi, tf.float32) * (2 * np.pi)
    return np.absolute(d)
