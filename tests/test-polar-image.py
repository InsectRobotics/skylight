import matplotlib.pyplot as plt
import numpy as np
import healpy as hp


def plot_polar_contour(values, azimuths, zeniths):

    zeniths = np.array(zeniths)
    values = np.diag(values)
    # values[values == 0] = 1.

    r, theta = np.meshgrid(zeniths, np.array(azimuths))
    print r
    print theta
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    plt.autumn()
    cax = ax.contourf(theta, r, values, 30)
    plt.autumn()
    cb = fig.colorbar(cax)
    cb.set_label("Pixel reflectance")

    return fig, ax, cax

val = np.arange(hp.nside2npix(8))
zen, azi = hp.pix2ang(8, val)

plot_polar_contour(val, azi, zen)
plt.show()
