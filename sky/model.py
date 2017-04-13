import numpy as np
import healpy as hp
import ephem
from datetime import datetime
from .utils import *


class SkyModel(object):
    NSIDE = 32
    VIEW_ROT = (0, 90, 0)
    a_default = -1.
    b_default = -.55
    c_default = 10.
    d_default = -3.
    e_default = .45
    alpha_default = -132.1
    beta_default = 59.77

    def __init__(self, observer=None, nside=NSIDE):
        self.sun = ephem.Sun()
        self.obs = observer
        if observer is None:
            self.obs = ephem.city("Edinburgh")
            self.obs.date = datetime(2017, 6, 21, 10, 0, 0)

        # calculate the pixel indices
        i = np.arange(hp.nside2npix(nside))
        # get the longitude and co-latitude with respect to the zenith
        self.theta, self.phi = hp.pix2ang(nside, i)        # return longitude and co-latitude in radians
        # we initialise the sun at the zenith
        # so the angular distance between the sun and every point is equal to their distance from the zenith
        self.theta_s, self.phi_s = self.theta.copy(), self.phi.copy()

        # initialise the scattering indicatrix with zeros
        self.si = np.zeros_like(self.theta_s)   # scattering indicatrix
        self.lg = np.zeros_like(self.theta)     # luminance gradation
        self.L = np.zeros_like(self.theta)      # total luminance
        self.T = np.zeros_like(self.theta)      # colour temperature

        # initialise the polarization features
        self.DOP = np.zeros_like(self.theta)    # Degree of Polarisation
        self.AOP = np.zeros_like(self.theta)    # Angle of Polarisation

    def generate(self, show=False):
        # update the relevant sun position
        self.sun.compute(self.obs)

        # calculate the angular distance between the sun and every point on the map
        lon, lat = sun2lonlat(self.sun, show=show)
        x, y, z = 0, np.rad2deg(lat), 180 + np.rad2deg(lon)
        self.theta_s, self.phi_s = hp.Rotator(rot=(z, y, x))(self.theta, self.phi)
        self.theta_s, self.phi_s = self.theta_s % np.pi, self.phi_s % (2 * np.pi)

        if show:
            self.plot_sun(self, fig=1)

        # calculate luminance of the sky
        self.si = self.scattering_indicatrix(self.theta_s)
        self.lg = self.luminance_gradation(self.theta)
        self.L = self.luminance(self.theta_s, self.theta)
        self.T = self.colour_temperature(self.L)

        if show:
            self.plot_luminance(self, fig=2)

        # calculate the polarisation features
        self.DOP = 2 * rayleigh(self.theta_s)
        self.AOP = (self.phi_s + np.pi/2) % np.pi

        if show:
            self.plot_polarisation(self, fig=3, show=True)

    @classmethod
    def luminance(cls, x, z, a=a_default, b=b_default, c=c_default, d=d_default, e=e_default):
        """
        Combines the scattering indicatrix and luminance gradation functions to compute the total luminance observed at
        the given sky element(s).
        
        :param x: angular distance between the observed element and the sun [0, pi]
        :param z: angular distance between the observed element and the zenith [0, pi/2]
        :param a: scalar -- affects the amplitude of the curve (luminance gradation)
        :param b: scalar -- affects the curvature of the curve (luminance gradation)
        :param c: scalar -- affects the amplitude of the exponential component (scattering indicatrix)
        :param d: scalar -- affects the curvature of the exponential component (scattering indicatrix)
        :param e: scalar -- affects the amplitude of the sinusoidal component (scattering indicatrix)
        :return:  the total observed luminance (Cd/m^2) at the given element(s)
        """
        return cls.scattering_indicatrix(x, c, d, e) * cls.luminance_gradation(z, a, b)

    @classmethod
    def luminance_gradation(cls, z, a=a_default, b=b_default):
        """
        The luminance gradation function relates the luminance of a sky element to its zenith angle.
        
        :param z: angular distance between the observed element and the zenith -- [0, pi/2]
        :param a: scalar -- affects the amplitude of the curve
        :param b: scalar -- affects the curvature of the curve
        :return:  the observed luminance gradation (Cd/m^2) at the given element(s) -- [0, 1] for default parameters
        """
        phi = np.zeros_like(z)
        # apply border conditions to avoid dividing with zero
        z_p = np.all([z >= 0, z < np.pi / 2], axis=0)
        phi[z_p] = 1. + a * np.exp(b / np.cos(z[z_p]))
        phi[np.isclose(z, np.pi/2)] = 1.
        return phi

    @classmethod
    def scattering_indicatrix(cls, x, c=c_default, d=d_default, e=e_default):
        """
        The scattering indicatrix which relates the relative luminance of the sky element
        to its angular distance from the sun.
        
        :param x: angular distance between the observed element and the sun -- [0, pi]
        :param c: scalar -- affects the amplitude of the exponential component
        :param d: scalar -- affects the curvature of the exponential component
        :param e: scalar -- affects the amplitude of the sinusoidal component
        :return:  the observed scattering indicatrix at the given element(s) -- [0, inf) for default parameters
        """
        return 1. + c * (np.exp(d * x) - np.exp(d * np.pi/2)) + e * np.square(np.cos(x))

    @classmethod
    def colour_temperature(cls, L, alpha=alpha_default, beta=beta_default):
        """
        The temperature of the colour in the given sky element(s)
        
        :param L: the observed luminance
        :param alpha: constant shifting of temperature
        :param beta: linear transformation of the temperature
        :return: the temperature of the colour (MK^(-1)) with respect to its luminance
        """
        ct = np.zeros_like(L)
        # compute the temperature of the colour only for the positive values of luminance (set zero values for the rest)
        ct[L > 0] = -alpha + beta * np.log(L[L > 0])
        return ct

    @classmethod
    def plot_sun(cls, sky, fig=1, show=False):
        import matplotlib.pyplot as plt

        lon, lat = sun2lonlat(sky.sun)
        f = plt.figure(fig, figsize=(15, 5))
        hp.orthview(sky.theta, rot=cls.VIEW_ROT, min=0, max=np.pi, flip="geo", cmap="Greys", half_sky=True,
                    title="Elevation", unit=r'rad', sub=(1, 4, 1), fig=1)
        hp.orthview(sky.phi, rot=cls.VIEW_ROT, min=0, max=2 * np.pi, flip="geo", cmap="Greys", half_sky=True,
                    title="Azimuth", unit=r'rad', sub=(1, 4, 2), fig=1)
        hp.orthview(sky.theta_s, rot=cls.VIEW_ROT, min=0, max=np.pi, flip="geo", cmap="Greys", half_sky=True,
                    title="Elevation", unit=r'rad', sub=(1, 4, 3), fig=1)
        hp.orthview(sky.phi_s, rot=cls.VIEW_ROT, min=0, max=2 * np.pi, flip="geo", cmap="Greys", half_sky=True,
                    title="Azimuth", unit=r'rad', sub=(1, 4, 4), fig=1)
        hp.projplot(lat, lon, 'yo')

        if show:
            plt.show()

        return f

    @classmethod
    def plot_luminance(cls, sky, fig=1, show=False):
        import matplotlib.pyplot as plt

        lon, lat = sun2lonlat(sky.sun)
        f = plt.figure(fig, figsize=(15, 5))
        hp.orthview(sky.si, rot=cls.VIEW_ROT, min=0, max=10, flip="geo", cmap="Greys", half_sky=True,
                    title="Scattering indicatrix", unit=r'', sub=(1, 4, 1), fig=2)
        hp.orthview(sky.lg, rot=cls.VIEW_ROT, min=0, max=1, flip="geo", cmap="Greys", half_sky=True,
                    title="Luminance gradation", unit=r'Cd/m^2', sub=(1, 4, 2), fig=2)
        hp.orthview(sky.L, rot=cls.VIEW_ROT, min=0, max=5.6, flip="geo", cmap="Greys", half_sky=True,
                    title="Luminance", unit=r'Cd/m^2', sub=(1, 4, 3), fig=2)
        hp.orthview(sky.T, rot=cls.VIEW_ROT, min=0, max=257, flip="geo", cmap="Greys", half_sky=True,
                    title="Colour temperature", unit=r'MK^(-1)', sub=(1, 4, 4), fig=2)
        hp.projplot(lat, lon, 'yo')

        if show:
            plt.show()

        return f

    @classmethod
    def plot_polarisation(cls, sky, fig=1, show=False):
        import matplotlib.pyplot as plt

        lon, lat = sun2lonlat(sky.sun)
        f = plt.figure(fig, figsize=(15, 5))
        hp.orthview(sky.DOP, rot=cls.VIEW_ROT, min=0, max=1, flip="geo", cmap="Greys", half_sky=True,
                    title="Degree of (linear) Polarisation", unit=r'', sub=(1, 2, 1), fig=3)
        hp.orthview(sky.AOP, rot=cls.VIEW_ROT, min=0, max=np.pi, flip="geo", cmap="Greys", half_sky=True,
                    title="Angle of (linear) Polarisation", unit=r'rad', sub=(1, 2, 2), fig=3)
        hp.projplot(lat, lon, 'yo')

        if show:
            plt.show()

        return f
