import numpy as np
import healpy as hp
import ephem
import yaml
from datetime import datetime
from .utils import *

with open("../sky/CIE-standard-parameters.yaml", 'r') as file:
    try:
        STANDARD_PARAMETERS = yaml.load(file)
    except yaml.YAMLError as exc:
        print exc


def get_luminance_params(sky_type, indikatrix=None):
    if indikatrix is None:  # sky_type is the index of the type of sky according to standard parameters
        assert sky_type is None or 1 <= sky_type <= 15, "Type should be in [1, 15]."
        gradation = get_sky_gradation(sky_type)
        indikatrix = get_sky_indikatrix(sky_type)
    else:  # sky_type is the gradation of the sky with respect to the zenith
        gradation = sky_type
        sky_type = get_sky_type(gradation, indikatrix)

    assert gradation is None or 1 <= gradation <= 6, "Gradation should be in [1, 6]."
    assert indikatrix is None or 1 <= indikatrix <= 6, "Indikatrix should be in [1, 6]."

    a = STANDARD_PARAMETERS["gradation"][gradation]["a"]
    b = STANDARD_PARAMETERS["gradation"][gradation]["b"]
    c = STANDARD_PARAMETERS["indikatrix"][indikatrix]["c"]
    d = STANDARD_PARAMETERS["indikatrix"][indikatrix]["d"]
    e = STANDARD_PARAMETERS["indikatrix"][indikatrix]["e"]

    return a, b, c, d, e


def get_sky_description(sky_type, indikatrix=None):
    if indikatrix is not None and 1 <= indikatrix <= 6:
        sky_type = get_sky_type(sky_type, indikatrix)

    if 1 <= sky_type <= 15:
        return STANDARD_PARAMETERS["type"][sky_type - 1]["description"]
    else:
        return None


def get_sky_gradation(sky_type):
    if 1 <= sky_type <= 15:
        return STANDARD_PARAMETERS["type"][sky_type - 1]["gradation"]
    else:
        return -1


def get_sky_indikatrix(sky_type):
    if 1 <= sky_type <= 15:
        return STANDARD_PARAMETERS["type"][sky_type - 1]["indikatrix"]
    else:
        return -1


def get_sky_type(gradation, indikatrix):
    if not (1 <= gradation <= 6):
        return -1
    if not (1 <= indikatrix <= 6):
        return -2

    for sky_type, value in enumerate(STANDARD_PARAMETERS["type"]):
        if value["gradation"] == gradation and value["indikatrix"] == indikatrix:
            return sky_type + 1

    return 0


class SkyModel(object):

    NSIDE = 32
    VIEW_ROT = (0, 90, 0)
    sky_type_default = 11
    gradation_default = 4
    indikatrix_default = 4
    alpha_default = -132.1
    beta_default = 59.77

    def __init__(self, observer=None, gradation=-1, indikatrix=-1, sky_type=-1, nside=NSIDE):
        self.sun = ephem.Sun()
        self.obs = observer
        if observer is None:
            self.obs = ephem.city("Edinburgh")
            self.obs.date = datetime(2017, 6, 21, 10, 0, 0)

        if 1 <= sky_type <= 15:
            gradation = get_sky_gradation(sky_type)
            indikatrix = get_sky_indikatrix(sky_type)
        else:
            gradation = gradation if 1 <= gradation <= 6 else self.gradation_default      # default
            indikatrix = indikatrix if 1 <= indikatrix <= 6 else self.indikatrix_default  # default
            sky_type = get_sky_type(gradation, indikatrix)

        self.gradation = gradation
        self.indikatrix = indikatrix
        self.description = get_sky_description(sky_type) if sky_type > 0 else [""]

        # calculate the pixel indices
        i = np.arange(hp.nside2npix(nside))
        # get the longitude and co-latitude with respect to the zenith
        self.theta, self.phi = hp.pix2ang(nside, i)  # return longitude and co-latitude in radians
        # we initialise the sun at the zenith
        # so the angular distance between the sun and every point is equal to their distance from the zenith
        self.theta_s, self.phi_s = self.theta.copy(), self.phi.copy()

        # initialise the luminance features
        self.si = np.zeros_like(self.theta_s)  # scattering indicatrix
        self.lg = np.zeros_like(self.theta)  # luminance gradation
        self.L = np.zeros_like(self.theta)  # total luminance
        self.T = np.zeros_like(self.theta)  # colour temperature

        # initialise the electric field information
        self.E_par = np.zeros_like(self.L)  # the electric wave parallel to the polarisation axis
        self.E_per = np.zeros_like(self.L)  # the electric wave perpendicular to the polarisation axis

        # initialise the polarization features
        self.DOP = np.zeros_like(self.theta)  # Degree of Polarisation
        self.AOP = np.zeros_like(self.theta)  # Angle of Polarisation

    def generate(self, show=False):
        # update the relevant sun position
        self.sun.compute(self.obs)

        if show:
            print "Gradation towards zenith:", self.gradation, "| Scattering indikatrix:", self.indikatrix, "|", self.description[-1]

        # calculate the angular distance between the sun and every point on the map
        lon, lat = sun2lonlat(self.sun, show=show)
        x, y, z = 0, np.rad2deg(lat), 180 + np.rad2deg(lon)
        self.theta_s, self.phi_s = hp.Rotator(rot=(z, y, x))(self.theta, self.phi)
        self.theta_s, self.phi_s = self.theta_s % np.pi, self.phi_s % (2 * np.pi)

        # calculate luminance of the sky
        self.si = self.scattering_indikatrix(self.theta_s, self.indikatrix)
        self.lg = self.luminance_gradation(self.theta, self.gradation)
        self.L = self.luminance(self.theta_s, self.theta, self.gradation, self.indikatrix)
        self.T = self.colour_temperature(self.L)

        if show:
            self.plot_luminance(self, fig=2, show=True)

        # calculate the polarisation features
        self.DOP = degree_of_polarisation(self.theta_s)
        self.AOP = (self.phi_s + np.pi / 2) % np.pi

        # analyse the electric field components to the parallel and the perpendicular to the polarisation axis
        self.E_par = np.sqrt(self.L) * np.sqrt(self.DOP) * \
            np.array([np.sin(self.AOP), np.cos(self.AOP)])
        self.E_per = np.sqrt(self.L) * np.sqrt(1 - self.DOP) * \
            np.array([np.sin(self.AOP + np.pi / 2), np.cos(self.AOP + np.pi / 2)])

        # if show:
        #     self.plot_polarisation(self, fig=3, show=True)

    @classmethod
    def luminance(cls, x, z, a=gradation_default, b=indikatrix_default, c=None, d=None, e=None):
        """
        Combines the scattering indicatrix and luminance gradation functions to compute the total luminance observed at
        the given sky element(s).
        
        :param x: angular distance between the observed element and the sun [0, pi]
        :param z: angular distance between the observed element and the zenith [0, pi/2]
        :param a: gradation type [1, 6] or (if 'c', 'd' and 'e' are given) scalar -- affects the amplitude of the curve
         (luminance gradation)
        :param b: indikatrix type [1, 6] or (if 'c', 'd' and 'e' are given) scalar -- affects the curvature of the curve
         (luminance gradation)
        :param c: scalar -- affects the amplitude of the exponential component (scattering indikatrix)
        :param d: scalar -- affects the curvature of the exponential component (scattering indicatrix)
        :param e: scalar -- affects the amplitude of the sinusoidal component (scattering indicatrix)
        :return:  the total observed luminance (Cd/m^2) at the given element(s)
        """
        if c is None or d is None or e is None:
            phi = cls.luminance_gradation(z, a)
            f = cls.scattering_indikatrix(x, b)
        else:
            phi = cls.luminance_gradation(z, a, b)
            f = cls.scattering_indikatrix(x, c, d, e)
        return f * phi

    @classmethod
    def luminance_gradation(cls, z, a=gradation_default, b=None):
        """
        The luminance gradation function relates the luminance of a sky element to its zenith angle.
        
        :param z: angular distance between the observed element and the zenith -- [0, pi/2]
        :param a: gradation type [1, 6] or (if 'b' is given) scalar -- affects the amplitude of the curve
        :param b: scalar -- affects the curvature of the curve
        :return:  the observed luminance gradation (Cd/m^2) at the given element(s) -- [0, 1] for default parameters
        """
        if b is None:  # 'a' is the gradation type
            b = STANDARD_PARAMETERS["gradation"][a]["b"]
            a = STANDARD_PARAMETERS["gradation"][a]["a"]
        phi = np.zeros_like(z)
        # apply border conditions to avoid dividing with zero
        z_p = np.all([z >= 0, z < np.pi / 2], axis=0)
        phi[z_p] = 1. + a * np.exp(b / np.cos(z[z_p]))
        phi[np.isclose(z, np.pi / 2)] = 1.
        return phi

    @classmethod
    def scattering_indikatrix(cls, x, c=indikatrix_default, d=None, e=None):
        """
        The scattering indikatrix which relates the relative luminance of the sky element
        to its angular distance from the sun.
        
        :param x: angular distance between the observed element and the sun -- [0, pi]
        :param c: indikatrix type [1, 6] or (if 'd' and 'e' are given) scalar -- affects the amplitude of the exponential
         component
        :param d: scalar -- affects the curvature of the exponential component
        :param e: scalar -- affects the amplitude of the sinusoidal component
        :return:  the observed scattering indikatrix at the given element(s) -- [0, inf) for default parameters
        """
        if d is None or e is None:  # 'c' is the indikatrix type
            d, e = STANDARD_PARAMETERS["indikatrix"][c]["d"], STANDARD_PARAMETERS["indikatrix"][c]["e"]
            c = STANDARD_PARAMETERS["indikatrix"][c]["c"]
        return 1. + c * (np.exp(d * x) - np.exp(d * np.pi / 2)) + e * np.square(np.cos(x))

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
    def plot_luminance(cls, sky, fig=1, title="", show=False):
        import matplotlib.pyplot as plt

        lon, lat = sun2lonlat(sky.sun)
        f = plt.figure(fig, figsize=(15, 5))
        hp.orthview(sky.si, rot=cls.VIEW_ROT, min=0, max=10, flip="geo", cmap="Greys", half_sky=True,
                    title="Scattering indicatrix", unit=r'', sub=(1, 4, 1), fig=2)
        hp.orthview(sky.lg, rot=cls.VIEW_ROT, min=0, max=10, flip="geo", cmap="Greys", half_sky=True,
                    title="Luminance gradation", unit=r'Cd/m^2', sub=(1, 4, 2), fig=2)
        hp.orthview(sky.L, rot=cls.VIEW_ROT, min=0, max=5.6, flip="geo", cmap="Greys", half_sky=True,
                    title="Luminance", unit=r'Cd/m^2', sub=(1, 4, 3), fig=2)
        hp.orthview(sky.T, rot=cls.VIEW_ROT, min=0, max=257, flip="geo", cmap="Greys", half_sky=True,
                    title="Colour temperature", unit=r'MK^(-1)', sub=(1, 4, 4), fig=2)
        hp.projplot(lat, lon, 'yo')
        f.suptitle(title)

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


class SkyPoint(object):
    def __init__(self, altitude, azimuth):
        self.altitude = altitude
        self.azimuth = azimuth


class PolarisedPoint(SkyPoint):
    def __init__(self, altitude, azimuth, colour,
                 lin_direction=0., lin_degree=0., cir_direction=0., cir_degree=0., elipticity=0.):
        super(PolarisedPoint, self).__init__(altitude, azimuth)
        self.colour = colour
        self.lin_direction = lin_direction
        self.lin_degree = lin_degree
        self.cir_direction = cir_direction
        self.cir_degree = cir_degree
        self.elipticity = elipticity


class Polariser(object):

    def __init__(self, angle=0.):
        self.angle = angle
        self.R = rotation_matrix(angle)

    def rotate(self, angle):
        return self.fix_angle(self.angle + angle)

    def fix_angle(self, angle):
        self.angle = angle
        self.R = rotation_matrix(angle)
        return self

    def apply(self, sky):
        """
        Applies the polarisation filter and return the relative intensity that passes through it
        
        :param sky: the sky model
        :return: the percentage of the light that passes through
        """
        # the intensity that the observer would perceive before the filter
        I_0 = (np.square(sky.E_par) + np.square(sky.E_per)).sum(axis=0)

        # the intensity after applying the polarisation filter
        I_1 = np.square(np.dot(self.R, sky.E_par)[1])

        # compute the relevant light intensity
        I = np.zeros_like(I_0)
        I[I_0 > 0] = I_1[I_0 > 0] / I_0[I_0 > 0]

        return I
