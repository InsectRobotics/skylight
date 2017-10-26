import numpy as np
import healpy as hp
import ephem
from colorpy.illuminants import get_blackbody_illuminant
from colorpy.rayleigh import rayleigh_illuminated_spectrum
# from colorpy.illuminants import
from datetime import datetime
from .utils import *


class SkyModel(object):
    sky_type_default = 11
    gradation_default = 5
    indicatrix_default = 4
    turbidity_default = 2
    NSIDE = 32
    VIEW_ROT = (0, 90, 0)

    # Transformation matrix of turbidity to luminance coefficients
    T_L = np.array([[ 0.1787, -1.4630],
                    [-0.3554,  0.4275],
                    [-0.0227,  5.3251],
                    [ 0.1206, -2.5771],
                    [-0.0670,  0.3703]])

    def __init__(self, observer=None, turbidity=-1, gradation=-1, indicatrix=-1, sky_type=-1, nside=NSIDE):
        self.__sun = ephem.Sun()
        self.obs = observer
        if observer is None:
            self.obs = ephem.city("Edinburgh")
            self.obs.date = datetime(2017, 6, 21, 10, 0, 0)
        self.sun.compute(self.obs)
        self.lon, self.lat = sun2lonlat(self.__sun)

        # atmospheric condition
        if 1 <= sky_type <= 15:
            gradation = get_sky_gradation(sky_type)
            indicatrix = get_sky_indicatrix(sky_type)

        if gradation < 0 and indicatrix < 0 and turbidity < 0:
            turbidity = self.turbidity_default

        if turbidity >= 1:  # initialise atmospheric coefficients wrt turbidity
            # A: Darkening or brightening of the horizon
            # B: Luminance gradient near the horizon
            # C: Relative intensity of the circumsolar region
            # D: Width of the circumsolar region
            # E: relative backscattered light
            params = self.luminance_coefficients(turbidity)
            self.A, self.B, self.C, self.D, self.E = params
            self.gradation = get_sky_gradation(params)
            self.indicatrix = get_sky_indicatrix(params)
        else:
            self.gradation = gradation if 1 <= gradation <= 6 else self.gradation_default  # default
            self.indicatrix = indicatrix if 1 <= indicatrix <= 6 else self.indicatrix_default  # default

            # set A and B parameters for gradation luminance
            self.A = STANDARD_PARAMETERS["gradation"][self.gradation]["a"]
            self.B = STANDARD_PARAMETERS["gradation"][self.gradation]["b"]

            # set C, D and E parameters for scattering indicatrix
            self.C = STANDARD_PARAMETERS["indicatrix"][self.indicatrix]["c"]
            self.D = STANDARD_PARAMETERS["indicatrix"][self.indicatrix]["d"]
            self.E = STANDARD_PARAMETERS["indicatrix"][self.indicatrix]["e"]
        sky_type = get_sky_type(self.gradation, self.indicatrix)

        self.turbidity = self.turbidity_from_coefficients(self.A, self.B, self.C, self.D, self.E)
        self.description = get_sky_description(sky_type) if sky_type > 0 else [""]

        # calculate the pixel indices
        i = np.arange(hp.nside2npix(nside))
        # get the longitude and co-latitude with respect to the zenith
        self.theta, self.phi = hp.pix2ang(nside, i)  # return longitude and co-latitude in radians
        # we initialise the sun at the zenith
        # so the angular distance between the sun and every point is equal to their distance from the zenith
        self.theta_s, self.phi_s = self.theta.copy(), self.phi.copy()

        # initialise the luminance features
        self.L = np.zeros_like(self.theta)  # total luminance
        self.L_z = self.zenith_luminance(self.turbidity)  # zenith luminance (K cd/m^2)
        self.si = np.zeros_like(self.theta_s)  # scattering indicatrix
        self.lg = np.zeros_like(self.theta)  # luminance gradation
        self.mask = None

        # initialise the electric field information
        self.E_par = np.zeros_like(self.L)  # the electric wave parallel to the polarisation axis
        self.E_per = np.zeros_like(self.L)  # the electric wave perpendicular to the polarisation axis

        # initialise the polarization features
        self.DOP = np.zeros_like(self.theta)  # Degree of Polarisation
        self.AOP = np.zeros_like(self.theta)  # Angle of Polarisation

        self.nside = nside
        self.is_generated = False

    def generate(self):
        # update the relevant sun position
        self.sun.compute(self.obs)
        self.lon, self.lat = self.sun2lonlat()

        # calculate the angular distance between the sun and every point on the map
        x, y, z = 0, np.rad2deg(self.lat), -np.rad2deg(self.lon)
        self.theta_s, self.phi_s = hp.Rotator(rot=(z, y, x))(self.theta, self.phi)
        self.theta_s, self.phi_s = self.theta_s % np.pi, self.phi_s % (2 * np.pi)

        # calculate luminance of the sky
        self.si = self._scattering_indicatrix(self.theta_s)
        self.lg = self._luminance_gradation(self.theta)
        F_theta = self._relative_luminance(self.theta_s, self.theta)
        F_0 = self._relative_luminance(np.array([self.lat]), np.zeros(1))
        self.L_z = self.zenith_luminance(self.turbidity, self.lat)  # zenith luminance (K cd/m^2)
        self.L = self.L_z * F_theta / F_0  # K cd/m^2
        self.mask = self.L > 0

        # calculate the polarisation features
        self.DOP = self.maximum_degree_of_polarisation() * self._linear_polarisation()
        self.AOP = (self.phi_s + np.pi / 2) % np.pi

        # analyse the electric field components to the parallel and the perpendicular to the polarisation axis
        self.E_par = np.sqrt(self.L) * np.sqrt(self.DOP) * \
            np.array([np.sin(self.AOP), np.cos(self.AOP)])
        self.E_per = np.sqrt(self.L) * np.sqrt(1 - self.DOP) * \
            np.array([np.sin(self.AOP + np.pi / 2), np.cos(self.AOP + np.pi / 2)])

        self.is_generated = True

    def get_features(self, theta, phi):
        # update the relevant sun position
        self.sun.compute(self.obs)
        self.lon, self.lat = self.sun2lonlat()

        # calculate the angular distance between the sun and every point on the map
        x, y, z = 0, np.rad2deg(self.lat), -np.rad2deg(self.lon)
        theta_s, phi_s = hp.Rotator(rot=(z, y, x))(theta, phi)
        theta_s, phi_s = theta_s % np.pi, phi_s % (2 * np.pi)

        # calculate luminance of the sky
        F_theta = self._relative_luminance(theta_s, theta)
        F_0 = self._relative_luminance(np.array([self.lat]), np.zeros(1))
        L_z = self.zenith_luminance(self.turbidity, self.lat)  # zenith luminance (K cd/m^2)
        L = L_z * F_theta / F_0  # K cd/m^2

        # calculate the polarisation features
        DOP = self.maximum_degree_of_polarisation() * self._linear_polarisation(x=theta_s, z=theta)
        AOP = (phi_s + np.pi / 2) % np.pi

        return L / 25., DOP, AOP

    def maximum_degree_of_polarisation(self, c1=.6, c2=4.):
        return np.exp(-(self.turbidity-c1)/c2)

    @property
    def sun(self):
        return self.__sun

    @sun.setter
    def sun(self, value):
        self.__sun = value
        self.lon, self.lat = self.sun2lonlat()

    def sun2lonlat(self, **kwargs):
        return sun2lonlat(self.__sun, **kwargs)

    def _relative_luminance(self, x, z):
        """
        Combines the scattering indicatrix and luminance gradation functions to compute the total luminance observed at
        the given sky element(s).

        :param sky: scalar -- Relative backscattered light (scattering indicatrix)
        :param x: angular distance between the observed element and the sun location [0, pi]
        :param z: angular distance between the observed element and the zenith point [0, pi/2]
        :return:  the total observed luminance (Cd/m^2) at the given element(s)
        """
        phi = self._luminance_gradation(z)
        f = self._scattering_indicatrix(x)
        return f * phi

    def _luminance_gradation(self, z):
        """
        The luminance gradation function relates the luminance of a sky element to its zenith angle.

        :param sky: the sky model
        :param z: angular distance between the observed element and the zenith point -- [0, pi/2]
        :return:  the observed luminance gradation (Cd/m^2) at the given element(s) -- [0, 1] for default parameters
        """
        a, b = self.A, self.B
        phi = np.zeros_like(z)
        # apply border conditions to avoid dividing with zero
        z_p = np.all([z >= 0, z < np.pi / 2], axis=0)
        phi[z_p] = 1. + a * np.exp(b / np.cos(z[z_p]))
        phi[np.isclose(z, np.pi/2)] = 1.
        return phi

    def _scattering_indicatrix(self, x):
        """
        The scattering indicatrix which relates the relative luminance of the sky element
        to its angular distance from the sun.

        :param sky: the sky model
        :param x: angular distance between the observed element and the sun location -- [0, pi]
        :return:  the observed scattering indicatrix at the given element(s) -- [0, inf) for default parameters
        """
        c, d, e = self.C, self.D, self.E
        # return 1. + c * (np.exp(d * x) - np.exp(d * np.pi / 2)) + e * np.square(np.cos(x))
        return 1. + c * np.exp(d * x) + e * np.square(np.cos(x))

    def _linear_polarisation(self, x=None, z=None, c=np.pi/2):
        if x is None:
            x = self.theta_s
        if z is None:
            z = self.theta
        lon, lat = sun2lonlat(self.sun)
        lp = degree_of_polarisation(x, 1.)
        i_prez = self._relative_luminance(x, z)
        i_sun = self._relative_luminance(np.zeros(1), np.array([lat]))
        i_90 = self._relative_luminance(np.ones(1) * np.pi / 2, np.absolute(np.array([lat]) - np.pi / 2))
        i = np.zeros_like(lp)
        i[i_prez > 0] = (1./i_prez[i_prez > 0] - 1./i_sun) * (i_90 * i_sun) / (i_sun - i_90)
        p = 1./c * lp * (z * np.cos(z) + (np.pi/2 - z) * i)
        return np.clip(p, 0, 1)

    @classmethod
    def luminance_coefficients(cls, tau):
        """

        :param tau: turbidity
        :return: A_L, B_L, C_L, D_L, E_L 
        """
        return cls.T_L.dot(np.array([tau, 1.]))

    @classmethod
    def turbidity_from_coefficients(cls, A, B, C, D, E):
        T_T = np.linalg.pinv(cls.T_L)
        tau, c = T_T.dot(np.array([A, B, C, D, E]))

        return tau / c

    @classmethod
    def zenith_luminance(cls, tau, theta_s=0.):
        chi = (4. / 9 - tau / 120.) * (np.pi - 2 * theta_s)
        return (4.0453 * tau - 4.9710) * np.tan(chi) - 0.2155 * tau + 2.4192

    @classmethod
    def plot_sun(cls, sky, fig=1, title="", mode=15, sub=(1, 4, 1), show=False):
        import matplotlib.pyplot as plt

        assert (isinstance(mode, int) and 0 <= mode < 16) or isinstance(mode, basestring),\
            "Mode should be an integer between 0 and 15, or a string of the form 'bbbb' where b is for binary."

        if isinstance(mode, basestring):
            mode = (int(mode[0]) << 3) + (int(mode[1]) << 2) + (int(mode[2]) << 1) + (int(mode[3]) << 0)
        sub2 = sub[2]

        lon, lat = sun2lonlat(sky.sun)
        f = plt.figure(fig, figsize=(15, 5))
        if (mode >> 3) % 2 == 1:
            hp.orthview(sky.theta, rot=cls.VIEW_ROT, min=0, max=np.pi, flip="geo", cmap="Greys", half_sky=True,
                        title="Elevation", unit=r'rad', sub=(sub[0], sub[1], sub2), fig=fig)
            sub2 += 1
        if (mode >> 2) % 2 == 1:
            hp.orthview(sky.phi, rot=cls.VIEW_ROT, min=0, max=2 * np.pi, flip="geo", cmap="Greys", half_sky=True,
                        title="Azimuth", unit=r'rad', sub=(sub[0], sub[1], sub2), fig=fig)
            sub2 += 1
        if (mode >> 1) % 2 == 1:
            hp.orthview(sky.theta_s, rot=cls.VIEW_ROT, min=0, max=np.pi, flip="geo", cmap="Greys", half_sky=True,
                        title="Elevation", unit=r'rad', sub=(sub[0], sub[1], sub2), fig=fig)
            sub2 += 1
        if (mode >> 0) % 2 == 1:
            hp.orthview(sky.phi_s, rot=cls.VIEW_ROT, min=0, max=2 * np.pi, flip="geo", cmap="Greys", half_sky=True,
                        title="Azimuth", unit=r'rad', sub=(sub[0], sub[1], sub2), fig=fig)
            sub2 += 1
        hp.projplot(lat, lon, 'yo')
        f.suptitle(title)

        if show:
            plt.show()

        return f

    @classmethod
    def plot_luminance(cls, sky, fig=2, title="", mode=15, sub=(1, 4, 1), show=False):
        import matplotlib.pyplot as plt

        assert (isinstance(mode, int) and 0 <= mode < 16) or isinstance(mode, basestring),\
            "Mode should be an integer between 0 and 15, or a string of the form 'bbbb' where b is for binary."

        if isinstance(mode, basestring):
            mode = (int(mode[0]) << 3) + (int(mode[1]) << 2) + (int(mode[2]) << 1) + (int(mode[3]) << 0)
        sub2 = sub[2]

        lon, lat = sun2lonlat(sky.sun)
        f = plt.figure(fig, figsize=(5 + 3 * (sub[1] - 1), 5 + 3 * (sub[0] - 1)))
        if (mode >> 3) % 2 == 1:
            hp.orthview(sky.si, rot=cls.VIEW_ROT, min=0, max=12, flip="geo", cmap="Greys", half_sky=True,
                        title="Scattering indicatrix", unit=r'', sub=(sub[0], sub[1], sub2), fig=fig)
            sub2 += 1
        if (mode >> 2) % 2 == 1:
            hp.orthview(sky.lg, rot=cls.VIEW_ROT, min=0, max=3, flip="geo", cmap="Greys", half_sky=True,
                        title="Luminance gradation", unit=r'cd/m^2', sub=(sub[0], sub[1], sub2), fig=fig)
            sub2 += 1
        if (mode >> 1) % 2 == 1:
            hp.orthview(sky.L, rot=cls.VIEW_ROT, min=0, max=30, flip="geo", cmap="Greys", half_sky=True,
                        title="Luminance", unit=r'K cd/m^2', sub=(sub[0], sub[1], sub2), fig=fig)
            sub2 += 1
        if (mode >> 0) % 2 == 1:
            hp.orthview(sky.T, rot=cls.VIEW_ROT, min=0, max=20000, flip="geo", cmap="Greys", half_sky=True,
                        title="Colour temperature", unit=r'K', sub=(sub[0], sub[1], sub2), fig=fig)
            sub2 += 1
        # hp.projplot(lat, lon, 'yo')
        f.suptitle(title)

        if show:
            plt.show()

        return f

    @classmethod
    def plot_polarisation(cls, sky, fig=3, title="", mode=3, sub=(1, 2, 1), show=False):
        import matplotlib.pyplot as plt

        assert (isinstance(mode, int) and 0 <= mode < 4) or isinstance(mode, basestring),\
            "Mode should be an integer between 0 and 3, or a string of the form 'bb' where b is for binary."

        if isinstance(mode, basestring):
            mode = (int(mode[0]) << 1) + (int(mode[1]) << 0)
        sub2 = sub[2]

        lon, lat = sun2lonlat(sky.sun)
        f = plt.figure(fig, figsize=(15, 5))
        if (mode >> 1) % 2 == 1:
            hp.orthview(sky.DOP, rot=cls.VIEW_ROT, min=0, max=1, flip="geo", cmap="Greys", half_sky=True,
                        title="Degree of (linear) Polarisation", unit=r'', sub=(sub[0], sub[1], sub2), fig=fig)
            sub2 += 1
        if (mode >> 0) % 2 == 1:
            hp.orthview(sky.AOP, rot=cls.VIEW_ROT, min=0, max=np.pi, flip="geo", cmap="Greys", half_sky=True,
                        title="Angle of (linear) Polarisation", unit=r'rad', sub=(sub[0], sub[1], sub2), fig=fig)
            sub2 += 1
        hp.projplot(lat, lon, 'yo')
        f.suptitle(title)

        if show:
            plt.show()

        return f

    @classmethod
    def rotate_sky(cls, sky, angle):
        sky.theta, sky.phi = cls.rotate(sky.theta, sky.phi, angle)
        # sky.lon, sky.lat = rot(sky.lon, sky.lat)
        return sky

    @classmethod
    def rotate(cls, theta, phi, angle):
        return hp.Rotator(rot=(angle, 0, 0))(theta, phi)


class BlackbodySkyModel(SkyModel):
    alpha_default = -132.1
    beta_default = 59.77

    def __init__(self, *args, **kwargs):
        super(BlackbodySkyModel, self).__init__(*args, **kwargs)

        # initialise the luminance features
        self.T = np.zeros_like(self.theta)  # colour temperature
        self.W = np.zeros(471)              # wavelengths
        self.S = np.zeros((self.theta.shape[0], self.W.shape[0]))  # spectrum

    def generate(self, show=False):
        super(BlackbodySkyModel, self).generate()

        # calculate luminance of the sky
        self.T = self.colour_temperature(self.L * 1000.)
        for i, t in enumerate(self.T[self.mask]):
            b = rayleigh_illuminated_spectrum(get_blackbody_illuminant(t))
            self.S[i, :] = b[:, 1]
        self.W[:] = b[:, 0]

        if show:
            title = "Gradation towards zenith: {:d} | Scattering indicatrix: {:d} | {}".format(self.gradation,
                                                                                               self.indicatrix,
                                                                                               self.description[-1])
            self.plot_luminance(self, fig=2, title=title, show=True)
            # self.plot_polarisation(self, fig=3, show=True)

    @classmethod
    def colour_temperature(cls, L, alpha=alpha_default, beta=beta_default):
        """
        The temperature of the colour in the given sky element(s)
        
        :param L: the observed luminance
        :param alpha: constant shifting of temperature
        :param beta: linear transformation of the temperature
        :return: the temperature of the colour (MK^(-1)) with respect to its luminance
        """
        gamma = cls.correlated_colour_temperature(L, alpha, beta)
        T = np.zeros_like(gamma)
        # compute the temperature of the colour only for the positive values of luminance (set zero values for the rest)
        T[gamma > 0] = (10 ** 6) / gamma[gamma > 0]
        return T

    @classmethod
    def correlated_colour_temperature(cls, L, alpha=alpha_default, beta=beta_default):
        """
        The correlated colour temperature expressed in Remeks (MK^(-1)).
        
        :param L: the observed luminance (cd/m^2)
        :param alpha: constant shifting of temperature
        :param beta: linear transformation of the temperature
        :return: the correlated colour temperature (MK^(-1)) with respect to its luminance
        """
        gamma = np.zeros_like(L)
        # compute the temperature of the colour only for the positive values of luminance (set zero values for the rest)
        gamma[L > 0] = -alpha + beta * np.log(L[L > 0])
        return gamma

    @classmethod
    def generate_features(cls, sky):
        # s = sky.L[sky.mask]  # ignore luminance for now and use only the spectrum
        b = sky.B[sky.mask, :]              # spectrum and luminance information (473)
        d = sky.DOP[sky.mask, np.newaxis]   # degree of polarisation (1)
        a = sky.AOP[sky.mask, np.newaxis]   # angle of polarisation (1)

        features = np.concatenate((b, d, a), axis=1)
        return features


class ChromaticitySkyModel(SkyModel):

    # Transformation matrix of turbidity to x chromaticity coefficients
    T_x = np.array([[-0.0193, -0.2592],
                    [-0.0665,  0.0008],
                    [-0.0004,  0.2125],
                    [-0.0641, -0.8989],
                    [-0.0033,  0.0452]])
    # Transformation matrix of turbidity to y chromaticity coefficients
    T_y = np.array([[-0.0167, -0.2608],
                    [-0.0950,  0.0092],
                    [-0.0079,  0.2102],
                    [-0.0441, -1.6537],
                    [-0.0109,  0.0529]])
    # Zenith chomaticity transformation matrix x
    R_x = np.array([[ 0.0017, -0.0037,  0.0021,  0.0000],
                    [-0.0290,  0.0638, -0.0320,  0.0039],
                    [ 0.1169, -0.2120,  0.0605,  0.2589]])
    # Zenith chomaticity transformation matrix y
    R_y = np.array([[ 0.0028, -0.0061,  0.0032,  0.0000],
                    [-0.0421,  0.0897, -0.0415,  0.0052],
                    [ 0.1535, -0.2676,  0.0667,  0.2669]])
    # Spectral radiant power of a D-illuminant and the first two eigenvector functions
    S_D = np.array([[63.4, 65.8, 94.8, 104.8, 105.9, 96.8, 113.9, 125.6, 125.5, 121.3, 121.3, 113.5, 113.1, 110.8,
                     106.5, 108.8, 105.3, 104.4, 100, 96, 95.1, 89.1, 90.5, 90.3, 88.4, 84, 85.1, 81.9, 82.6, 84.9,
                     81.3, 71.9, 74.3, 76.4, 63.3, 71.7, 77, 65.2, 47.7, 68.6, 65],
                    [38.5, 35, 43.4, 46.3, 43.9, 37.1, 36.7, 35.9, 32.6, 27.9, 24.3, 20.1, 16.2, 13.2, 8.6, 6.1, 4.2,
                     1.9, 0, -1.6, -3.5, -3.5, -5.8, -7.2, -8.6, -9.5, -10.9, -10.7, -12, -14, -13.6, -12, -13.3,
                     -12.9, -10.6, -11.6, -12.2, -10.2, -7.8, -11.2, -10.4],
                    [3, 1.2, -1.1, -0.5, -0.7, -1.2, -2.6, -2.9, -2.8, -2.6, -2.6, -1.8, -1.5, -1.3, -1.2, -1, -0.5,
                     -0.3, 0, 0.2, 0.5, 2.1, 3.2, 4.1, 4.7, 5.1, 6.7, 7.3, 8.6, 9.8, 10.2, 8.3, 9.6, 8.5, 7, 7.6, 8,
                     6.7, 5.2, 7.4, 6.8]])
    # wavelengths
    W_D = np.linspace(380, 780, S_D.shape[1], endpoint=True)

    def __init__(self, *args, **kwargs):
        super(ChromaticitySkyModel, self).__init__(*args, **kwargs)

        # initialise chromaticity coordinates
        self.C_x = np.zeros_like(self.L)        # x chromaticity
        self.C_x_z = self.zenith_x(self.turbidity)

        self.C_y = np.zeros_like(self.L)        # y chromaticity
        self.C_y_z = self.zenith_y(self.turbidity)

        # initialise spectrum features
        self.W = self.W_D  # wavelengths
        self.S = np.zeros((self.theta.shape[0], self.S_D.shape[1]))  # spectrum

    def generate(self, show=False):
        super(ChromaticitySkyModel, self).generate()

        lon, lat = self.sun2lonlat()
        F_theta = self._relative_luminance(self.theta_s, self.theta)
        F_0 = self._relative_luminance(np.array([lat]), np.zeros(1))
        self.C_x_z = self.zenith_x(self.turbidity, lat)  # zenith x
        self.C_x = self.C_x_z * F_theta / F_0
        self.C_y_z = self.zenith_y(self.turbidity, lat)  # zenith y
        self.C_y = self.C_y_z * F_theta / F_0
        self.S[self.L > 0, :] = self.spectral_radiance(self.C_x[self.L > 0], self.C_y[self.L > 0])

        if show:
            title = "Gradation towards zenith: {:d} | Scattering indicatrix: {:d} | {}".format(self.gradation,
                                                                                               self.indicatrix,
                                                                                               self.description[-1])
            self.plot_luminance(self, fig=2, title=title, show=True)
            self.plot_polarisation(self, fig=3, show=True)

    @classmethod
    def x_coefficients(cls, tau):
        """

        :param tau: turbidity
        :return: A_x, B_x, C_x, D_x, E_x 
        """
        return cls.T_x.dot(np.array([tau, 1.]))

    @classmethod
    def y_coefficients(cls, tau):
        """

        :param tau: turbidity
        :return: A_y, B_y, C_y, D_y, E_y 
        """
        return cls.T_y.dot(np.array([tau, 1.]))

    @classmethod
    def zenith_x(cls, tau, theta_s=0.):
        return np.array([np.square(tau), tau, 1])\
            .dot(cls.R_x)\
            .dot(np.array([np.power(theta_s, 3), np.square(theta_s), theta_s, 1]))

    @classmethod
    def zenith_y(cls, tau, theta_s=0.):
        return np.array([np.square(tau), tau, 1])\
            .dot(cls.R_y)\
            .dot(np.array([np.power(theta_s, 3), np.square(theta_s), theta_s, 1]))

    @classmethod
    def plot_luminance(cls, sky, fig=2, title="", mode=31, sub=(1, 5, 1), show=False):
        import matplotlib.pyplot as plt

        assert (isinstance(mode, int) and 0 <= mode < 32) or isinstance(mode, basestring),\
            "Mode should be an integer between 0 and 15, or a string of the form 'bbbb' where b is for binary."

        if isinstance(mode, basestring):
            mode = (int(mode[0]) << 3) + (int(mode[1]) << 2) + (int(mode[2]) << 1) + (int(mode[3]) << 0)
        sub2 = sub[2]

        lon, lat = sun2lonlat(sky.sun)
        f = plt.figure(fig, figsize=(5 + 3 * (sub[1] - 1), 5 + 3 * (sub[0] - 1)))
        if (mode >> 4) % 2 == 1:
            hp.orthview(sky.si, rot=cls.VIEW_ROT, min=0, max=12, flip="geo", cmap="Greys", half_sky=True,
                        title="Scattering indicatrix", unit=r'', sub=(sub[0], sub[1], sub2), fig=fig)
            sub2 += 1
        if (mode >> 3) % 2 == 1:
            hp.orthview(sky.lg, rot=cls.VIEW_ROT, min=0, max=3, flip="geo", cmap="Greys", half_sky=True,
                        title="Luminance gradation", unit=r'cd/m^2', sub=(sub[0], sub[1], sub2), fig=fig)
            sub2 += 1
        if (mode >> 2) % 2 == 1:
            hp.orthview(sky.L, rot=cls.VIEW_ROT, min=0, max=30, flip="geo", cmap="Greys", half_sky=True,
                        title="Luminance", unit=r'K cd/m^2', sub=(sub[0], sub[1], sub2), fig=fig)
            sub2 += 1
        if (mode >> 1) % 2 == 1:
            hp.orthview(sky.C_x, rot=cls.VIEW_ROT, min=0, max=1.5, flip="geo", cmap="Greys", half_sky=True,
                        title="Chromaticity x", sub=(sub[0], sub[1], sub2), fig=fig)
            sub2 += 1
        if (mode >> 0) % 2 == 1:
            hp.orthview(sky.C_y, rot=cls.VIEW_ROT, min=0, max=1.5, flip="geo", cmap="Greys", half_sky=True,
                        title="Chromaticity y", sub=(sub[0], sub[1], sub2), fig=fig)
            sub2 += 1
        # hp.projplot(lat, lon, 'yo')
        f.suptitle(title)

        if show:
            plt.show()

        return f

    @classmethod
    def spectral_radiance(cls, x, y):
        return cls.S_D[0] +\
               cls.M1(x, y)[:, np.newaxis] * cls.S_D[1] +\
               cls.M2(x, y)[:, np.newaxis] * cls.S_D[2]

    @classmethod
    def M1(cls, x, y):
        return (-1.3515 - 1.7703 * x + 5.9114 * y) / (0.0241 + 0.2562 * x - 0.7341 * y)

    @classmethod
    def M2(cls, x, y):
        return (0.0300 - 31.4424 * x + 30.0717 * y) / (0.0241 + 0.2562 * x - 0.7341 * y)

    @classmethod
    def generate_features(cls, sky):
        l = sky.L[sky.mask, np.newaxis]     # luminance information (1)
        x = sky.C_x[sky.mask, np.newaxis]   # Chromaticity x information (1)
        y = sky.C_y[sky.mask, np.newaxis]   # Chromaticity y information (1)
        d = sky.DOP[sky.mask, np.newaxis]   # degree of polarisation (1)
        a = sky.AOP[sky.mask, np.newaxis]   # angle of polarisation (1)

        features = np.concatenate((l, x, y, d, a), axis=1)
        return features
