import numpy as np
import healpy as hp
import ephem
import matplotlib.pyplot as plt
from datetime import datetime
from .utils import *

eps = np.finfo(float).eps

# Transformation matrix of turbidity to luminance coefficients
T_L = np.array([[ 0.1787, -1.4630],
                [-0.3554,  0.4275],
                [-0.0227,  5.3251],
                [ 0.1206, -2.5771],
                [-0.0670,  0.3703]])


class Sky(object):

    def __init__(self, theta_s=0., phi_s=0.):
        self.__a, self.__b, self.__c, self.__d, self.__e = 0., 0., 0., 0., 0.
        self.tau_L = 2.
        self.__c1 = .6
        self.__c2 = 4.
        self.theta_s = theta_s
        self.phi_s = phi_s
        self.verbose = False
        self.eta = np.full(1, np.nan)
        self.theta = np.full(1, np.nan)
        self.phi = np.full(1, np.nan)
        self.__aop = np.full(1, np.nan)
        self.__dop = np.full(1, np.nan)
        self.__y = np.full(1, np.nan)

        self.__is_generated = False

    def __call__(self, *args, **kwargs):

        theta = kwargs.get('theta', self.theta)
        phi = kwargs.get('phi', self.phi)
        uniform_polariser = kwargs.get('uniform_polariser', False)
        noise = kwargs.get('noise', 0.)

        # SKY INTEGRATION
        gamma = np.arccos(np.cos(theta) * np.cos(self.theta_s) +
                          np.sin(theta) * np.sin(self.theta_s) * np.cos(phi - self.phi_s))
        # Intensity
        i_prez = self.L(gamma, theta)
        i_00 = self.L(0., self.theta_s)
        i_90 = self.L(np.pi / 2, np.absolute(self.theta_s - np.pi / 2))
        # influence of sky intensity
        i = (1. / (i_prez + eps) - 1. / (i_00 + eps)) * i_00 * i_90 / (i_00 - i_90 + eps)
        if uniform_polariser:
            y = np.maximum(np.full_like(i_prez, self.Y_z), 0.)
        else:
            y = np.maximum(self.Y_z * i_prez / (i_00 + eps), 0.)  # Illumination

        # Degree of Polarisation
        lp = np.square(np.sin(gamma)) / (1 + np.square(np.cos(gamma)))
        if uniform_polariser:
            p = np.ones_like(lp)
        else:
            p = np.clip(2. / np.pi * self.M_p * lp * (theta * np.cos(theta) + (np.pi / 2 - theta) * i), 0., 1.)

        # Angle of polarisation
        if uniform_polariser:
            a = np.full_like(p, self.phi_s + np.pi)
        else:
            _, a = tilt(self.theta_s, self.phi_s + np.pi, theta, phi)

        # create cloud disturbance
        if noise > 0:
            eta = np.absolute(np.random.randn(*p.shape)) < noise
            if self.verbose:
                print "Noise level: %.4f (%.2f %%)" % (noise, 100. * eta.sum() / float(eta.size))
            p[eta] = 0.  # destroy the polarisation pattern
        else:
            eta = np.zeros(1)

        self.__theta = theta
        self.__phi = phi
        self.__y = y
        self.__dop = p
        self.__aop = a
        self.eta = eta

        self.__is_generated = True

    def L(self, chi, z):
        """
        Prez. et. al. Luminance function

        :param chi:
        :param z:
        :return:
        """
        i = z < (np.pi / 2)
        f = np.zeros_like(z)
        if z.ndim > 0:
            f[i] = (1. + self.A * np.exp(self.B / (np.cos(z[i]) + eps)))
        elif i:
            f = (1. + self.A * np.exp(self.B / (np.cos(z) + eps)))
        phi = (1. + self.C * np.exp(self.D * chi) + self.E * np.square(np.cos(chi)))
        return f * phi

    @property
    def A(self):
        return self.__a

    @property
    def B(self):
        return self.__b

    @property
    def C(self):
        return self.__c

    @property
    def D(self):
        return self.__d

    @property
    def E(self):
        return self.__e

    @property
    def c1(self):
        return self.__c1

    @property
    def c2(self):
        return self.__c2

    @property
    def tau_L(self):
        return self.__tau_L

    @tau_L.setter
    def tau_L(self, value):
        self._update_luminance_coefficients(value)

    @property
    def Y_z(self):
        chi = (4. / 9. - self.tau_L / 120.) * (np.pi - 2 * self.theta_s)
        return (4.0453 * self.tau_L - 4.9710) * np.tan(chi) - 0.2155 * self.tau_L + 2.4192

    @property
    def M_p(self):
        return np.exp(-(self.tau_L - self.c1) / (self.c2 + eps))

    @property
    def I(self):
        return

    @property
    def Y(self):
        return self.__y

    @property
    def DOP(self):
        return self.__dop

    @property
    def AOP(self):
        return self.__aop

    def _update_luminance_coefficients(self, tau_L):
        self.__a, self.__b, self.__c, self.__d, self.__e = T_L.dot(np.array([tau_L, 1.]))
        self._update_turbidity(self.A, self.B, self.C, self.D, self.E)

    def _update_turbidity(self, a, b, c, d, e):
        T_T = np.linalg.pinv(T_L)
        tau_L, c = T_T.dot(np.array([a, b, c, d, e]))
        self.__tau_L = tau_L / c  # turbidity correction

    def copy(self):
        sky = Sky()
        sky.tau_L = self.tau_L
        sky.theta_s = self.theta_s
        sky.phi_s = self.phi_s
        sky.__c1 = self.__c1
        sky.__c2 = self.__c2
        sky.theta_s = self.theta_s
        sky.phi_s = self.phi_s
        sky.verbose = self.verbose
        sky.eta = self.eta
        sky.theta = self.theta
        sky.phi = self.phi
        sky.__aop = self.__aop
        sky.__dop = self.__dop
        sky.__y = self.__y

        sky.__is_generated = False
        return sky


class SkyModel(object):
    sky_type_default = 11
    gradation_default = 5
    indicatrix_default = 4
    turbidity_default = 2
    NSIDE = 1
    VIEW_ROT = (0, 90, 0)

    # Transformation matrix of turbidity to luminance coefficients
    T_L = np.array([[ 0.1787, -1.4630],
                    [-0.3554,  0.4275],
                    [-0.0227,  5.3251],
                    [ 0.1206, -2.5771],
                    [-0.0670,  0.3703]])

    def __init__(self, observer=None, theta_poi=None, phi_poi=None,
                 turbidity=-1, gradation=-1, indicatrix=-1, sky_type=-1, nside=NSIDE):
        self.__sun = ephem.Sun()
        self.__obs = observer
        if observer is None:
            self.__obs = ephem.city("Edinburgh")
            self.__obs.date = datetime(2017, 6, 21, 10, 0, 0)
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

        if theta_poi is None or phi_poi is None:
            # calculate the pixel indices
            i = np.arange(hp.nside2npix(nside))
            # get the longitude and co-latitude with respect to the zenith
            self.__theta_z, self.__phi_z = hp.pix2ang(nside, i)  # return longitude and co-latitude in radians
        else:
            self.__theta_z = theta_poi
            self.__phi_z = phi_poi
        # we initialise the sun at the zenith
        # so the angular distance between the sun and every point is equal to their distance from the zenith
        self.theta_s, self.phi_s = self.theta_z.copy(), self.phi_z.copy()

        self.mask = None

        self.nside = nside
        self.is_generated = False

    def reset(self):

        # initialise the luminance features
        self.mask = None
        self.is_generated = False

    @property
    def obs(self):
        return self.__obs

    @obs.setter
    def obs(self, value):
        self.__obs = value
        self.sun.compute(value)
        self.lon, self.lat = self.sun2lonlat()
        self.is_generated = False

    @property
    def sun(self):
        return self.__sun

    @sun.setter
    def sun(self, value):
        self.__sun = value
        self.lon, self.lat = self.sun2lonlat()
        self.is_generated = False

    @property
    def theta_z(self):
        return self.__theta_z

    @theta_z.setter
    def theta_z(self, value):
        if np.any(np.isnan(value)):
            return
        if self.__theta_z.size != value.size or not np.all(np.isclose(self.__theta_z, value)):
            self.__theta_z = value
            self.theta_s = value.copy()
            self.reset()

    @property
    def phi_z(self):
        return self.__phi_z

    @phi_z.setter
    def phi_z(self, value):
        if np.any(np.isnan(value)):
            return
        if self.__phi_z.size != value.size or not np.all(np.isclose(self.__phi_z, value)):
            self.__phi_z = value
            self.phi_s = value.copy()
            self.reset()

    @property
    def luminance_gradation(self):
        """
        The luminance gradation function relates the luminance of a sky element to its zenith angle.
        :return:  the observed luminance gradation (Cd/m^2) at the given element(s) -- [0, 1] for default parameters
        """
        if not self.is_generated:
            self.generate()

        z = self.theta_z  # angular distance between the observed element and the zenith point -- [0, pi/2]
        a, b = self.A, self.B
        phi = np.zeros_like(z)
        # apply border conditions to avoid dividing with zero
        z_p = np.all([z >= 0, z < np.pi / 2], axis=0)
        phi[z_p] = 1. + a * np.exp(b / np.cos(z[z_p]))
        phi[np.isclose(z, np.pi/2)] = 1.
        return phi

    @property
    def scattering_indicatrix(self):
        """
        The scattering indicatrix which relates the relative luminance of the sky element
        to its angular distance from the sun.
        :return:  the observed scattering indicatrix at the given element(s) -- [0, inf) for default parameters
        """
        if not self.is_generated:
            self.generate()

        chi = self.theta_s  # angular distance between the observed element and the sun location -- [0, pi]
        c, d, e = self.C, self.D, self.E
        # return 1. + c * (np.exp(d * x) - np.exp(d * np.pi / 2)) + e * np.square(np.cos(x))
        return 1. + c * np.exp(d * chi) + e * np.square(np.cos(chi))

    @property
    def prez_luminance(self):
        """
        Combines the scattering indicatrix and luminance gradation functions to compute the total luminance observed at
        the given sky element(s).
        :return:  the total observed luminance (Cd/m^2) at the given element(s)
        """
        if not self.is_generated:
            self.generate()

        phi = self.luminance_gradation
        f = self.scattering_indicatrix
        return f * phi

    @property
    def prez_luminance_0(self):
        """
        :return: the luminance (Cd/m^2) at the zenith point
        """
        if not self.is_generated:
            self.generate()

        theta = 0.
        chi = self.lat
        phi = 1. + self.A * np.exp(self.B / np.cos(theta))
        f = 1. + self.C * np.exp(self.D * chi) + self.E * np.square(np.cos(chi))
        return phi * f

    @property
    def prez_luminance_90(self):
        """
        :return: the luminance (Cd/m^2) in the horizon
        """
        if not self.is_generated:
            self.generate()

        theta = np.pi / 2
        chi = np.absolute(self.lat - np.pi / 2)
        phi = 1. + self.A * np.exp(self.B / np.cos(theta))
        f = 1. + self.C * np.exp(self.D * chi) + self.E * np.square(np.cos(chi))
        return phi * f

    @property
    def L_z(self):
        """
        :return: the zenith luminance (K cd/m^2)
        """
        if not self.is_generated:
            self.generate()

        return self.zenith_luminance(self.turbidity, self.lat)

    @property
    def L(self):
        """
        :return: The luminance of the sky (K cd/m^2)
        """
        if not self.is_generated:
            self.generate()

        # calculate luminance of the sky
        F_theta = self.prez_luminance
        F_0 = self.prez_luminance_0

        L = self.L_z * F_theta / F_0  # K cd/m^2
        self.mask = L > 0
        return L

    @property
    def DOP(self):
        """
        :return: the linear degree of polarisation in the sky
        """
        if not self.is_generated:
            self.generate()

        dop = self.maximum_degree_of_polarisation() * self._linear_polarisation(chi=self.theta_s, z=self.theta_z)
        dop[np.isnan(dop)] = 1.
        return dop
    
    @property
    def AOP(self):
        """
        :return: the angle of linear polarisation in the sky
        """
        if not self.is_generated:
            self.generate()

        return (self.phi_s - self.lon + np.pi/2) % (2 * np.pi)

    @property
    def E_par(self):
        """
        :return: the electric wave parallel to the polarisation axis
        """
        if not self.is_generated:
            self.generate()

        return np.sqrt(self.L) * np.sqrt(self.DOP) * \
               np.array([np.sin(self.AOP), np.cos(self.AOP)])

    @property
    def E_per(self):
        """
        :return: the electric wave perpendicular to the polarisation axis
        """
        if not self.is_generated:
            self.generate()

        return np.sqrt(self.L) * np.sqrt(1 - self.DOP) * \
               np.array([np.sin(self.AOP + np.pi / 2), np.cos(self.AOP + np.pi / 2)])

    def generate(self):
        # update the relevant sun position
        self.sun.compute(self.obs)
        self.lon, self.lat = self.sun2lonlat()

        # calculate the angular distance between the sun and every point on the map
        x, y, z = 0, np.rad2deg(self.lat), -np.rad2deg(self.lon)
        self.theta_s, self.phi_s = hp.Rotator(rot=(z, y, x))(self.theta_z, self.phi_z)
        self.theta_s, self.phi_s = self.theta_s % (2 * np.pi), (self.phi_s + np.pi) % (2 * np.pi) - np.pi

        self.is_generated = True

    def _linear_polarisation(self, chi=None, z=None, c=np.pi/2):
        if chi is None:
            chi = self.theta_s
        if z is None:
            z = self.theta_z
        lp = degree_of_polarisation(chi, 1.)
        i_prez = self.prez_luminance
        i_sun = self.prez_luminance_0
        i_90 = self.prez_luminance_90
        i = np.zeros_like(lp)
        i[i_prez > 0] = np.clip((1./i_prez[i_prez > 0] - 1./i_sun) * (i_90 * i_sun) / (i_sun - i_90), 0, 1)
        p = 1./c * lp * (z * np.cos(z) + (np.pi/2 - z) * (1 - i))
        return np.clip(p, 0, 1)

    def get_features(self, theta, phi):
        self.theta_z = theta
        self.phi_z = phi

        if not self.is_generated:
            self.generate()

        return self.L, self.DOP, self.AOP

    def maximum_degree_of_polarisation(self, c1=.6, c2=4.):
        return np.exp(-(self.turbidity-c1)/c2)

    def sun2lonlat(self, **kwargs):
        return sun2lonlat(self.__sun, **kwargs)

    @staticmethod
    def luminance_coefficients(tau):
        """

        :param tau: turbidity
        :return: A_L, B_L, C_L, D_L, E_L 
        """
        return SkyModel.T_L.dot(np.array([tau, 1.]))

    @staticmethod
    def turbidity_from_coefficients(a, b, c, d, e):
        T_T = np.linalg.pinv(SkyModel.T_L)
        tau, c = T_T.dot(np.array([a, b, c, d, e]))

        return tau / c

    @staticmethod
    def zenith_luminance(tau, theta_s=0.):
        chi = (4. / 9 - tau / 120.) * (np.pi - 2 * theta_s)
        return (4.0453 * tau - 4.9710) * np.tan(chi) - 0.2155 * tau + 2.4192

    @staticmethod
    def plot_sun(sky, fig=1, title="", mode=15, sub=(1, 4, 1), show=False):

        assert (isinstance(mode, int) and 0 <= mode < 16) or isinstance(mode, basestring),\
            "Mode should be an integer between 0 and 15, or a string of the form 'bbbb' where b is for binary."

        if isinstance(mode, basestring):
            mode = (int(mode[0]) << 3) + (int(mode[1]) << 2) + (int(mode[2]) << 1) + (int(mode[3]) << 0)
        sub2 = sub[2]

        lon, lat = sun2lonlat(sky.sun)
        f = plt.figure(fig, figsize=(15, 5))
        if (mode >> 3) % 2 == 1:
            hp.orthview(sky.theta_z, rot=SkyModel.VIEW_ROT, min=0, max=np.pi, flip="geo", cmap="Greys", half_sky=True,
                        title="Elevation", unit=r'rad', sub=(sub[0], sub[1], sub2), fig=fig)
            sub2 += 1
        if (mode >> 2) % 2 == 1:
            hp.orthview(sky.phi_z, rot=SkyModel.VIEW_ROT, min=0, max=2 * np.pi, flip="geo", cmap="Greys", half_sky=True,
                        title="Azimuth", unit=r'rad', sub=(sub[0], sub[1], sub2), fig=fig)
            sub2 += 1
        if (mode >> 1) % 2 == 1:
            hp.orthview(sky.theta_s, rot=SkyModel.VIEW_ROT, min=0, max=np.pi, flip="geo", cmap="Greys", half_sky=True,
                        title="Elevation", unit=r'rad', sub=(sub[0], sub[1], sub2), fig=fig)
            sub2 += 1
        if (mode >> 0) % 2 == 1:
            hp.orthview(sky.phi_s, rot=SkyModel.VIEW_ROT, min=0, max=2 * np.pi, flip="geo", cmap="Greys", half_sky=True,
                        title="Azimuth", unit=r'rad', sub=(sub[0], sub[1], sub2), fig=fig)
            sub2 += 1
        hp.projplot(lat, lon, 'yo')
        f.suptitle(title)

        if show:
            plt.show()

        return f

    @staticmethod
    def plot_luminance(sky, fig=2, title="", mode=15, sub=(1, 4, 1), show=False):
        import matplotlib.pyplot as plt

        assert (isinstance(mode, int) and 0 <= mode < 16) or isinstance(mode, basestring),\
            "Mode should be an integer between 0 and 15, or a string of the form 'bbbb' where b is for binary."

        if isinstance(mode, basestring):
            mode = (int(mode[0]) << 3) + (int(mode[1]) << 2) + (int(mode[2]) << 1) + (int(mode[3]) << 0)
        sub2 = sub[2]

        lon, lat = sun2lonlat(sky.sun)
        f = plt.figure(fig, figsize=(5 + 3 * (sub[1] - 1), 5 + 3 * (sub[0] - 1)))
        if (mode >> 3) % 2 == 1:
            hp.orthview(sky.si, rot=SkyModel.VIEW_ROT, min=0, max=12, flip="geo", cmap="Greys", half_sky=True,
                        title="Scattering indicatrix", unit=r'', sub=(sub[0], sub[1], sub2), fig=fig)
            sub2 += 1
        if (mode >> 2) % 2 == 1:
            hp.orthview(sky.lg, rot=SkyModel.VIEW_ROT, min=0, max=3, flip="geo", cmap="Greys", half_sky=True,
                        title="Luminance gradation", unit=r'cd/m^2', sub=(sub[0], sub[1], sub2), fig=fig)
            sub2 += 1
        if (mode >> 1) % 2 == 1:
            hp.orthview(sky.L, rot=SkyModel.VIEW_ROT, min=0, max=30, flip="geo", cmap="Greys", half_sky=True,
                        title="Luminance", unit=r'K cd/m^2', sub=(sub[0], sub[1], sub2), fig=fig)
            sub2 += 1
        if (mode >> 0) % 2 == 1:
            hp.orthview(sky.T, rot=SkyModel.VIEW_ROT, min=0, max=20000, flip="geo", cmap="Greys", half_sky=True,
                        title="Colour temperature", unit=r'K', sub=(sub[0], sub[1], sub2), fig=fig)
            sub2 += 1
        # hp.projplot(lat, lon, 'yo')
        f.suptitle(title)

        if show:
            plt.show()

        return f

    @staticmethod
    def plot_polarisation(sky, fig=3, title="", mode=3, sub=(1, 2, 1), show=False):
        import matplotlib.pyplot as plt

        assert (isinstance(mode, int) and 0 <= mode < 4) or isinstance(mode, basestring),\
            "Mode should be an integer between 0 and 3, or a string of the form 'bb' where b is for binary."

        if isinstance(mode, basestring):
            mode = (int(mode[0]) << 1) + (int(mode[1]) << 0)
        sub2 = sub[2]

        lon, lat = sun2lonlat(sky.sun)
        f = plt.figure(fig, figsize=(15, 5))
        if (mode >> 1) % 2 == 1:
            hp.orthview(sky.DOP, rot=SkyModel.VIEW_ROT, min=0, max=1, flip="geo", cmap="Greys", half_sky=True,
                        title="Degree of (linear) Polarisation", unit=r'', sub=(sub[0], sub[1], sub2), fig=fig)
            sub2 += 1
        if (mode >> 0) % 2 == 1:
            hp.orthview(sky.AOP, rot=SkyModel.VIEW_ROT, min=0, max=np.pi, flip="geo", cmap="Greys", half_sky=True,
                        title="Angle of (linear) Polarisation", unit=r'rad', sub=(sub[0], sub[1], sub2), fig=fig)
            sub2 += 1
        hp.projplot(lat, lon, 'yo')
        f.suptitle(title)

        if show:
            plt.show()

        return f

    @staticmethod
    def rotate_sky(sky, yaw=0., pitch=0., roll=0., zenith=True):
        if zenith:
            sky.theta_z, sky.phi_z = SkyModel.rotate(sky.theta_z, sky.phi_z, yaw=yaw, pitch=pitch, roll=roll)
        else:
            theta = np.pi / 2 - sky.theta_z
            phi = np.pi - sky.phi_z

            theta, phi = SkyModel.rotate(theta, phi, yaw=yaw, pitch=pitch, roll=roll)
            sky.phi_z = (2 * np.pi - phi) % (2 * np.pi) - np.pi
            sky.theta_z = (3 * np.pi/2 - theta) % (2 * np.pi) - np.pi

        return sky

    @staticmethod
    def rotate(theta, phi, yaw=0., pitch=0., roll=0.):
        if not np.isclose(roll, 0.):
            theta, phi = hp.Rotator(rot=(0., 0., np.rad2deg(roll)))(theta, phi)
        if not np.isclose(pitch, 0.):
            theta, phi = hp.Rotator(rot=(0., np.rad2deg(pitch), 0.))(theta, phi)
        if not np.isclose(yaw, 0.):
            theta, phi = hp.Rotator(rot=(np.rad2deg(yaw), 0., 0.))(theta, phi)
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        phi = (phi + np.pi) % (2 * np.pi) - np.pi
        # if isinstance(theta, np.ndarray) and isinstance(phi, np.ndarray):
        #     phi[theta > np.pi/2]
        return theta, phi

    def copy(self):
        sky = SkyModel()
        sky.__obs = self.__obs.copy()
        sky.__sun = self.__sun.copy()
        sky.lon, sky.lat = self.lon, self.lat
        sky.A, sky.B, sky.C, sky.D, sky.E = self.A, self.B, self.C, self.D, self.E
        sky.gradation = self.gradation
        sky.indicatrix = self.indicatrix
        sky.turbidity = self.turbidity
        sky.description = self.description
        sky.__theta_z = self.__theta_z.copy()
        sky.__phi_z = self.__phi_z.copy()
        sky.theta_s = self.theta_s.copy()
        sky.phi_s = self.phi_s.copy()
        if self.mask is not None:
            sky.mask = self.mask.copy()
        sky.nside = self.nside
        sky.is_generated = False

        return sky
