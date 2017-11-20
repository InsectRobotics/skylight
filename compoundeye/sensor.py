import numpy as np
import numpy.linalg as la
import healpy as hp

from model import CompoundEye
from utils import pca_kernel
from sky import ChromaticitySkyModel, sph2vec

LENS_RADIUS = 1  # mm
A_lens = np.pi * np.square(LENS_RADIUS)  # mm ** 2
NB_EN = 8
DEBUG = True


class CompassSensor(CompoundEye):

    def __init__(self, nb_lenses=20, fov=np.deg2rad(60)):

        self.nb_lenses = nb_lenses
        self.fov = fov
        self.S_a = nb_lenses * A_lens  # the surface area of the sensor
        self.R_c = np.sqrt(self.S_a / (2 * np.pi * (1. - np.cos(fov / 2))))  # the radius of the curvature
        self.alpha = self.R_c * np.sin(fov / 2)
        self.height = self.R_c * (1. - np.cos(fov / 2))
        self.learning_rate = 0.1

        if DEBUG:
            print "Number of lenses:              %d" % nb_lenses
            print "Field of view:                %.2f degrees" % np.rad2deg(fov)
            print "Lens radius (r):               %.2f mm" % LENS_RADIUS
            print "Lens surface area (A):         %.2f mm" % A_lens
            print "Sensor area (S_a):            %.2f mm^2" % self.S_a
            print "Radius of the curvature (R_c): %.2f mm" % self.R_c
            print "Sphere area (S):             %.2f mm^2" % (4 * np.pi * np.square(self.R_c))
            print "Sensor height (h):             %.2f mm" % self.height
            print "Surface coverage:              %.2f" % self.coverage

        thetas, phis, self.nside = self.__angles_distribution()
        ommatidia = np.array([thetas.flatten(), phis.flatten()]).T

        super(CompassSensor, self).__init__(ommatidia, central_microvili=(0, 0), noise_factor=.0,
                                            activate_dop_sensitivity=False)

        self._channel_filters.pop("g")
        self._channel_filters.pop("b")

        self.w = np.random.randn(thetas.size, NB_EN)
        self.w_whitening = np.eye(thetas.size)
        self.m = np.zeros(thetas.size)

    @property
    def coverage(self):
        """

        :return: the percentage of the sphere's surface that is covered from lenses
        """
        # the surface area of the complete sphere
        S = 4 * np.pi * np.square(self.R_c)
        return self.S_a / S

    @property
    def L(self):
        x = super(CompassSensor, self).L
        x_max = x.max()
        x_min = x.min()
        return (x - x_min) / (x_max - x_min)

    def update_parameters(self, x, t=None):
        """

        :param x:
        :type x: np.ndarray, ChromaticitySkyModel
        :param t:
        :type t: np.ndarray
        :return:
        """
        if isinstance(x, ChromaticitySkyModel):
            sky_model = x  # type: ChromaticitySkyModel
            x = np.empty((0, self.L.size), dtype=np.float32)
            t = np.empty((0, NB_EN), dtype=np.float32)
            r = self.facing_direction
            for j in xrange(180):
                self.rotate(np.deg2rad(2))
                self.set_sky(sky_model)
                lon = (sky_model.lon + self.facing_direction) % (2 * np.pi)
                lat = sky_model.lat
                x = np.vstack([x, self.L.flatten()])
                t = np.vstack([t, encode_sun(lon, lat)])
            self.facing_direction = r
            self.set_sky(sky_model)

        # self.w = (1. - self.learning_rate) * self.w + self.learning_rate * la.pinv(x).dot(t)
        self.w_whitening = pca_kernel(x)
        self.m = x.mean(axis=0)
        if t is not None:
            self.w = la.pinv(self._pprop(x), 1e-01).dot(t)

        return self._fprop(x)

    def __call__(self, *args, **kwargs):
        if isinstance(args[0], np.ndarray):
            self._lum = args[0]  # type: np.ndarray
        elif isinstance(args[0], ChromaticitySkyModel):
            self.set_sky(args[0])
        else:
            raise AttributeError("Unknown attribute type: %s" % type(args[0]))

        return self._fprop(self.L)

    def _pprop(self, x):
        return (x.reshape((-1, self.nb_lenses)) - self.m).dot(self.w_whitening)

    def _fprop(self, x):
        x = self._pprop(x)
        y = []
        for x0 in x.dot(self.w):
            y.append(decode_sun(x0))
        return np.array(y)

    def __angles_distribution(self, choice="complete_circles"):

        # the number of lenses required to cover a 360 deg surface
        npix_required = int(np.ceil(self.nb_lenses / self.coverage))
        # compute the parameters of the sphere
        nside = 0
        npix = hp.nside2npix(2 ** nside)
        nb_slots_available = int(np.ceil(npix * self.coverage))
        while self.nb_lenses > nb_slots_available:
            nside += 1
            npix = hp.nside2npix(2 ** nside)
            theta, _ = hp.pix2ang(2 ** nside, np.arange(npix))
            nb_slots_available = (theta < self.fov / 2).sum()
        nside = 2 ** nside

        def complete_circles():
            iii = np.arange(nb_slots_available)
            theta, phi = hp.pix2ang(nside, iii)
            u_theta = np.sort(np.unique(theta))
            nb_slots = np.zeros_like(u_theta, dtype=int)
            for j, uth in enumerate(u_theta):
                nb_slots[j] = (theta == uth).sum()

            j = np.zeros(self.nb_lenses, dtype=int)
            k = 0
            if nb_slots.sum() == self.nb_lenses:
                return iii
            else:
                x = []
                start = 0
                for jj, shift in enumerate(nb_slots / 2):
                    shifts = np.append(np.arange(jj % 2, shift, 2), np.arange(jj % 2 + 1, shift, 2))
                    for jjj in shifts:
                        x.append([])
                        x[-1].append(start + jjj)
                        x[-1].append(start + shift + jjj)
                    start += 2 * shift
                x = np.append(np.array(x[0::2]), np.array(x[1::2])).flatten()
                j[:] = x[:self.nb_lenses]
            return j

        choices = {
            "centre": lambda: np.arange(nb_slots_available)[:self.nb_lenses],
            "edge": lambda: np.arange(nb_slots_available)[::-1][:self.nb_lenses],
            "uniform": lambda: np.random.permutation(nb_slots_available)[:self.nb_lenses],
            "random": lambda: np.random.permutation(nb_slots_available)[:self.nb_lenses],
            "complete_circles": complete_circles
        }

        # calculate the pixel indices
        if choice in choices.keys():
            ii = choices[choice]()
        else:
            ii = choices["uniform"]()
        ii = np.sort(ii)
        # get the longitude and co-latitude with respect to the zenith
        theta, phi = hp.pix2ang(nside, ii)  # return longitude and co-latitude in radians

        if DEBUG:
            print ""
            print "Number of lenses required (full sphere): %d" % npix_required
            print "Number of lenses (full sphere):          %d" % npix
            print "Slots available:                         %d" % nb_slots_available
            print "NSIDE:                                   %d" % nside

        return theta, phi, nside

    @classmethod
    def visualise(cls, sensor):
        """

        :param sensor:
        :type sensor: CompassSensor
        :return:
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse, Rectangle
        from sky.utils import sph2vec

        xyz = sph2vec(np.pi/2 - sensor.theta, sensor.phi, sensor.R_c)

        plt.figure("Sensor Design", figsize=(10, 10))

        # top view
        ax_t = plt.subplot2grid((4, 4), (1, 1), colspan=2, rowspan=2, aspect="equal", adjustable='box-forced')
        outline = Ellipse(xy=np.zeros(2),
                          width=2 * sensor.R_c,
                          height=2 * sensor.R_c)
        sensor_outline = Ellipse(xy=np.zeros(2),
                                 width=2 * sensor.alpha + LENS_RADIUS,
                                 height=2 * sensor.alpha + LENS_RADIUS)
        ax_t.add_artist(outline)
        outline.set_clip_box(ax_t.bbox)
        outline.set_alpha(.2)
        outline.set_facecolor("grey")
        ax_t.add_artist(sensor_outline)
        sensor_outline.set_clip_box(ax_t.bbox)
        sensor_outline.set_alpha(.5)
        sensor_outline.set_facecolor("grey")

        for (x, y, z), th, ph, L in zip(xyz.T, sensor.theta, sensor.phi, sensor.L):
            lens = Ellipse(xy=[x, y], width=LENS_RADIUS, height=np.cos(th) * LENS_RADIUS,
                           angle=np.rad2deg(-ph))
            ax_t.add_artist(lens)
            lens.set_clip_box(ax_t.bbox)
            lens.set_facecolor(np.array([0., 0., 1.]) * L)

        ax_t.set_xlim(-sensor.R_c - 2, sensor.R_c + 2)
        ax_t.set_ylim(-sensor.R_c - 2, sensor.R_c + 2)
        ax_t.set_xticklabels([])
        ax_t.set_yticklabels([])

        # side view #1 (x, z)
        ax = plt.subplot2grid((4, 4), (0, 1), colspan=2, aspect="equal", adjustable='box-forced', sharex=ax_t)
        outline = Ellipse(xy=np.zeros(2),
                          width=2 * sensor.R_c,
                          height=2 * sensor.R_c)
        fade_one = Rectangle(xy=[-sensor.R_c, -sensor.R_c],
                             width=2 * sensor.R_c,
                             height=2 * sensor.R_c - sensor.height - LENS_RADIUS)
        ax.add_artist(outline)
        outline.set_clip_box(ax.bbox)
        outline.set_alpha(.5)
        outline.set_facecolor("grey")
        ax.add_artist(fade_one)
        fade_one.set_clip_box(ax.bbox)
        fade_one.set_alpha(.6)
        fade_one.set_facecolor("white")
        for (x, y, z), th, ph, L in zip(xyz.T, sensor.theta, sensor.phi, sensor.L):
            if y > 0:
                continue
            lens = Ellipse(xy=[x, z], width=LENS_RADIUS, height=np.sin(-y / sensor.R_c) * LENS_RADIUS,
                           angle=np.rad2deg(np.arcsin(-x / sensor.R_c)))
            ax.add_artist(lens)
            lens.set_clip_box(ax.bbox)
            lens.set_facecolor(np.array([0., 0., 1.]) * L)

        ax.set_xlim(-sensor.R_c - 2, sensor.R_c + 2)
        ax.set_ylim(0, sensor.R_c + 2)
        ax.set_xticks([])
        ax.set_yticks([])

        # side view #2 (-x, z)
        ax = plt.subplot2grid((4, 4), (3, 1), colspan=2, aspect="equal", adjustable='box-forced', sharex=ax_t)
        outline = Ellipse(xy=np.zeros(2),
                          width=2 * sensor.R_c,
                          height=2 * sensor.R_c)
        fade_one = Rectangle(xy=[-sensor.R_c, -sensor.R_c + sensor.height + LENS_RADIUS],
                             width=2 * sensor.R_c,
                             height=2 * sensor.R_c - sensor.height - LENS_RADIUS)
        ax.add_artist(outline)
        outline.set_clip_box(ax.bbox)
        outline.set_alpha(.5)
        outline.set_facecolor("grey")
        ax.add_artist(fade_one)
        fade_one.set_clip_box(ax.bbox)
        fade_one.set_alpha(.6)
        fade_one.set_facecolor("white")
        for (x, y, z), th, ph, L in zip(xyz.T, sensor.theta, sensor.phi, sensor.L):
            if y < 0:
                continue
            lens = Ellipse(xy=[x, -z], width=LENS_RADIUS, height=np.sin(-y / sensor.R_c) * LENS_RADIUS,
                           angle=np.rad2deg(np.arcsin(x / sensor.R_c)))
            ax.add_artist(lens)
            lens.set_clip_box(ax.bbox)
            lens.set_facecolor(np.array([0., 0., 1.]) * L)

        ax.set_xlim(-sensor.R_c - 2, sensor.R_c + 2)
        ax.set_ylim(-sensor.R_c - 2, 0)
        ax.set_yticks([])

        # side view #3 (y, z)
        ax = plt.subplot2grid((4, 4), (1, 3), rowspan=2, aspect="equal", adjustable='box-forced', sharey=ax_t)
        outline = Ellipse(xy=np.zeros(2),
                          width=2 * sensor.R_c,
                          height=2 * sensor.R_c)
        fade_one = Rectangle(xy=[-sensor.R_c, -sensor.R_c],
                             width=2 * sensor.R_c - sensor.height - LENS_RADIUS,
                             height=2 * sensor.R_c)
        ax.add_artist(outline)
        outline.set_clip_box(ax.bbox)
        outline.set_alpha(.5)
        outline.set_facecolor("grey")
        ax.add_artist(fade_one)
        fade_one.set_clip_box(ax.bbox)
        fade_one.set_alpha(.6)
        fade_one.set_facecolor("white")
        for (x, y, z), th, ph, L in zip(xyz.T, sensor.theta, sensor.phi, sensor.L):
            if x > 0:
                continue
            lens = Ellipse(xy=[z, y], width=LENS_RADIUS, height=np.sin(-x / sensor.R_c) * LENS_RADIUS,
                           angle=np.rad2deg(np.arcsin(y / sensor.R_c)) + 90)
            ax.add_artist(lens)
            lens.set_clip_box(ax.bbox)
            lens.set_facecolor(np.array([0., 0., 1.]) * L)

        ax.set_ylim(-sensor.R_c - 2, sensor.R_c + 2)
        ax.set_xlim(0, sensor.R_c + 2)
        ax.set_yticks([])
        ax.set_xticks([])

        # side view #4 (-y, z)
        ax = plt.subplot2grid((4, 4), (1, 0), rowspan=2, aspect="equal", adjustable='box-forced', sharey=ax_t)
        outline = Ellipse(xy=np.zeros(2),
                          width=2 * sensor.R_c,
                          height=2 * sensor.R_c)
        fade_one = Rectangle(xy=[-sensor.R_c + sensor.height + LENS_RADIUS, -sensor.R_c],
                             width=2 * sensor.R_c - sensor.height - LENS_RADIUS,
                             height=2 * sensor.R_c)
        ax.add_artist(outline)
        outline.set_clip_box(ax.bbox)
        outline.set_alpha(.5)
        outline.set_facecolor("grey")
        ax.add_artist(fade_one)
        fade_one.set_clip_box(ax.bbox)
        fade_one.set_alpha(.6)
        fade_one.set_facecolor("white")
        for (x, y, z), th, ph, L in zip(xyz.T, sensor.theta, sensor.phi, sensor.L):
            if x < 0:
                continue
            lens = Ellipse(xy=[-z, y], width=LENS_RADIUS, height=np.sin(-x / sensor.R_c) * LENS_RADIUS,
                           angle=np.rad2deg(np.arcsin(-y / sensor.R_c)) - 90)
            ax.add_artist(lens)
            lens.set_clip_box(ax.bbox)
            lens.set_facecolor(np.array([0., 0., 1.]) * L)

        ax.set_ylim(-sensor.R_c - 2, sensor.R_c + 2)
        ax.set_xlim(-sensor.R_c - 2, 0)
        ax.set_xticks([])

        plt.tight_layout(pad=0.)

        plt.show()


def encode_sun(lon, lat):
    return np.sin(np.linspace(0, 2 * np.pi, NB_EN, endpoint=False) + lon + np.pi / 2) * lat / (NB_EN / 2.)


def decode_sun(x):
    fund_freq = np.fft.fft(x)[1]
    lon = -np.angle(np.conj(fund_freq))
    lat = np.absolute(fund_freq)
    return lon, lat


def mse(y, t, theta=True, phi=True):
    if theta:
        thy = y[:, 1]
        tht = t[:, 1]
    else:
        thy = np.zeros_like(y[:, 1])
        tht = np.zeros_like(t[:, 1])
    if phi:
        phy = y[:, 0]
        pht = t[:, 0]
    else:
        phy = np.zeros_like(y[:, 0])
        pht = np.zeros_like(t[:, 0])
    v1 = sph2vec(thy, phy)
    v2 = sph2vec(tht, pht)
    return np.rad2deg(np.arccos((v1 * v2).sum(axis=0)).mean())


if __name__ == "__main__":
    from sky import get_seville_observer
    from datetime import datetime

    s = CompassSensor(nb_lenses=60, fov=np.pi/3)
    p = np.zeros(hp.nside2npix(s.nside))
    i = hp.ang2pix(s.nside, s.theta, s.phi)

    # default observer is in Seville (where the data come from)
    observer = get_seville_observer()
    observer.date = datetime.now()

    # create and generate a sky instance
    sky = ChromaticitySkyModel(observer=observer, nside=1)
    sky.generate()

    lon, lat = sky.lon, sky.lat
    print "Reality: Lon = %.2f, Lat = %.2f" % (np.rad2deg(lon), np.rad2deg(lat))
    lon, lat = s.update_parameters(sky)
    print "Prediction: Lon = %.2f, Lat = %.2f" % (np.rad2deg(lon), np.rad2deg(lat))
    s.set_sky(sky)
    CompassSensor.visualise(s)


if __name__ == "__main__2__":
    import matplotlib.pyplot as plt
    from sky import ChromaticitySkyModel, get_seville_observer
    from datetime import datetime

    s = CompassSensor(nb_lenses=12, fov=np.pi/6)
    p = np.zeros(hp.nside2npix(s.nside))
    i = hp.ang2pix(s.nside, s.theta, s.phi)

    # default observer is in Seville (where the data come from)
    observer = get_seville_observer()
    observer.date = datetime.now()

    # create and generate a sky instance
    sky = ChromaticitySkyModel(observer=observer, nside=1)
    sky.generate()

    s.set_sky(sky)
    p[i] = s.L
    # p[i] = s.DOP
    # p[i] = np.rad2deg(s.AOP)
    p_i_max = p[i].max()
    p_i_min = p[i].min()
    p[i] = (p[i] - p_i_min) / (p_i_max - p_i_min)
    hp.orthview(p, rot=(0, 90, 0))
    plt.show()
