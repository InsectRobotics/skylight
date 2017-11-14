import numpy as np
import healpy as hp

from model import CompoundEye

LENS_RADIUS = 1  # mm
A_lens = np.pi * np.square(LENS_RADIUS)  # mm ** 2
DEBUG = True


class CompassSensor(CompoundEye):

    def __init__(self, nb_lenses=20, fov=np.deg2rad(60)):

        self.nb_lenses = nb_lenses
        self.fov = fov
        self.S_a = nb_lenses * A_lens  # the surface area of the sensor
        self.R_c = np.sqrt(self.S_a / (2 * np.pi * (1. - np.cos(fov / 2))))  # the radius of the curvature
        self.height = self.R_c * (1. - np.cos(fov / 2))

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

    @property
    def coverage(self):
        """

        :return: the percentage of the sphere's surface that is covered from lenses
        """
        # the surface area of the complete sphere
        S = 4 * np.pi * np.square(self.R_c)
        return self.S_a / S

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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sky import ChromaticitySkyModel, get_seville_observer
    from datetime import datetime

    s = CompassSensor(nb_lenses=1504, fov=np.pi)
    p = np.zeros(hp.nside2npix(s.nside))
    i = hp.ang2pix(s.nside, s.theta, s.phi)

    # default observer is in Seville (where the data come from)
    observer = get_seville_observer()
    observer.date = datetime.now()

    # create and generate a sky instance
    sky = ChromaticitySkyModel(observer=observer, nside=1)
    sky.generate()

    s.set_sky(sky)
    p[i] = s.L[:, 2]
    # p[i] = s.DOP
    # p[i] = np.rad2deg(s.AOP)
    hp.orthview(p, rot=(0, 90, 0))
    plt.show()
