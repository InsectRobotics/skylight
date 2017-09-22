import numpy as np
from utils import get_microvilli_angle, load_beeeye
from sky import ChromaticitySkyModel

SkyBlue = np.array([.05, .53, .79])[..., np.newaxis]


class CompoundEye(object):

    def __init__(self, ommatidia):

        # eye specifications (ommatidia topography)
        self.theta = ommatidia[:, 0]
        self.phi = ommatidia[:, 1]
        if ommatidia.shape[1] > 3:
            self._dop_filter = ommatidia[:, 3]
            self._aop_filter = ommatidia[:, 2]
        elif ommatidia.shape[1] > 2:
            self._aop_filter = ommatidia[:, 2]
            _, self._dop_filter = get_microvilli_angle(self.theta, self.phi)
        else:
            self._aop_filter, self._dop_filter = get_microvilli_angle(self.theta, self.phi)
        self._dop_filter[:] = 1.

        # specify the polarisation angle for each channel in the microvilli
        self._channel_filters = {
            "r": np.sin,
            "g": np.sin,
            "b": np.cos
        }

        # the raw receptive information
        self._lum = np.zeros_like(self.theta)
        self._dop = np.ones_like(self._dop_filter)
        self._aop = np.zeros_like(self._aop_filter)

    @property
    def L(self):
        f_rgb = self.filter_rgb(self._aop)
        lum = self._lum + SkyBlue * (1. - self._lum)
        lum = lum * np.sqrt(np.square(1. - self._dop) + np.square(self._dop * f_rgb))

        return np.clip(lum.T, 0, 1)

    @property
    def DOP(self):
        return self._dop

    @property
    def AOP(self):
        return self._aop

    def set_sky(self, sky):
        self._lum, self._dop, self._aop = sky.get_features(np.pi/2-self.theta, np.pi-self.phi)

    def filter_rgb(self, aop):
        f = []
        for c in ['r', 'g', 'b']:
            f.append(self.filter(aop, c))
        return np.array(f)

    def filter(self, aop, colour="r"):
        c = self._channel_filters
        if colour in c.keys():
            d = aop - self._aop_filter
            z = np.sqrt(np.square(np.sin(d)) * 2 + np.square(np.cos(d)))
            return np.absolute(c[colour](d) / z) * np.clip(self._dop_filter, 0, 1)
        else:
            raise AttributeError("CompoundEye.filter: Unsupported channel!")


if __name__ == "__main__":
    from datetime import datetime
    from ephem import city
    import matplotlib.pyplot as plt

    # initialise sky
    obs = city("Edinburgh")
    obs.date = datetime.now()
    sky = ChromaticitySkyModel(observer=obs, nside=1)
    sky.generate()

    # initialise ommatidia features
    ommatidia_left, ommatidia_right = load_beeeye()
    l_eye = CompoundEye(ommatidia_left)
    r_eye = CompoundEye(ommatidia_right)
    l_eye.set_sky(sky)
    r_eye.set_sky(sky)

    # plot result
    s, p = 20, 4
    # plot eye's structure
    if False:
        plt.figure("Compound eyes - Structure", figsize=(15, 21))

        lum_r = l_eye._lum + (1. - l_eye._lum) * .05
        lum_g = l_eye._lum + (1. - l_eye._lum) * .53
        lum_b = l_eye._lum + (1. - l_eye._lum) * .79
        L = np.clip(np.concatenate((
            lum_r[..., np.newaxis],
            lum_g[..., np.newaxis],
            lum_b[..., np.newaxis]
        ), axis=-1), 0, 1)
        plt.subplot(321)
        plt.title("Left")
        plt.scatter(l_eye.phi, l_eye.theta, c=L, marker=".", s=np.power(s, p * np.absolute(l_eye.theta) / np.pi))
        plt.ylabel("epsilon (elevation)")
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi/2, np.pi/2])
        plt.xticks([-np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], [])
        plt.yticks([-np.pi/2, -np.pi/3, -np.pi/6, 0, np.pi/6, np.pi/3, np.pi/2],
                   ["-90", "-60", "-30", "0", "30", "60", "90"])

        lum_r = r_eye._lum + (1. - r_eye._lum) * .05
        lum_g = r_eye._lum + (1. - r_eye._lum) * .53
        lum_b = r_eye._lum + (1. - r_eye._lum) * .79
        L = np.clip(np.concatenate((
            lum_r[..., np.newaxis],
            lum_g[..., np.newaxis],
            lum_b[..., np.newaxis]
        ), axis=-1), 0, 1)
        plt.subplot(322)
        plt.title("Right")
        plt.scatter(r_eye.phi, r_eye.theta, c=L, marker=".", s=np.power(s, p * np.absolute(l_eye.theta) / np.pi))
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi/2, np.pi/2])
        plt.xticks([-np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], [])
        plt.yticks([-np.pi/2, -np.pi/3, -np.pi/6, 0, np.pi/6, np.pi/3, np.pi/2], [])

        plt.subplot(323)
        plt.scatter(l_eye.phi, l_eye.theta, c=l_eye._dop_filter, cmap="Blues", vmin=0, vmax=1,
                    marker=".", s=np.power(s, p * np.absolute(l_eye.theta) / np.pi))
        plt.ylabel("epsilon (elevation)")
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi / 2, np.pi / 2])
        plt.xticks([-np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi], [])
        plt.yticks([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2],
                   ["-90", "-60", "-30", "0", "30", "60", "90"])

        plt.subplot(324)
        plt.scatter(r_eye.phi, r_eye.theta, c=r_eye._dop_filter, cmap="Blues", vmin=0, vmax=1,
                    marker=".", s=np.power(s, p * np.absolute(l_eye.theta) / np.pi))
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi / 2, np.pi / 2])
        plt.xticks([-np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi], [])
        plt.yticks([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2], [])

        plt.subplot(325)
        plt.scatter(l_eye.phi, l_eye.theta, c=l_eye._aop_filter % np.pi, cmap="hsv", vmin=0, vmax=np.pi,
                    marker=".", s=np.power(s, p * np.absolute(l_eye.theta) / np.pi))
        plt.xlabel("alpha (azimuth)")
        plt.ylabel("epsilon (elevation)")
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi / 2, np.pi / 2])
        plt.xticks([-np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi],
                   ["-180", "-135", "-90", "-45", "0", "45", "90", "135", "180"])
        plt.yticks([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2],
                   ["-90", "-60", "-30", "0", "30", "60", "90"])

        plt.subplot(326)
        plt.scatter(r_eye.phi, r_eye.theta, c=r_eye._aop_filter % np.pi, cmap="hsv", vmin=0, vmax=np.pi,
                    marker=".", s=np.power(s, p * np.absolute(l_eye.theta) / np.pi))
        plt.xlabel("alpha (azimuth)")
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi / 2, np.pi / 2])
        plt.xticks([-np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi],
                   ["-180", "-135", "-90", "-45", "0", "45", "90", "135", "180"])
        plt.yticks([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2], [])
    # plot bee's view
    if True:
        plt.figure("Compound eyes - Bee's view", figsize=(15, 21))

        plt.subplot(321)
        plt.title("Left")
        plt.scatter(l_eye.phi, l_eye.theta, c=l_eye.L, marker=".", s=np.power(s, p * np.absolute(l_eye.theta) / np.pi))
        plt.ylabel("epsilon (elevation)")
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi/2, np.pi/2])
        plt.xticks([-np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], [])
        plt.yticks([-np.pi/2, -np.pi/3, -np.pi/6, 0, np.pi/6, np.pi/3, np.pi/2],
                   ["-90", "-60", "-30", "0", "30", "60", "90"])

        plt.subplot(322)
        plt.title("Right")
        plt.scatter(r_eye.phi, r_eye.theta, c=r_eye.L, marker=".", s=np.power(s, p * np.absolute(l_eye.theta) / np.pi))
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi/2, np.pi/2])
        plt.xticks([-np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], [])
        plt.yticks([-np.pi/2, -np.pi/3, -np.pi/6, 0, np.pi/6, np.pi/3, np.pi/2], [])

        plt.subplot(323)
        plt.scatter(l_eye.phi, l_eye.theta, c=l_eye.DOP, cmap="Blues", vmin=0, vmax=1,
                    marker=".", s=np.power(s, p * np.absolute(l_eye.theta) / np.pi))
        plt.ylabel("epsilon (elevation)")
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi / 2, np.pi / 2])
        plt.xticks([-np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi], [])
        plt.yticks([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2],
                   ["-90", "-60", "-30", "0", "30", "60", "90"])

        plt.subplot(324)
        plt.scatter(r_eye.phi, r_eye.theta, c=r_eye.DOP, cmap="Blues", vmin=0, vmax=1,
                    marker=".", s=np.power(s, p * np.absolute(l_eye.theta) / np.pi))
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi / 2, np.pi / 2])
        plt.xticks([-np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi], [])
        plt.yticks([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2], [])

        plt.subplot(325)
        plt.scatter(l_eye.phi, l_eye.theta, c=l_eye.AOP, cmap="hsv", vmin=0, vmax=np.pi,
                    marker=".", s=np.power(s, p * np.absolute(l_eye.theta) / np.pi))
        plt.xlabel("alpha (azimuth)")
        plt.ylabel("epsilon (elevation)")
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi / 2, np.pi / 2])
        plt.xticks([-np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi],
                   ["-180", "-135", "-90", "-45", "0", "45", "90", "135", "180"])
        plt.yticks([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2],
                   ["-90", "-60", "-30", "0", "30", "60", "90"])

        plt.subplot(326)
        plt.scatter(r_eye.phi, r_eye.theta, c=r_eye.AOP, cmap="hsv", vmin=0, vmax=np.pi,
                    marker=".", s=np.power(s, p * np.absolute(l_eye.theta) / np.pi))
        plt.xlabel("alpha (azimuth)")
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi / 2, np.pi / 2])
        plt.xticks([-np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi],
                   ["-180", "-135", "-90", "-45", "0", "45", "90", "135", "180"])
        plt.yticks([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2], [])
    # plot filters
    if False:
        plt.figure("Compound eyes - Filters", figsize=(15, 21))

        plt.subplot(3, 4, 1)
        plt.title("microvilli")
        plt.scatter(l_eye.phi, l_eye.theta, c=l_eye._aop_filter % np.pi, cmap="hsv", vmin=0, vmax=np.pi,
                    marker=".", s=np.power(s, p * np.absolute(l_eye.theta) / np.pi))
        plt.ylabel("epsilon (elevation)")
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi / 2, np.pi / 2])
        plt.xticks([-np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi], [])
        plt.yticks([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2],
                   ["-90", "-60", "-30", "0", "30", "60", "90"])

        plt.subplot(3, 4, 2)
        plt.title("aop")
        plt.scatter(l_eye.phi, l_eye.theta, c=l_eye.AOP, cmap="hsv", vmin=0, vmax=np.pi,
                    marker=".", s=np.power(s, p * np.absolute(l_eye.theta) / np.pi))
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi / 2, np.pi / 2])
        plt.xticks([-np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi], [])
        plt.yticks([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2], [])

        plt.subplot(3, 4, 3)
        plt.title("microvilli")
        plt.scatter(r_eye.phi, r_eye.theta, c=r_eye._aop_filter % np.pi, cmap="hsv", vmin=0, vmax=np.pi,
                    marker=".", s=np.power(s, p * np.absolute(l_eye.theta) / np.pi))
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi / 2, np.pi / 2])
        plt.xticks([-np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi], [])
        plt.yticks([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2], [])

        plt.subplot(3, 4, 4)
        plt.title("aop")
        plt.scatter(r_eye.phi, r_eye.theta, c=r_eye.AOP, cmap="hsv", vmin=0, vmax=np.pi,
                    marker=".", s=np.power(s, p * np.absolute(l_eye.theta) / np.pi))
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi / 2, np.pi / 2])
        plt.xticks([-np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi], [])
        plt.yticks([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2], [])

        l_aop_r = l_eye.filter(l_eye.AOP, colour="r")
        l_aop_g = l_eye.filter(l_eye.AOP, colour="g")
        l_aop_b = l_eye.filter(l_eye.AOP, colour="b")

        plt.subplot(3, 4, 5)
        plt.title("green")
        plt.scatter(l_eye.phi, l_eye.theta, c=l_aop_g, cmap="Greens", vmin=0, vmax=1,
                    marker=".", s=np.power(s, p * np.absolute(l_eye.theta) / np.pi))
        plt.ylabel("epsilon (elevation)")
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi / 2, np.pi / 2])
        plt.xticks([-np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi], [])
        plt.yticks([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2],
                   ["-90", "-60", "-30", "0", "30", "60", "90"])

        plt.subplot(3, 4, 6)
        plt.title("blue")
        plt.scatter(l_eye.phi, l_eye.theta, c=l_aop_b, cmap="Blues", vmin=0, vmax=1,
                    marker=".", s=np.power(s, p * np.absolute(l_eye.theta) / np.pi))
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi / 2, np.pi / 2])
        plt.xticks([-np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi], [])
        plt.yticks([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2], [])

        r_aop_r = r_eye.filter(r_eye.AOP, colour="r")
        r_aop_g = r_eye.filter(r_eye.AOP, colour="g")
        r_aop_b = r_eye.filter(r_eye.AOP, colour="b")

        plt.subplot(3, 4, 7)
        plt.title("green")
        plt.scatter(r_eye.phi, r_eye.theta, c=r_aop_g, cmap="Greens", vmin=0, vmax=1,
                    marker=".", s=np.power(s, p * np.absolute(l_eye.theta) / np.pi))
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi / 2, np.pi / 2])
        plt.xticks([-np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi], [])
        plt.yticks([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2], [])

        plt.subplot(3, 4, 8)
        plt.title("blue")
        plt.scatter(r_eye.phi, r_eye.theta, c=r_aop_b, cmap="Blues", vmin=0, vmax=1,
                    marker=".", s=np.power(s, p * np.absolute(l_eye.theta) / np.pi))
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi / 2, np.pi / 2])
        plt.xticks([-np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi], [])
        plt.yticks([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2], [])

        l_aop = np.clip(np.array([l_aop_r * .05, l_aop_g * .53, l_aop_b * .79]).T, 0, 1)
        l_aop_avg = l_aop.mean(axis=1)

        plt.subplot(3, 4, 9)
        plt.title("avg")
        plt.scatter(l_eye.phi, l_eye.theta, c=l_aop_avg, cmap="Greys", vmin=0, vmax=1,
                    marker=".", s=np.power(s, p * np.absolute(l_eye.theta) / np.pi))
        plt.xlabel("alpha (azimuth)")
        plt.ylabel("epsilon (elevation)")
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi / 2, np.pi / 2])
        plt.xticks([-np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi],
                   ["-180", "-135", "-90", "-45", "0", "45", "90", "135", "180"])
        plt.yticks([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2],
                   ["-90", "-60", "-30", "0", "30", "60", "90"])

        plt.subplot(3, 4, 10)
        plt.title("sin")
        plt.scatter(l_eye.phi, l_eye.theta, c=l_aop,
                    marker=".", s=np.power(s, p * np.absolute(l_eye.theta) / np.pi))
        plt.xlabel("alpha (azimuth)")
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi / 2, np.pi / 2])
        plt.xticks([-np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi],
                   ["-180", "-135", "-90", "-45", "0", "45", "90", "135", "180"])
        plt.yticks([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2], [])

        r_aop = np.array([r_aop_r * .05, r_aop_g * .53, r_aop_b * .79]).T
        r_aop_avg = r_aop.mean(axis=1)

        plt.subplot(3, 4, 11)
        plt.title("avg")
        plt.scatter(r_eye.phi, r_eye.theta, c=r_aop_avg, cmap="Greys", vmin=0, vmax=1,
                    marker=".", s=np.power(s, p * np.absolute(l_eye.theta) / np.pi))
        plt.xlabel("alpha (azimuth)")
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi / 2, np.pi / 2])
        plt.xticks([-np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi],
                   ["-180", "-135", "-90", "-45", "0", "45", "90", "135", "180"])
        plt.yticks([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2], [])

        plt.subplot(3, 4, 12)
        plt.title("sin")
        plt.scatter(r_eye.phi, r_eye.theta, c=r_aop,
                    marker=".", s=np.power(s, p * np.absolute(l_eye.theta) / np.pi))
        plt.xlabel("alpha (azimuth)")
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi / 2, np.pi / 2])
        plt.xticks([-np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi],
                   ["-180", "-135", "-90", "-45", "0", "45", "90", "135", "180"])
        plt.yticks([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2], [])

    # from sky import cubebox, skydome
    #
    #
    # def plot_luminance(**kwargs):
    #     plt.figure("Luminance", figsize=(6, 9))
    #     ax = plt.subplot(2, 1, 1)
    #     plt.imshow(kwargs["skydome"])
    #     ax.set_anchor('W')
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.subplot(6, 4, 17)
    #     plt.imshow(kwargs["left"])
    #     plt.text(32, 40, "left", fontsize="16", fontweight="bold", color="white",
    #              horizontalalignment="center", verticalalignment="center")
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.subplot(6, 4, 18)
    #     plt.imshow(kwargs["front"])
    #     plt.text(32, 40, "front", fontsize="16", fontweight="bold", color="white",
    #              horizontalalignment="center", verticalalignment="center")
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.subplot(6, 4, 19)
    #     plt.imshow(kwargs["right"])
    #     plt.text(32, 40, "right", fontsize="16", fontweight="bold", color="white",
    #              horizontalalignment="center", verticalalignment="center")
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.subplot(6, 4, 20)
    #     plt.imshow(kwargs["back"])
    #     plt.text(32, 40, "back", fontsize="16", fontweight="bold", color="white",
    #              horizontalalignment="center", verticalalignment="center")
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.subplot(6, 4, 14)
    #     plt.imshow(kwargs["top"])
    #     plt.text(32, 32, "top", fontsize="16", fontweight="bold", color="white",
    #              horizontalalignment="center", verticalalignment="center")
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.subplot(6, 4, 22)
    #     plt.imshow(kwargs["bottom"])
    #     plt.text(32, 32, "bottom", fontsize="16", fontweight="bold", color="white",
    #              horizontalalignment="center", verticalalignment="center")
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    #
    #
    # # create cubebox parts
    # L_left, DOP_left, AOP_left = cubebox(sky, "left")
    # L_front, DOP_front, AOP_front = cubebox(sky, "front")
    # L_right, DOP_right, AOP_right = cubebox(sky, "right")
    # L_back, DOP_back, AOP_back = cubebox(sky, "back")
    # L_top, DOP_top, AOP_top = cubebox(sky, "top")
    # L_bottom, DOP_bottom, AOP_bottom = cubebox(sky, "bottom")
    #
    # # create skydome
    # L, DOP, AOP = skydome(sky)
    #
    # # plot cubeboxes
    # plot_luminance(skydome=L,
    #                left=L_left, front=L_front, right=L_right, back=L_back, top=L_top, bottom=L_bottom)

    plt.show()
