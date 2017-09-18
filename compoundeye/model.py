import numpy as np
from utils import get_microvilli_angle, load_beeeye
from sky import ChromaticitySkyModel


class BeeEye(object):

    def __init__(self, ommatidia):
        self.theta = ommatidia[:, 0]
        self.phi = ommatidia[:, 1]
        self._aop_filter, self._dop_filter = get_microvilli_angle(self.theta, self.phi)
        self._dop_filter[:] = 1.
        self._lum = np.zeros_like(self.theta)
        self._dop = np.ones_like(self._dop_filter)
        self._aop = np.zeros_like(self._aop_filter)

    @property
    def L(self):
        filter_r = self.filter(self._aop, "r")
        filter_g = self.filter(self._aop, "g")
        filter_b = self.filter(self._aop, "b")
        lum_r = self._lum + (1. - self._lum) * .05
        lum_g = self._lum + (1. - self._lum) * .53
        lum_b = self._lum + (1. - self._lum) * .79
        # lum_r = lum_r * ((1. - self._dop_filter) + (self._dop_filter * filter_r))
        # lum_g = lum_g * ((1. - self._dop_filter) + (self._dop_filter * filter_g))
        # lum_b = lum_b * ((1. - self._dop_filter) + (self._dop_filter * filter_b))
        lum_r = lum_r * np.sqrt(np.square(1. - self._dop) + np.square(self._dop * filter_r))
        lum_g = lum_g * np.sqrt(np.square(1. - self._dop) + np.square(self._dop * filter_g))
        lum_b = lum_b * np.sqrt(np.square(1. - self._dop) + np.square(self._dop * filter_b))
        # lum_r = lum_r * np.sqrt(np.square(1. - self._dop_filter) + np.square(self._dop_filter * filter_r))
        # lum_g = lum_g * np.sqrt(np.square(1. - self._dop_filter) + np.square(self._dop_filter * filter_g))
        # lum_b = lum_b * np.sqrt(np.square(1. - self._dop_filter) + np.square(self._dop_filter * filter_b))

        return np.clip(np.concatenate((
            lum_r[..., np.newaxis],
            lum_g[..., np.newaxis],
            lum_b[..., np.newaxis]
        ), axis=-1), 0, 1)

    @property
    def DOP(self):
        return self._dop

    @property
    def AOP(self):
        return self._aop

    def set_sky(self, sky):
        self._lum, self._dop, self._aop = sky.get_features(np.pi/2-self.theta, np.pi-self.phi)

    def filter(self, aop, colour="r"):
        c = {
            "r": (np.sin, np.cos),
            "g": (np.sin, np.cos),
            "b": (np.cos, np.sin)
        }

        if colour in c.keys():
            return np.sqrt(
                np.square(c[colour][0](self._aop_filter) * np.cos(aop)) +
                np.square(c[colour][1](self._aop_filter) * np.sin(aop))
            ) * np.clip(self._dop_filter, 0, 1)
        else:
            raise AttributeError("BeeEye.filter: Unsupported channel!")


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
    l_eye = BeeEye(ommatidia_left)
    r_eye = BeeEye(ommatidia_right)
    l_eye.set_sky(sky)
    r_eye.set_sky(sky)

    # plot result
    s, p = 20, 4
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

    from sky import cubebox, skydome


    def plot_luminance(**kwargs):
        plt.figure("Luminance", figsize=(6, 9))
        ax = plt.subplot(2, 1, 1)
        plt.imshow(kwargs["skydome"])
        ax.set_anchor('W')
        plt.xticks([])
        plt.yticks([])
        plt.subplot(6, 4, 17)
        plt.imshow(kwargs["left"])
        plt.text(32, 40, "left", fontsize="16", fontweight="bold", color="white",
                 horizontalalignment="center", verticalalignment="center")
        plt.xticks([])
        plt.yticks([])
        plt.subplot(6, 4, 18)
        plt.imshow(kwargs["front"])
        plt.text(32, 40, "front", fontsize="16", fontweight="bold", color="white",
                 horizontalalignment="center", verticalalignment="center")
        plt.xticks([])
        plt.yticks([])
        plt.subplot(6, 4, 19)
        plt.imshow(kwargs["right"])
        plt.text(32, 40, "right", fontsize="16", fontweight="bold", color="white",
                 horizontalalignment="center", verticalalignment="center")
        plt.xticks([])
        plt.yticks([])
        plt.subplot(6, 4, 20)
        plt.imshow(kwargs["back"])
        plt.text(32, 40, "back", fontsize="16", fontweight="bold", color="white",
                 horizontalalignment="center", verticalalignment="center")
        plt.xticks([])
        plt.yticks([])
        plt.subplot(6, 4, 14)
        plt.imshow(kwargs["top"])
        plt.text(32, 32, "top", fontsize="16", fontweight="bold", color="white",
                 horizontalalignment="center", verticalalignment="center")
        plt.xticks([])
        plt.yticks([])
        plt.subplot(6, 4, 22)
        plt.imshow(kwargs["bottom"])
        plt.text(32, 32, "bottom", fontsize="16", fontweight="bold", color="white",
                 horizontalalignment="center", verticalalignment="center")
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout(pad=0, w_pad=0, h_pad=0)


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
