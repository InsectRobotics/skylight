import numpy as np
from utils import get_microvilli_angle, load_beeeye
from sky import ChromaticitySkyModel, get_seville_observer


class BeeEye(object):

    def __init__(self, ommatidia):
        self.theta = ommatidia[:, 0]
        self.phi = ommatidia[:, 1]
        self._aop_filter, self._dop_filter = get_microvilli_angle(self.theta, self.phi)
        self._lum = np.zeros_like(self.theta)

    @property
    def L(self):
        return self._lum

    def set_sky(self, sky):
        lum, dop, aop = sky.get_features(self.theta, self.phi)
        filter_r = self.filter(aop, dop, "r")
        filter_g = self.filter(aop, dop, "g")
        filter_b = self.filter(aop, dop, "b")
        lum_r = np.sqrt(np.square(lum) + np.square((1. - lum) * .05))
        lum_g = np.sqrt(np.square(lum) + np.square((1. - lum) * .53))
        lum_b = np.sqrt(np.square(lum) + np.square((1. - lum) * .79))
        # lum_r = lum_r * np.sqrt(np.square(1. - dop) + np.square(dop * filter_r))
        # lum_g = lum_g * np.sqrt(np.square(1. - dop) + np.square(dop * filter_g))
        # lum_b = lum_b * np.sqrt(np.square(1. - dop) + np.square(dop * filter_b))

        self._lum = np.clip(np.concatenate((
            lum_r[..., np.newaxis],
            lum_g[..., np.newaxis],
            lum_b[..., np.newaxis]
        ), axis=-1), 0, 1)

    def filter(self, aop, dop=0, colour="r"):
        c = {
            "r": (np.sin, np.cos),
            "g": (np.cos, np.sin),
            "b": (np.sin, np.cos)
        }

        if colour in c.keys():
            return np.sqrt(
                np.square(c[colour][0](self._aop_filter) * np.cos(aop)) +
                np.square(c[colour][0](self._aop_filter) * np.sin(aop))
            ) * np.clip(self._dop_filter, 0, 1)
        else:
            raise AttributeError("BeeEye.filter: Unsupported channel!")


if __name__ == "__main__":
    from datetime import datetime
    import matplotlib.pyplot as plt

    # initialise sky
    obs = get_seville_observer()
    obs.date = datetime.now()
    sky = ChromaticitySkyModel(observer=obs, nside=1)
    sky.generate()

    # initialise ommatidia features
    ommatidia_left, ommatidia_right = load_beeeye()
    l_eye = BeeEye(ommatidia_left)
    r_eye = BeeEye(ommatidia_right)
    r_eye.set_sky(sky)

    # plot result
    plt.scatter(r_eye.phi, r_eye.theta, c=r_eye.L)
    # plt.scatter(r_eye.phi, r_eye.theta, c=dop, cmap="Blues", vmin=0, vmax=1)
    # plt.scatter(r_eye.phi, r_eye.theta, c=aop, cmap="hsv", vmin=0, vmax=np.pi)
    plt.xlabel("alpha (azimuth)")
    plt.ylabel("epsilon (elevation)")
    plt.xlim([-np.pi, np.pi])
    plt.ylim([-np.pi/2, np.pi/2])
    plt.xticks([-np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],
               ["-180", "-135", "-90", "-45", "0", "45", "90", "135", "180"])
    plt.yticks([-np.pi/2, -np.pi/3, -np.pi/6, 0, np.pi/6, np.pi/3, np.pi/2],
               ["-90", "-60", "-30", "0", "30", "60", "90"])
    # plt.colorbar()
    plt.show()
