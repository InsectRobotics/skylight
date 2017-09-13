from sky import ChromaticitySkyModel, get_seville_observer
from datetime import datetime, timedelta
from sys import argv
from ephem import city, Sun
from sky import sph2pix, pix2sph, Width as W, Height as H
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
plt.ion()


def vec2sph(vec):
    """
    Transforms a cartessian vector to spherical coordinates.
    :param vec:     the cartessian vector
    :return theta:  elevation
    :return phi:    azimuth
    :return rho:    radius
    """
    rho = la.norm(vec, axis=-1)  # length of the radius

    if vec.ndim == 1:
        vec = vec[np.newaxis, ...]
        if rho == 0:
            rho = 1.
    else:
        rho = np.concatenate([rho[..., np.newaxis]] * vec.shape[1], axis=-1)
        rho[rho == 0] = 1.
    v = vec / rho  # normalised vector

    phi = np.arctan2(v[:, 0], v[:, 1])  # azimuth
    theta = np.arccos(v[:, 2])  # elevation

    # theta, phi = sphadj(theta, phi)  # bound the spherical coordinates
    return np.asarray([theta, phi, rho[:, -1]])


def cubebox_angles(side):
    if side == "left":
        y = np.linspace(1, -1, W, endpoint=False)
        z = np.linspace(1, -1, H, endpoint=False)
        y, z = np.meshgrid(y, z)
        x = -np.ones(W * H)
    elif side == "front":
        x = np.linspace(-1, 1, W, endpoint=False)
        z = np.linspace(1, -1, H, endpoint=False)
        x, z = np.meshgrid(x, z)
        y = -np.ones(W * H)
    elif side == "right":
        y = np.linspace(-1, 1, W, endpoint=False)
        z = np.linspace(1, -1, H, endpoint=False)
        y, z = np.meshgrid(y, z)
        x = np.ones(W * H)
    elif side == "back":
        x = np.linspace(1, -1, W, endpoint=False)
        z = np.linspace(1, -1, H, endpoint=False)
        x, z = np.meshgrid(x, z)
        y = np.ones(W * H)
    elif side == "top":
        x = np.linspace(-1, 1, W, endpoint=False)
        y = np.linspace(1, -1, W, endpoint=False)
        x, y = np.meshgrid(x, y)
        z = np.ones(W * W)
    elif side == "bottom":
        x = np.linspace(-1, 1, W, endpoint=False)
        y = np.linspace(-1, 1, W, endpoint=False)
        x, y = np.meshgrid(x, y)
        z = -np.ones(W * W)
    else:
        x, y, z = np.zeros((3, H * W))
    vec = np.stack([x.reshape(H * W), y.reshape(H * W), z.reshape(H * W)]).T
    theta, phi, _ = vec2sph(vec)
    return theta, phi


def cubebox(sky, side):
    theta, phi = cubebox_angles(side)
    L, DOP, AOP = sky.get_features(theta, phi)
    L = L.reshape((W, H))
    DOP[np.isnan(DOP)] = -1
    DOP = DOP.reshape((W, H))
    AOP = AOP.reshape((W, H))

    L_cube = np.zeros((W, H, 3))
    L_cube[..., 0] = L + (1. - L) * .05  # deep sky blue * .53
    L_cube[..., 1] = L + (1. - L) * .53  # deep sky blue * .81
    L_cube[..., 2] = L + (1. - L) * .79  # deep sky blue * .92
    L_cube = np.clip(L_cube, 0, 1)

    DOP_cube = np.zeros((W, H, 3))
    DOP_cube[..., 0] = DOP * .53 + (1. - DOP)
    DOP_cube[..., 1] = DOP * .81 + (1. - DOP)
    DOP_cube[..., 2] = DOP * 1.0 + (1. - DOP)
    DOP_cube = np.clip(DOP_cube, 0, 1)

    AOP_cube = AOP % np.pi
    AOP_cube = np.clip(AOP_cube, 0, np.pi)

    return L_cube, DOP_cube, AOP_cube


def skydome(sky):
    x, y = np.arange(W), np.arange(H)
    x, y = np.meshgrid(x, y)
    x, y = x.flatten(), y.flatten()
    theta, phi = pix2sph(np.array([x, y]), H, W)
    sky_L, sky_DOP, sky_AOP = sky.get_features(theta, phi)
    sky_L = np.clip(sky_L, 0, 1)
    sky_DOP = np.clip(sky_DOP, 0, 1)

    L = np.zeros((W, H, 3))
    L[x, y, 0] = sky_L + (1. - sky_L) * .05
    L[x, y, 1] = sky_L + (1. - sky_L) * .53
    L[x, y, 2] = sky_L + (1. - sky_L) * .79

    DOP = np.zeros((W, H, 3))
    DOP[x, y, 0] = sky_DOP * .53 + (1. - sky_DOP)
    DOP[x, y, 1] = sky_DOP * .81 + (1. - sky_DOP)
    DOP[x, y, 2] = sky_DOP * 1.0 + (1. - sky_DOP)

    AOP = sky_AOP % np.pi
    AOP = np.clip(AOP, 0, np.pi).reshape((W, H))

    return L, DOP, AOP


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


def plot_DOP(**kwargs):
    plt.figure("Degree of Polarisation", figsize=(6, 9))
    ax = plt.subplot(2, 1, 1)
    plt.imshow(kwargs["skydome"])
    ax.set_anchor('W')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(6, 4, 17)
    plt.imshow(kwargs["left"])
    plt.text(32, 40, "left", fontsize="16", fontweight="bold", color="black",
             horizontalalignment="center", verticalalignment="center")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(6, 4, 18)
    plt.imshow(kwargs["front"])
    plt.text(32, 40, "front", fontsize="16", fontweight="bold", color="black",
             horizontalalignment="center", verticalalignment="center")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(6, 4, 19)
    plt.imshow(kwargs["right"])
    plt.text(32, 40, "right", fontsize="16", fontweight="bold", color="black",
             horizontalalignment="center", verticalalignment="center")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(6, 4, 20)
    plt.imshow(kwargs["back"])
    plt.text(32, 40, "back", fontsize="16", fontweight="bold", color="black",
             horizontalalignment="center", verticalalignment="center")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(6, 4, 14)
    plt.imshow(kwargs["top"])
    plt.text(32, 32, "top", fontsize="16", fontweight="bold", color="black",
             horizontalalignment="center", verticalalignment="center")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(6, 4, 22)
    plt.imshow(kwargs["bottom"])
    plt.text(32, 32, "bottom", fontsize="16", fontweight="bold", color="black",
             horizontalalignment="center", verticalalignment="center")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)


def plot_AOP(**kwargs):
    plt.figure("Angle of Polarisation", figsize=(6, 9))
    ax = plt.subplot(2, 1, 1)
    plt.imshow(kwargs["skydome"], vmin=0, vmax=np.pi, cmap="hsv")
    ax.set_anchor('W')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(6, 4, 17)
    plt.imshow(kwargs["left"], vmin=0, vmax=np.pi, cmap="hsv")
    plt.text(32, 32, "left", fontsize="16", fontweight="bold", color="white",
             horizontalalignment="center", verticalalignment="center")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(6, 4, 18)
    plt.imshow(kwargs["front"], vmin=0, vmax=np.pi, cmap="hsv")
    plt.text(32, 32, "front", fontsize="16", fontweight="bold", color="white",
             horizontalalignment="center", verticalalignment="center")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(6, 4, 19)
    plt.imshow(kwargs["right"], vmin=0, vmax=np.pi, cmap="hsv")
    plt.text(32, 32, "right", fontsize="16", fontweight="bold", color="white",
             horizontalalignment="center", verticalalignment="center")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(6, 4, 20)
    plt.imshow(kwargs["back"], vmin=0, vmax=np.pi, cmap="hsv")
    plt.text(32, 32, "back", fontsize="16", fontweight="bold", color="white",
             horizontalalignment="center", verticalalignment="center")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(6, 4, 14)
    plt.imshow(kwargs["top"], vmin=0, vmax=np.pi, cmap="hsv")
    plt.text(32, 32, "top", fontsize="16", fontweight="bold", color="white",
             horizontalalignment="center", verticalalignment="center")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(6, 4, 22)
    plt.imshow(kwargs["bottom"], vmin=0, vmax=np.pi, cmap="hsv")
    plt.text(32, 32, "bottom", fontsize="16", fontweight="bold", color="white",
             horizontalalignment="center", verticalalignment="center")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)


def plot_skydome(L, DOP, AOP):
    plt.figure("Sky-dome - luminance", figsize=(4.5, 4.5))
    plt.imshow(L)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)

    plt.figure("Sky-dome - Degree of Polarisation", figsize=(4.5, 4.5))
    plt.imshow(DOP)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)

    plt.figure("Sky-dome - Angle of Polarisation", figsize=(4.5, 4.5))
    plt.imshow(AOP, vmin=0, vmax=np.pi, cmap="hsv")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)


sun = Sun()

if __name__ == '__main__':

    try:
        tau = float(argv[argv.index('-t')+1])
    except ValueError:
        tau = -1
    try:
        nside = int(argv[argv.index('-n')+1])
    except ValueError:
        nside = 1
    try:
        city_name = argv[argv.index('-p')+1]
    except ValueError:
        city_name = "Seville"
    if "seville" in city_name.lower():
        obs = get_seville_observer()
    else:
        obs = city(city_name)
    try:
        obs.date = datetime.strptime(argv[argv.index('-d')+1], "%Y-%m-%d")
    except ValueError:
        obs.date = datetime.now()
    try:
        delta = timedelta(minutes=int(argv[argv.index('-td')+1]))
    except ValueError:
        delta = timedelta(minutes=30)
    try:
        mode = ["luminance", "dop", "aop"].index(argv[argv.index('-m')+1].lower())
    except ValueError:
        mode = 0

    cur = obs.next_rising(sun).datetime()
    end = obs.next_setting(sun).datetime()
    if cur > end:
        cur = obs.previous_rising(sun).datetime()

    print "Simulating..."
    print "City: %s" % city_name, "- Turbidity: %.2f" % tau
    print "Date: %s" % obs.date.datetime().strftime("%Y-%m-%d")
    print "Mode: %s" % ["luminance", "dop", "aop"][mode], " - nside = %d" % nside

    while cur <= end:
        obs.date = cur
        sky = ChromaticitySkyModel(observer=obs, turbidity=tau, nside=nside)
        sky.generate()

        # create cubebox parts
        L_left, DOP_left, AOP_left = cubebox(sky, "left")
        L_front, DOP_front, AOP_front = cubebox(sky, "front")
        L_right, DOP_right, AOP_right = cubebox(sky, "right")
        L_back, DOP_back, AOP_back = cubebox(sky, "back")
        L_top, DOP_top, AOP_top = cubebox(sky, "top")
        L_bottom, DOP_bottom, AOP_bottom = cubebox(sky, "bottom")

        # create skydome
        L, DOP, AOP = skydome(sky)

        # plot cubeboxes
        plot_luminance(skydome=L,
                       left=L_left, front=L_front, right=L_right, back=L_back, top=L_top, bottom=L_bottom)
        plot_DOP(skydome=DOP,
                 left=DOP_left, front=DOP_front, right=DOP_right, back=DOP_back, top=DOP_top, bottom=DOP_bottom)
        plot_AOP(skydome=AOP,
                 left=AOP_left, front=AOP_front, right=AOP_right, back=AOP_back, top=AOP_top, bottom=AOP_bottom)

        plt.draw()
        plt.pause(.01)

        # increase the current time
        cur = cur + delta
    plt.show()
