from sky import Sky, get_seville_observer, cubebox, skydome
from compoundeye import CompoundEye
from datetime import datetime, timedelta
from sys import argv
from ephem import city, Sun
import numpy as np
import matplotlib.pyplot as plt
plt.ion()


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

    cur = obs.next_rising(sun).datetime() + delta
    end = obs.next_setting(sun).datetime()
    if cur > end:
        cur = obs.previous_rising(sun).datetime()

    print "Simulating..."
    print "City: %s" % city_name, "- Turbidity: %.2f" % tau
    print "Date: %s" % obs.date.datetime().strftime("%Y-%m-%d")
    print "Mode: %s" % ["luminance", "dop", "aop"][mode], " - nside = %d" % nside

    def fibonacci_sphere(samples, fov):

        theta = np.deg2rad(fov / 2)  # angular distance of the outline from the zenith
        phi = (1. + np.sqrt(5)) * np.pi

        r_l = 1.  # the small radius of a hexagon (mm)
        R_l = r_l * 2 / np.sqrt(3)  # the big radius of a lens (mm)
        S_l = 3 * r_l * R_l  # area of a lens (mm^2)

        S_a = samples * S_l  # area of the dome surface (mm^2)
        R_c = np.sqrt(S_a / (2 * np.pi * (1. - np.cos(theta))))  # radius of the curvature (mm)
        S_c = 4 * np.pi * np.square(R_c)  # area of the whole sphere (mm^2)

        total_samples = int(samples * S_c / (1.2 * S_a))

        indices = np.arange(0, total_samples, dtype=float)

        thetas = np.arccos(2 * indices / (total_samples - .5) - 1)
        phis = (phi * indices) % (2 * np.pi)

        return thetas[-samples:], phis[-samples:]


    theta, phi = fibonacci_sphere(1000, 180)

    while cur <= end:
        obs.date = cur
        sky = Sky.from_observer(obs=obs)
        y, p, a = sky(theta, phi)

        for r in xrange(0, 360, 90):
            # create cubebox parts
            L_left, DOP_left, AOP_left = cubebox(sky, "left", rot=r, eye_model=CompoundEye)
            L_front, DOP_front, AOP_front = cubebox(sky, "front", rot=r, eye_model=CompoundEye)
            L_right, DOP_right, AOP_right = cubebox(sky, "right", rot=r, eye_model=CompoundEye)
            L_back, DOP_back, AOP_back = cubebox(sky, "back", rot=r, eye_model=CompoundEye)
            L_top, DOP_top, AOP_top = cubebox(sky, "top", rot=r, eye_model=CompoundEye)
            L_bottom, DOP_bottom, AOP_bottom = cubebox(sky, "bottom", rot=r, eye_model=CompoundEye)

            # create skydome
            L, DOP, AOP = skydome(sky, rot=r, eye_model=CompoundEye)

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
