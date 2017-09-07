from sky import ChromaticitySkyModel, get_seville_observer
from datetime import datetime, timedelta
from sys import argv
from ephem import city, Sun
from sky import sph2pix, Width as W, Height as H
import numpy as np
import matplotlib.pyplot as plt
plt.ion()


sun = Sun()

if __name__ == '__main__':

    try:
        tau = float(argv[argv.index('-t')+1])
    except ValueError:
        tau = -1
    try:
        nside = int(argv[argv.index('-n')+1])
    except ValueError:
        nside = 512
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

        sph = np.stack([sky.theta, sky.phi, np.ones_like(sky.theta)])
        x, y = sph2pix(sph, W, H)

        if mode == 0:
            sph = np.stack([sky.theta, sky.phi, np.ones_like(sky.theta)])
            image = np.zeros((W, H, 3))
            x, y = sph2pix(sph, W, H)
            sky.DOP[np.isnan(sky.DOP)] = -1
            i = np.argsort(sky.DOP)
            # print sky.L.max()  # min(sky.L.max()) = 20, max(sky.L.max()) = 71
            L = sky.L[i] / 20.
            image[x[i], y[i], 0] = np.power(L, 1) + np.power(1. - L, 1) * .05  # deep sky blue * .53
            image[x[i], y[i], 1] = np.power(L, 1) + np.power(1. - L, 1) * .53  # deep sky blue * .81
            image[x[i], y[i], 2] = np.power(L, 1) + np.power(1. - L, 1) * .79  # deep sky blue * .92
            image = np.clip(image, 0, 1)
            plt.figure(1, figsize=(15, 15))
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])

            # ChromaticitySkyModel.plot_luminance(sky, fig=1, title="Luminance", mode="00100", sub=(1, 1, 1))
        elif mode == 1:

            sph = np.stack([sky.theta, sky.phi, np.ones_like(sky.theta)])
            image = np.zeros((W, H, 3))
            x, y = sph2pix(sph, W, H)
            sky.DOP[np.isnan(sky.DOP)] = 0.
            i = np.argsort(sky.DOP)
            image[x[i], y[i], 0] = sky.DOP[i] * .53
            image[x[i], y[i], 1] = sky.DOP[i] * .81
            image[x[i], y[i], 2] = sky.DOP[i] * 1.0
            plt.figure(1, figsize=(15, 15))
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])

            # ChromaticitySkyModel.plot_polarisation(sky, fig=1, title="", mode="10", sub=(1, 1, 1))
        elif mode == 2:

            image = np.zeros((W, H))
            sky.DOP[np.isnan(sky.DOP)] = 0.
            i = np.argsort(sky.DOP)
            image[x[i], y[i]] = sky.AOP[i]
            plt.figure(1, figsize=(15, 15))
            plt.imshow(image, cmap="hsv")
            plt.xticks([])
            plt.yticks([])

            # ChromaticitySkyModel.plot_polarisation(sky, fig=1, title="", mode="01", sub=(1, 1, 1))

        plt.draw()
        plt.pause(.01)

        # increase the current time
        cur = cur + delta
    plt.show()
