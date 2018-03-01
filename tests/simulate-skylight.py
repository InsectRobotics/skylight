from sky import SkyModel, get_seville_observer
from datetime import datetime, timedelta
from sys import argv
from ephem import city, Sun
from PIL import Image
from sky.utils import pix2sph, sph2pix, Width as W, Height as H
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

# W = H = 1000
sun = Sun()

W = H = 1000

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

    cur = obs.next_rising(sun).datetime() + delta
    end = obs.next_setting(sun).datetime()
    if cur > end:
        cur = obs.previous_rising(sun).datetime() + delta

    print "Simulating..."
    print "City: %s" % city_name, "- Turbidity: %.2f" % tau
    print "Date: %s" % obs.date.datetime().strftime("%Y-%m-%d")
    print "Mode: %s" % ["luminance", "dop", "aop"][mode], " - nside = %d" % nside

    x, y = np.arange(W), np.arange(H)
    x, y = np.meshgrid(x, y)
    x, y = x.flatten(), y.flatten()
    pix = np.array([x, y])
    theta, phi = pix2sph(pix, height=H, width=W)
    j = np.all([~np.isnan(theta), ~np.isnan(phi)], axis=0)
    x, y = x[j], y[j]
    theta, phi = theta[j], phi[j]
    sky = SkyModel(observer=obs, turbidity=tau, nside=1)
    sky.get_features(theta, phi)

    while cur <= end:
        obs.date = cur
        sky = SkyModel(observer=obs, turbidity=tau, nside=1)
        sky.get_features(theta, phi)
        # print np.rad2deg(sky.lon), np.rad2deg(sky.lat)
        for k in xrange(1):
            # sky = sky.rotate_sky(sky, yaw=-k * np.pi / 6)
            # sky = sky.rotate_sky(sky, pitch=-k * np.pi / 18)
            # sky = sky.rotate_sky(sky, roll=-k * np.pi / 9)
            # sky = sky.rotate_sky(
            #     sky,
            #     yaw=(k + 1) * np.pi/6,
            #     pitch=(k + 1) * np.pi/18,
            #     roll=(k + 1) * np.pi/9
            # )
            sky.generate()

            print "(%03d, %03d, %03d) - theta: %03d, phi: %3d" % (
                np.rad2deg((k + 1) * np.pi/6),
                np.rad2deg((k + 1) * np.pi/18),
                np.rad2deg((k + 1) * np.pi/9),
                np.rad2deg(sky.theta_z[0]), np.rad2deg(sky.phi_z[0])
            )

            if mode == 0:
                sph = np.stack([sky.theta_z, sky.phi_z, np.ones_like(sky.theta_z)])
                image = np.zeros((W, H, 3))
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

                # SkyModel.plot_luminance(sky, fig=1, title="Luminance", mode="00100", sub=(1, 1, 1))
            elif mode == 1:

                sph = np.stack([sky.theta_z, sky.phi_z, np.ones_like(sky.theta_z)])
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

                # SkyModel.plot_polarisation(sky, fig=1, title="", mode="10", sub=(1, 1, 1))
            elif mode == 2:
                from matplotlib.cm import get_cmap

                image = np.zeros((W, H, 4))
                sky.DOP[np.isnan(sky.DOP)] = 0.
                i = np.argsort(sky.DOP)
                image[x[i], y[i]] = get_cmap("hsv")(sky.AOP[i] / (2*np.pi))
                plt.figure(1, figsize=(15, 15))
                plt.imshow(image)
                plt.xticks([])
                plt.yticks([])

                # SkyModel.plot_polarisation(sky, fig=1, title="", mode="01", sub=(1, 1, 1))

            img = Image.fromarray((image * 255).astype(np.uint8))
            img.save('aop-%s.png' % datetime.strftime(cur, "%Y%m%d%H%M"))

            plt.draw()
            plt.pause(.01)

        # increase the current time
        cur = cur + delta
    plt.show()
