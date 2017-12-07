import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import ephem
import os
from datetime import datetime, timedelta
from sky.model import BlackbodySkyModel
from sky import get_seville_observer, sun2lonlat
from learn import CompassModel, get_loss
from code import decode_sph


angular_distance_deg = get_loss("add")

d = datetime(2017, 01, 21, 0, 0, 0)
dir_img = "/home/thor/skylight/%s/" % d.strftime("%Y/%m/%d")
dir_flt = dir_img + "filter/"
if not os.path.exists(dir_img):
    os.makedirs(dir_img)
if not os.path.exists(dir_flt):
    os.makedirs(dir_flt)

model_name = "seville-full"

model = CompassModel()
model.load_weights("%s.h5" % model_name)
model.summary()
model.compile(loss=angular_distance_deg, optimizer='rmsprop')

ws = model.get_weights()
w0 = ws[0].squeeze().T

fs = (15, 10)
jump = 0
plt.figure(1, figsize=fs)
for i in xrange(w0.shape[0]):
    if ((i + jump) % 2) == 0 and not ((i + jump) % 4) == 0:
        jump += 2
    plt.subplot(5, 4, i+1 + jump)
    plt.plot(np.arange(360, 831, 1), np.absolute(w0[i][:-2]))
    plt.title("Filter %02d" % (i + 1))
    if i < 8:
        plt.xticks([])
    plt.xlim([360, 830])
    plt.ylim([0, .6])

w_ave = np.sqrt(np.square(w0).sum(axis=0))
plt.subplot(122)
plt.plot(np.arange(360, 831, 1), w_ave[:-2])
plt.title("Average Filter")
plt.xlim([360, 830])
plt.ylim([0, .6])

plt.savefig("%s%s.png" % (dir_img, "compass-wavelength-filters"))
# plt.show()

plt.figure(2, figsize=fs)
lum = np.absolute(w0[:, :-2]).mean(axis=1)
dop = np.absolute(w0[:, -2:-1])
aop = np.absolute(w0[:, -1:])

plt.boxplot([lum, dop, aop], labels=["Luminance", "DOP", "AOP"])
plt.grid()
plt.title("Feature importance wrt category")

plt.savefig("%s%s.png" % (dir_img, "compass-category-filters"))
# plt.show()


# initialise observer in Seville on 21/06/2017
sun = ephem.Sun()
seville = get_seville_observer()
seville.date = d

# set time-limits on sunset and sunrise
cur = seville.next_rising(sun).datetime()
end = seville.next_setting(sun).datetime()
# set the time-step at 30 minutes
delta = timedelta(minutes=30)

while cur <= end:
    seville.date = cur
    sky = BlackbodySkyModel(observer=seville, nside=4)
    sky.generate()
    lon, lat = sun2lonlat(sky.sun)

    x = BlackbodySkyModel.generate_features(sky).reshape((-1, 1, 104, 473))
    y = model.filter.predict(x, batch_size=32).squeeze()

    sphere_ave = np.zeros(hp.nside2npix(4))
    sphere = np.zeros(hp.nside2npix(4))
    jump = 0
    plt.figure(3, figsize=fs)
    for i in xrange(y.shape[0]):
        if ((i + jump) % 2) == 0 and not ((i + jump) % 4) == 0:
            jump += 2
        sphere[:(y[i].shape[0]-2)] = y[i][:-2]
        hp.orthview(sphere, rot=BlackbodySkyModel.VIEW_ROT, flip="geo", cmap="coolwarm", half_sky=True,
                    min=-.2, max=.2, title="Filter %02d" % (i+1), sub=(5, 4, i+1 + jump), fig=3)
        hp.projplot(lat, lon, 'yo')
        sphere_ave += np.absolute(sphere)

    hp.orthview(sphere_ave, rot=BlackbodySkyModel.VIEW_ROT, flip="geo", cmap="Greys", half_sky=True,
                min=0, max=.2, title="Total Filter", sub=(1, 2, 2), fig=3)
    hp.projplot(lat, lon, 'yo')

    plt.savefig("%s%s.png" % (dir_flt, cur.strftime("%Y-%m-%d_%H-%M")))
    # plt.show()

    # increase the current time
    cur = cur + delta

# src = np.load('seville-4-20170621.npz')
# x = src['x'].reshape((-1, 1, 104, 473))
# y = np.deg2rad(src['y']) % (2 * np.pi)
# y = rad2compass(y)
#
# print x.shape
# print y.shape, y.min(), y.max()
#
# p = model.filter.predict(x, batch_size=32)
#
# plt.show()
#
# nside = 4
# npixel = hp.nside2npix(nside)
# sphere = np.zeros(npixel)
# for j in xrange(p.shape[0] // 360):
#     p0 = p[j * 360].squeeze()
#
#     plt.figure(3)
#     for i in xrange(p0.shape[0]):
#         sphere[:(p0[i].shape[0]-2)] = p0[i][:-2]
#         hp.orthview(sphere, rot=BlackbodySkyModel.VIEW_ROT, flip="geo", cmap="coolwarm", half_sky=True,
#                     min=-.2, max=.2, title="Filter %02d" % (i+1), sub=(2, 5, i+1), fig=3)
#     plt.show()


# p = model.predict(x, batch_size=32)
#
# score = angular_distance_deg(y, p).mean()
#
# score = model.evaluate(x, y, batch_size=64)
# print np.rad2deg(score)
