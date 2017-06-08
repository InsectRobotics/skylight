from sky import ChromaticitySkyModel, get_seville_observer
import ephem
import numpy as np
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.abspath("../"))
__dir__ = os.path.dirname(os.path.realpath(__file__))
__data__ = __dir__ + "/../data/datasets/"


# initialise observer in Seville on 21/06/2017
sun = ephem.Sun()
seville = get_seville_observer()
date = datetime(2017, 12, 21, 0, 0, 0)
seville.date = date

# set time-limits on sunset and sunrise
cur = seville.next_rising(sun).datetime()
end = seville.next_setting(sun).datetime()
# set the time-step at 30 minutes
delta = timedelta(minutes=30)

x, y, t = [], [], []

while cur <= end:
    seville.date = cur
    sky = ChromaticitySkyModel(observer=seville, nside=1)
    sky.generate()

    print "Date =", seville.date
    print "   A =",
    for angle in xrange(360):
        # rotate the agent by 1 degree
        x.append([sky.lon + angle * np.pi/180, sky.lat])
        y.append(angle)
        t.append(cur)
        print "%03d" % (angle + 1),
        if (angle + 1) % 15 == 0:
            print ""
            print "      ",
    print ""

    # increase the current time
    cur = cur + delta

x = np.array(x)
y = np.array(y)
t = np.array(t)

np.savez_compressed(__data__ + 'seville-sn-%d-%s.npz' % (sky.nside, date.strftime("%Y%m%d")), x=x, y=y, t=t)
print "X:", x.shape, "| Y:", y.shape, "| T:", t.shape
