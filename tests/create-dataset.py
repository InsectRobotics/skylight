from sky import ChromaticitySkyModel, get_seville_observer
import ephem
import numpy as np
from datetime import datetime, timedelta


# initialise observer in Seville on 21/06/2017
sun = ephem.Sun()
seville = get_seville_observer()
date = datetime(2017, 6, 1, 0, 0, 0)
seville.date = date

# set time-limits on sunset and sunrise
cur = seville.next_rising(sun).datetime()
end = seville.next_setting(sun).datetime()
# set the time-step at 30 minutes
delta = timedelta(minutes=30)

x, y = [], []

while cur <= end:
    seville.date = cur
    sky = ChromaticitySkyModel(observer=seville, nside=32)

    print "Date =", seville.date
    print "   A =",
    for angle in xrange(360):
        # rotate the agent by 1 degree
        sky = ChromaticitySkyModel.rotate(sky, 1)
        sky.generate()
        x.append(ChromaticitySkyModel.generate_features(sky))
        y.append(angle)
        print "%03d" % (angle + 1),
        if (angle + 1) % 15 == 0:
            print ""
            print "      ",
    print ""

    # increase the current time
    cur = cur + delta

x = np.array(x)
y = np.array(y)

np.savez_compressed('seville-cr-%d-%s.npz' % (sky.nside, date.strftime("%Y%m%d")), x=x, y=y)
print "X:", x.shape, "| Y:", y.shape
