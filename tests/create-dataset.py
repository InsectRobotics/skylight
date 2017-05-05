from sky import SkyModel
import ephem
import numpy as np
from datetime import datetime, timedelta


# initialise observer in Seville on 21/06/2017
sun = ephem.Sun()
seville = ephem.Observer()
seville.lat = '37.392509'
seville.lon = '-5.983877'
seville.date = datetime(2017, 3, 21, 0, 0, 0)

# set time-limits on sunset and sunrise
cur = seville.next_rising(sun).datetime()
end = seville.next_setting(sun).datetime()
# set the time-step at 30 minutes
delta = timedelta(minutes=30)

x, y = [], []

while cur <= end:
    seville.date = cur
    sky = SkyModel(observer=seville, nside=4)

    print "Date =", seville.date
    print "   A =",
    for angle in xrange(360):
        # rotate the agent by 1 degree
        sky = SkyModel.rotate(sky, 1)
        sky.generate()
        x.append(SkyModel.generate_features(sky))
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

np.savez_compressed('seville-4-20170321.npz', x=x, y=y)
