from sky import SkyModel


for st in xrange(10, 11):
    sky = SkyModel(sky_type=st+1)
    sky.generate(show=True)
    print "B", sky.B.shape, sky.B.mean(axis=0), sky.B.mean(axis=1)
    print "W", sky.W.shape, sky.W.min(), sky.W.max()
    print "T: %0.2f, %0.2f" % (sky.T.min(), sky.T.max())
