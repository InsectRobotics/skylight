from sky import SkyModel


for st in xrange(15):
    sky = SkyModel(sky_type=st+1)
    sky.generate(show=True)
