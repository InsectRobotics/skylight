from sky import SkyModel


for st in xrange(10, 11):
    sky = SkyModel(sky_type=st+1)
    sky.generate(show=True)
