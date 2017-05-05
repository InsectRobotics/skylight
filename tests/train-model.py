import numpy as np
from learn import CompassModel, angular_distance_deg, rad2compass


model_name = "seville-jun-dec"
names = ["seville-4-20170321", "seville-4-20170621", "seville-4-20170921", "seville-4-20171221"]

model = CompassModel()
model.load_weights("%s.h5" % model_name)
model.summary()
model.compile(loss=angular_distance_deg, optimizer='rmsprop')

x, y = [], []
for name in names[:-1]:
    print "Loading '%s.npz' ..." % name
    src = np.load('%s.npz' % name)
    x.append(src['x'].reshape((-1, 1, 104, 473)))
    y.append(rad2compass(np.deg2rad(src['y'])))

x = np.concatenate(tuple(x), axis=0)
y = np.concatenate(tuple(y), axis=0)

print x.shape
print y.shape, y.min(), y.max()

stats = model.fit(x, y, batch_size=64, nb_epoch=50, shuffle=True)
model.save_weights("%s.h5" % model_name, overwrite=True)
np.savez_compressed("%s-stats.npz" % model_name, stats=stats)

score = model.evaluate(x, y, batch_size=64)

print score
