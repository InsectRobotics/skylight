import numpy as np
from keras.models import Model
from keras.layers import Dense, Conv2D, Input, Dropout, Flatten


def rad2compass(phi, length=8):
    alpha = np.arange(length) * 2 * np.pi / length
    I = np.cos(phi[..., np.newaxis] - alpha)
    return I


model_name = "seville-jun-dec"
names = ["seville-4-20170621", "seville-4-20171221"]

x, y = [], []
for name in names:
    src = np.load('%s.npz' % name)
    x.append(src['x'].reshape((-1, 1, 104, 473)))
    y.append(rad2compass(np.deg2rad(src['y'])))

x = np.concatenate(tuple(x), axis=0)
y = np.concatenate(tuple(y), axis=0)

print x.shape
print y.shape, y.min(), y.max()

inp = Input((1, 104, 473))
out = Conv2D(10, 1, 473, activation='relu')(inp)
out = Flatten()(out)
out = Dense(500, activation='relu')(out)
out = Dropout(0.5)(out)
out = Dense(32, activation='relu')(out)
out = Dropout(0.5)(out)
out = Dense(8, activation='tanh')(out)

model = Model(inp, out)
model.load_weights("seville-4-20170621.h5" % name)
model.summary()
model.compile(loss='mae', optimizer='rmsprop')

stats = model.fit(x, y, batch_size=64, nb_epoch=50, shuffle=True)
model.save_weights("%s.h5" % model_name, overwrite=True)
np.savez_compressed("%s-stats.npz" % model_name, stats=stats)

score = model.evaluate(x, y, batch_size=64)

print score
