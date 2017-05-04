import numpy as np
from keras.models import Model
from keras.layers import Dense, Conv2D, Input, Dropout, Flatten


def rad2compass(phi, length=8):
    alpha = np.arange(length) * 2 * np.pi / length
    I = np.cos(phi[..., np.newaxis] - alpha)
    return I


def compass2rad(I):
    length = I.shape[-1]
    alpha = np.arange(length) * 2 * np.pi / length
    x0, y0 = (I * np.cos(alpha)).sum(axis=-1), (I * np.sin(alpha)).sum(axis=-1)
    phi = np.arctan2(y0, x0) % (2 * np.pi)
    return phi


def angdist(a, b):
    d = np.absolute(a - b)
    d[d > np.pi] = np.absolute(b[d > np.pi] - a[d > np.pi])
    return d


src = np.load('seville-4-20170621.npz')
x = src['x'].reshape((-1, 1, 104, 473))
y = np.deg2rad(src['y']) % (2 * np.pi)
y = rad2compass(y)

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
model.load_weights("seville-4-20170621.h5")
model.summary()
model.compile(loss='mae', optimizer='rmsprop')

# model.fit(x, y, batch_size=64, nb_epoch=50, shuffle=True)
# model.save_weights("seville-4-20170621.h5", overwrite=True)

p = model.predict(x, batch_size=32)
y = compass2rad(y)
p = compass2rad(p)

score = angdist(y, p).mean()

# score = model.evaluate(x, y, batch_size=64)
print np.rad2deg(score)
