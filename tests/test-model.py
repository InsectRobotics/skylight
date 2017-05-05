import numpy as np
from keras.models import Model
from keras.layers import Dense, Conv2D, Input, Dropout, Flatten
from learn.backend import rad2compass
from learn.loss_function import angular_distance_deg


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

score = angular_distance_deg(y, p).mean()

# score = model.evaluate(x, y, batch_size=64)
print np.rad2deg(score)
