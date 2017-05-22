import numpy as np
from learn import from_file
import matplotlib.pyplot as plt


# names = ["seville-bb-4-20170321", "seville-bb-4-20170621", "seville-bb-4-20170921", "seville-bb-4-20171221"]
names = [
    "seville-cr-32-20170321",
    "seville-cr-32-20170621",
    "seville-cr-32-20170921",
    "seville-cr-32-20171221",
    "seville-cr-32-20170601"
]

model = from_file("dense-model.yaml")
model.compile(optimizer="rmsprop", loss="mae", metrics=["accuracy"])
model.summary()
# model.load_weights("%s.h5" % model_name)
x_train, y_train = model.load_dataset(names[:-1], pol=True, directionwise=False)
x_train = x_train.reshape((x_train[0], -1))
print x_train.shape, y_train.shape
x_test, y_test = model.load_dataset(names[-1:], pol=True, directionwise=False)
x_test = x_test.reshape((x_test[0], -1))
print x_test.shape, y_test.shape
# reset_state = x_train.shape[0] / 360
hist = model.train((x_train, y_train), valid_data=(x_test, y_test),
                   nb_epoch=300)

# plot progress
plt.figure(1, figsize=(15, 20))

plt.subplot(121)
plt.plot(hist['loss'])
plt.ylim([0, 1])

plt.subplot(122)
plt.plot(hist['accuracy'])
plt.ylim([0, 1])

plt.show()
