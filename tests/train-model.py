import numpy as np
from learn import from_file, angdist
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
x_train, y_train = model.load_dataset(names[:-1], pol=True, cx=False, directionwise=False)
x_train = x_train.reshape((x_train.shape[0], -1))
i = np.all(~np.isnan(x_train), axis=1)
x_train, y_train = x_train[i], np.cos(y_train[i])
print x_train.shape, y_train.shape
x_test, y_test = model.load_dataset(names[-1:], pol=True, cx=False, directionwise=False)
x_test = x_test.reshape((x_test.shape[0], -1))
i = np.all(~np.isnan(x_test), axis=1)
x_test, y_test = x_test[i], np.cos(y_test[i])
print x_test.shape, y_test.shape
# reset_state = x_train.shape[0] / 360
hist = model.train((x_train, y_train), valid_data=(x_test, y_test),
                   nb_epoch=300)

p_train = model.predict(x_train)
p_test = model.predict(x_test)

print "Training error:", 1. - angdist(np.arccos(p_train), np.arccos(y_train)).mean()
print "Test error:", 1. - angdist(np.arccos(p_test), np.arccos(y_test)).mean()

# plot progress
plt.figure(1, figsize=(15, 20))

plt.subplot(221)
plt.plot(hist.history['loss'])
plt.title("Training loss")
plt.ylim([0, 1])

plt.subplot(222)
plt.plot(1. - hist.history['acc'])
plt.title("Training error")
plt.ylim([0, 1])

plt.subplot(223)
plt.plot(hist.history['val_loss'])
plt.title("Validation loss")
plt.ylim([0, 1])

plt.subplot(224)
plt.plot(1. - hist.history['val_acc'])
plt.title("Validation error")
plt.ylim([0, 1])

plt.show()
