from learn import from_file, angdist, compass2rad2
import numpy as np
import matplotlib.pyplot as plt


# names = ["seville-bb-4-20170321", "seville-bb-4-20170621", "seville-bb-4-20170921", "seville-bb-4-20171221"]
names = [
    "seville-cr-32-20170321",
    # "seville-cr-32-20170621",
    # "seville-cr-32-20170921",
    # "seville-cr-32-20171221",
    "seville-cr-32-20170601"
]

model = from_file("dense-model.yaml")
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()
# model.load_weights("%s.h5" % model_name)
x_train, y_train = model.load_dataset(names[:-1], pol=True, y_shape=(-1, 360), directionwise=False)
x_train = x_train.reshape((x_train.shape[0], -1))
i = np.all(~np.isnan(x_train), axis=1)
x_train, y_train = x_train[i], y_train[i]
print x_train.shape, y_train.shape
x_test, y_test = model.load_dataset(names[-1:], pol=True, y_shape=(-1, 360), directionwise=False)
x_test = x_test.reshape((x_test.shape[0], -1))
i = np.all(~np.isnan(x_test), axis=1)
x_test, y_test = x_test[i], y_test[i]
print x_test.shape, y_test.shape
# reset_state = x_train.shape[0] / 360
hist = model.train((x_train, y_train), valid_data=(x_test, y_test),
                   epochs=300)

# e_train = model.evaluate(x_train, y_train)
# e_test = model.evaluate(x_test, y_test)
#
# print "Training error:", 1. - angdist(compass2rad2(p_train), compass2rad2(y_train)).mean() / np.pi
# print "Test error:", 1. - angdist(compass2rad2(p_test), compass2rad2(y_test)).mean() / np.pi

# plot progress
plt.figure(1, figsize=(15, 20))

plt.subplot(221)
plt.plot(hist.history['loss'])
plt.title("Training loss")
plt.ylim([0, 1])

plt.subplot(222)
plt.plot(hist.history['acc'])
plt.title("Training accuracy")
plt.ylim([0, 1])

plt.subplot(223)
plt.plot(hist.history['val_loss'])
plt.title("Validation loss")
plt.ylim([0, 1])

plt.subplot(224)
plt.plot(hist.history['val_acc'])
plt.title("Validation accuracy")
plt.ylim([0, 1])

plt.show()
