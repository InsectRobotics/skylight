import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath("../"))

from learn import CompassModel, from_file, angdist


names = [
    # "seville-cr-32-20170321",
    "seville-cr-32-20170621",
    # "seville-cr-32-20170921",
    "seville-cr-32-20171221",
    "seville-cr-32-20170601"
]

model = from_file("simple-model.yaml")

# Load training data
x, y = CompassModel.load_dataset(names[:-1], pol=True, y_shape=(-1, 1))
x = x.reshape((x.shape[0], -1))
# Clean training data
i = np.all(~np.isnan(x), axis=1)
x = x.reshape((x.shape[0], -1, 2))
x, y = x[i], y[i].squeeze()
print "Train:", x.shape, y.shape

# Load testing data
x_test, y_test = CompassModel.load_dataset(names[-1:], pol=True, y_shape=(-1, 1))
x_test = x_test.reshape((x_test.shape[0], -1))
# Clean testing data
i = np.all(~np.isnan(x_test), axis=1)
x_test = x_test.reshape((x_test.shape[0], -1, 2))
x_test, y_test = x_test[i], y_test[i].squeeze()
print "Test:", x_test.shape, y_test.shape

# model = from_file("simple-model.yaml")
model.summary()

hist = model.train((x, y), valid_data=(x_test, y_test))

p = np.deg2rad(model.predict(x))
acc = 1 - angdist(y, p).mean() / np.pi
print "Train - Accuracy:", acc

p_test = np.deg2rad(model.predict(x_test))
acc_test = 1 - angdist(y_test, p_test).mean() / np.pi
print "Test  - Accuracy:", acc_test

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
