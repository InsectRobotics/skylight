import numpy as np
import sys
import os
sys.path.append(os.path.abspath("../"))

from learn import CompassModel, from_file, angdist

city = "seville"
nside = 1  # 32
dates = [
    "20170121",
    "20170221",
    "20170321",
    "20170421",
    "20170521",
    "20170621",
    "20170721",
    "20170821",
    "20170921",
    "20171021",
    "20171121",
    "20171221",
    "20170601"
]

names = ["%s-cr-%d-%s" % (city, nside, date) for date in dates]

model = from_file("simple-cnn-1.yaml")
model.summary()

# Load training data
x, y = CompassModel.load_dataset(names[:-1], pol=True, x_shape=model.data_shape, y_shape=(-1, 1))
x = x.reshape((x.shape[0], -1))
# Clean training data
i = np.all(~np.isnan(x), axis=1)
x = x.reshape(model.data_shape)
x, y = x[i], y[i]
print "Train:", x.shape, y.shape

# Load testing data
x_test, y_test = CompassModel.load_dataset(names[-1:], pol=True, x_shape=model.data_shape, y_shape=(-1, 1))
x_test = x_test.reshape((x_test.shape[0], -1))
# Clean testing data
i = np.all(~np.isnan(x_test), axis=1)
x_test = x_test.reshape(model.data_shape)
x_test, y_test = x_test[i], y_test[i]
print "Test:", x_test.shape, y_test.shape

hist = model.train((x, y), valid_data=(x_test, y_test))

p = model.predict(x)
acc = 1 - angdist(y.squeeze(), p.squeeze()).mean() / np.pi
print "Train - Accuracy:", acc

p_test = model.predict(x_test)
acc_test = 1 - angdist(y_test.squeeze(), p_test.squeeze()).mean() / np.pi
print "Test  - Accuracy:", acc_test
