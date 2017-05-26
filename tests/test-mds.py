from sklearn.manifold import MDS
from learn import CompassModel
from time import time
import numpy as np
import matplotlib.pyplot as plt


names = [
    "seville-cr-32-20170321",
    # "seville-cr-32-20170621",
    # "seville-cr-32-20170921",
    # "seville-cr-32-20171221",
    "seville-cr-32-20170601"
]

# Load training data
x, y = CompassModel.load_dataset(names[:-1], pol=True, y_shape=(-1, 1))
x = x.reshape((x.shape[0], -1))
i = np.all(~np.isnan(x), axis=1)
x, y = x[i], y[i]
print "Train:", x.shape, y.shape

# PCA
mds = MDS(n_components=2, n_jobs=3)
print "Reducing dimensions..."
t0 = time()
x_new = mds.fit_transform(x)
t1 = time()
print "New train:", x_new.shape

fig = plt.figure(figsize=(15, 8))
plt.scatter(x_new[:, 0], x_new[:, 1], marker='.', c=y, cmap="hsv")
plt.title("MDS (%.2g sec)" % (t1 - t0))
plt.colorbar()
plt.show()
