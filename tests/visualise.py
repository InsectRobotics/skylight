from sklearn.decomposition import PCA
from learn import CompassModel
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# plt.ion()

names = [
    "seville-cr-32-20170321",
    "seville-cr-32-20170621",
    "seville-cr-32-20170921",
    "seville-cr-32-20171221",
    "seville-cr-32-20170601"
]

# Load training data
x, y = CompassModel.load_dataset(names[:-1], pol=True, cx=False)
x = x.reshape((x.shape[0], -1))
i = np.all(~np.isnan(x), axis=1)
x, y = x[i], y[i]
print "Train:", x.shape, y.shape

# Load testing data
x_test, y_test = CompassModel.load_dataset(names[-1:], pol=True, cx=False)
x_test = x_test.reshape((x_test.shape[0], -1))
i = np.all(~np.isnan(x_test), axis=1)
x_test, y_test = x_test[i], y_test[i]
print "Test: ", x_test.shape, y_test.shape

# PCA
pca = PCA(n_components=2)
print "Reducing dimensions..."
pca.fit(x)
x_new = pca.transform(x)
x_new_test = pca.transform(x_test)
print "New test:", x_new_test.shape

N = x_new.shape[0] / 360
plt.figure(1, figsize=(20, 15))
plt.subplot(121)
for i in xrange(N):
    plt.plot(x_new[(i*360):((i+1)*360), 0], np.rad2deg(y[(i*360):((i+1)*360)]))
plt.ylim([0, 360])
plt.title("Testing data dim=1")

plt.subplot(122)
for i in xrange(N):
    plt.plot(x_new[(i*360):((i+1)*360), 1], np.rad2deg(y[(i*360):((i+1)*360)]))
plt.ylim([0, 360])
plt.title("Testing data dim=2")

fig = plt.figure(2)
ax = fig.gca(projection='3d')
for i in xrange(N):
    ax.plot(
        x_new[(i*360):((i+1)*360), 0],
        x_new[(i*360):((i+1)*360), 1],
        np.rad2deg(y[(i*360):((i+1)*360)]), label="%02.1f" % (i/2.))
ax.legend()

N = x_new_test.shape[0] / 360
plt.figure(3, figsize=(20, 15))
plt.subplot(121)
for i in xrange(N):
    plt.plot(x_new_test[(i*360):((i+1)*360), 0], np.rad2deg(y_test[(i*360):((i+1)*360)]))
plt.ylim([0, 360])
plt.title("Testing data dim=1")

plt.subplot(122)
for i in xrange(N):
    plt.plot(x_new_test[(i*360):((i+1)*360), 1], np.rad2deg(y_test[(i*360):((i+1)*360)]))
plt.ylim([0, 360])
plt.title("Testing data dim=2")

fig = plt.figure(4)
ax = fig.gca(projection='3d')
for i in xrange(N):
    ax.plot(
        x_new_test[(i*360):((i+1)*360), 0],
        x_new_test[(i*360):((i+1)*360), 1],
        np.rad2deg(y_test[(i*360):((i+1)*360)]), label="%02.1f" % (i/2.))
ax.legend()
plt.show()

