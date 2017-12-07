from sklearn.externals import joblib
from sklearn.decomposition import PCA
from learn import CompassModel
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

names = [
    "seville-cr-32-20170321",
    "seville-cr-32-20170621",
    # "seville-cr-32-20170921",
    # "seville-cr-32-20171221",
    "seville-cr-32-20170601"
]

# Load training data
# x, y = CompassModel.load_dataset(names[:-1], pol=True, cx=False)
# x = x.reshape((x.shape[0], -1))
# i = np.all(~np.isnan(x), axis=1)
# x, y = x[i], y[i]
# print "Train:", x.shape, y.shape

# Load testing data
x_test, y_test = CompassModel.load_dataset(names[-1:], pol=True, cx=False)
x_test = x_test.reshape((x_test.shape[0], -1))
i = np.all(~np.isnan(x_test), axis=1)
x_test, y_test = x_test[i], y_test[i]
print "Test: ", x_test.shape, y_test.shape

# PCA
pca = PCA(n_components=1)
print "Reducing dimensions..."
pca.fit(x_test)
# x_new = pca.code(x)
x_new_test = pca.transform(x_test)
print "New test:", x_new_test.shape

plt.figure(1, figsize=(20, 15))
plt.plot(x_new_test, y_test, 'ro')
plt.ylim([0, 2*np.pi])
plt.title("Testing data")
plt.draw()
plt.pause(.001)

# Load Gaussian Process Regression model
print "Loading model..."
gpr = joblib.load('../data/gpr-mar-jun.pkl')

# p, p_std = gpr.predict(x, return_std=True)
# print "P:", p.shape, p_std.shape
print "Predicting test..."
p_test, p_test_std = gpr.predict(x_test, return_std=True)
print "P_test:", p_test.shape, p_test_std.shape

plt.figure(2)
# plt.subplot(121)
# plt.plot(x_new, y, 'ro')
# plt.plot(x_new, p, 'ko')
# plt.title("Training data")

# plt.subplot(122)
plt.plot(x_new_test, y_test, 'ro')
plt.plot(x_new_test, p_test, 'ko')
plt.title("Testing data")
plt.show()
