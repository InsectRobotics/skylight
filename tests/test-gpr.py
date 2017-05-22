import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.externals import joblib
from learn import CompassModel


names = [
    "seville-cr-32-20170321",
    "seville-cr-32-20170621",
    # "seville-cr-32-20170921",
    # "seville-cr-32-20171221",
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

# Train model
gp = GaussianProcessRegressor(normalize_y=True, copy_X_train=False)
gp.fit(x, y)

# Test model
train_score = gp.score(x, y)
print "Training score:", train_score
test_score = gp.score(x_test, y_test)
print "Testing score:", test_score

joblib.dump(gp, 'gpr.pkl')
