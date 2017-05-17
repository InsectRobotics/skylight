from sky.utils import *
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

x, y = [], []
for gradation in xrange(1, 7):
    a = STANDARD_PARAMETERS["gradation"][gradation]["a"]
    b = STANDARD_PARAMETERS["gradation"][gradation]["b"]

    for indicatrix in xrange(1, 7):
        c = STANDARD_PARAMETERS["indicatrix"][indicatrix]["c"]
        d = STANDARD_PARAMETERS["indicatrix"][indicatrix]["d"]
        e = STANDARD_PARAMETERS["indicatrix"][indicatrix]["e"]

        x0 = np.array([a, b, c, d, e])
        y0 = np.array([gradation-1, indicatrix-1])

        x.append(x0)
        y.append(y0)


x, y = np.array(x), np.array(y)
print x.shape, y.shape

grad = LogisticRegression(C=5)
grad.fit(x[:, :2], y[:, 0])

indi = LogisticRegression(C=5)
indi.fit(x[:, 2:], y[:, 1])

joblib.dump(grad, '../sky/gradation.pkl')
joblib.dump(indi, '../sky/indicatrix.pkl')
grad_clf = joblib.load('../sky/gradation.pkl')
indi_clf = joblib.load('../sky/indicatrix.pkl')
print grad_clf.score(x[:, :2], y[:, 0]), indi_clf.score(x[:, 2:], y[:, 1])
