import numpy as np
import quaternion

q1 = np.quaternion(1, 0, 0, 0)
q2 = np.quaternion(5, 6, 7, 8).normalized()

print "Q1", q1
print "Q2", q2
print "Q1 + Q2", q1 + q2
print "Q1 - Q2", q1 - q2
print "Q1 * Q2", q1 * q2
print "Q1 / Q2", q1 / q2
print "log(Q1)", np.log(q1)
print "exp(Q1)", np.exp(q1)
print "Q1 ** 2", np.power(q1, 2)
print "-Q1", -q1
print "conj(Q1)", np.conjugate(q1)
print "abs(Q1)", np.absolute(q1)

print np.rad2deg(quaternion.as_euler_angles(q1))
