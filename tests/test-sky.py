import numpy as np
from sky.model import BlackbodySkyModel, ChromaticitySkyModel
from colorpy.plots import spectrum_plot
from colorpy.ciexyz import xyz_from_spectrum
from colorpy.colormodels import rgb_from_xyz
from datetime import datetime
import string

for tau in xrange(1, 11):
    # sky = BlackbodySkyModel(sky_type=st + 1)
    # sky.generate(show=True)
    # print "L", sky.L.min(), sky.L.max(), sky.L_z
    # print "S", sky.S.shape, sky.S.min(), sky.S.max()
    # print "W", sky.W.shape, sky.W.min(), sky.W.max()
    # print "T: %0.2f, %0.2f" % (sky.T.min(), sky.T.max())

    # sky = ChromaticitySkyModel(sky_type=st+1)
    # sky.obs.date = datetime(2017, 06, 21, 9, 0, 0)
    # sky.generate(show=False)
    # print ""
    # print string.join(sky.description, '\n')
    # print ""
    # print "L:   %0.2f, %0.2f, %0.2f" % (sky.L.min(), sky.L.max(), sky.L_z)
    # print "C_x: %0.2f, %0.2f, %0.2f" % (sky.C_x.min(), sky.C_x.max(), sky.C_x_z)
    # print "C_y: %0.2f, %0.2f, %0.2f" % (sky.C_y.min(), sky.C_y.max(), sky.C_y_z)
    # print "S:   %0.2f, %0.2f" % (sky.S.min(), sky.S.max())
    # print "DOP: %0.2f, %0.2f" % (sky.DOP.min(), sky.DOP.max())
    # tau = sky.turbidity
    # print "Coefficients: %0.2f, %0.2f, %0.2f, %0.2f, %0.2f" % (sky.A, sky.B, sky.C, sky.D, sky.E)
    # print "Turbidity: ", tau
    # print "Gradation: ", sky.gradation
    # print "Indicatrix:", sky.indicatrix

    sky = ChromaticitySkyModel(turbidity=tau)
    sky.obs.date = datetime(2017, 06, 21, 9, 0, 0)
    sky.generate(show=True)
    print ""
    print "L:   %0.2f, %0.2f, %0.2f" % (sky.L.min(), sky.L.max(), sky.L_z)
    print "C_x: %0.2f, %0.2f, %0.2f" % (sky.C_x.min(), sky.C_x.max(), sky.C_x_z)
    print "C_y: %0.2f, %0.2f, %0.2f" % (sky.C_y.min(), sky.C_y.max(), sky.C_y_z)
    print "S:   %0.2f, %0.2f" % (sky.S.min(), sky.S.max())
    print "DOP: %0.2f, %0.2f" % (sky.DOP.min(), sky.DOP.max())
    print "Coefficients: %0.2f, %0.2f, %0.2f, %0.2f, %0.2f" % (sky.A, sky.B, sky.C, sky.D, sky.E)
    print "Turbidity: ", sky.turbidity
    print "Gradation: ", sky.gradation
    print "Indicatrix:", sky.indicatrix
    print ""
    # for i, s in enumerate(sky.S):
    #     S = np.array([sky.W, s]).T
    #     title = "Theta: %0.2f, Phi: %0.2f" % (sky.theta[i], sky.phi[i])
    #     name = "theta_%02d-phi_%03d" % (np.rad2deg(sky.theta[i]), np.rad2deg(sky.phi[i]))
    #     spectrum_plot(S, title, "%s.png" % name)
    #     rgb = rgb_from_xyz(xyz_from_spectrum(S))
    #     print rgb
