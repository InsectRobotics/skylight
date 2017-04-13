import numpy as np


def sun2lonlat(s, lonlat=False, show=False):
    lon, lat = s.az, s.alt
    colat = np.pi / 2 - lat

    if lon > np.pi:
        lon -= 2 * np.pi

    if show:
        print('Sun:\tLon = %.2f\t Lat = %.2f\t Co-Lat = %.2f' % \
                (np.rad2deg(lon), np.rad2deg(lat), np.rad2deg(colat)))

    if lonlat:  # return the longitude and the latitude in degrees
        return np.rad2deg(lon), np.rad2deg(lat)
    else:  # return the lngitude and the co-latitude in radians
        return lon, colat

def hard_sigmoid(x, s=10):
    return 1. / (1. + np.exp(-s * x))

def rayleigh(x, sigma=np.pi/2):
    # make sure the input is not negative
    x = np.absolute(x)
    return (x / np.square(sigma)) * np.exp(-np.square(x) / (2 * np.square(sigma)))
