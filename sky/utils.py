import numpy as np
import ephem
import yaml
import os
from sklearn.externals import joblib
__dir__ = os.path.dirname(os.path.realpath(__file__))
with open(__dir__ + "/../colour/CIE-standard-parameters.yaml", 'r') as f:
    try:
        STANDARD_PARAMETERS = yaml.load(f)
    except yaml.YAMLError as exc:
        print exc
gradpar = joblib.load(__dir__ + '/gradation.pkl')
indipar = joblib.load(__dir__ + '/indicatrix.pkl')


def sky_clearness(Z, Dh, I, kapa=1.041):
    """
    
    :param Z: solar zenith angle
    :param Dh: horizontal diffuse irradiance
    :param I: normal incidence direct irradiance
    :param kapa: a constant equal to 1.041 for Z in radians
    :return: the sky's clearness noted as epsilon
    """
    return ((Dh + I)/Dh + kapa * np.power(Z, 3))/(1 + kapa * np.power(Z, 3))


def sky_brightness(Dh, m, I_0):
    """
    
    :param Dh: the horizontal diffuse irradiance
    :param m: the relative optical airmass
    :param I_0: the extraterrestrial irradiance
    :return: the sky's brightness noted as Delta
    """
    return Dh * m / I_0


def water_content(Td):
    """
    
    :param Td: (C) the three-hourly surface dew point temperature
    :return: the atmospheric precipitable water content, denoted W (cm)
    """
    return np.exp(.07 * Td - .075)


def get_luminance_params(sky_type, indicatrix=None):
    if indicatrix is None:  # sky_type is the index of the type of sky according to standard parameters
        assert sky_type is None or 1 <= sky_type <= 15, "Type should be in [1, 15]."
        gradation = get_sky_gradation(sky_type)
        indicatrix = get_sky_indicatrix(sky_type)
    else:  # sky_type is the gradation of the sky with respect to the zenith
        gradation = sky_type

    assert gradation is None or 1 <= gradation <= 6, "Gradation should be in [1, 6]."
    assert indicatrix is None or 1 <= indicatrix <= 6, "indicatrix should be in [1, 6]."

    a = STANDARD_PARAMETERS["gradation"][gradation]["a"]
    b = STANDARD_PARAMETERS["gradation"][gradation]["b"]
    c = STANDARD_PARAMETERS["indicatrix"][indicatrix]["c"]
    d = STANDARD_PARAMETERS["indicatrix"][indicatrix]["d"]
    e = STANDARD_PARAMETERS["indicatrix"][indicatrix]["e"]

    return a, b, c, d, e


def get_sky_description(sky_type, indicatrix=None):
    if indicatrix is not None and 1 <= indicatrix <= 6:
        sky_type = get_sky_type(sky_type, indicatrix)

    if 1 <= sky_type <= 15:
        return STANDARD_PARAMETERS["type"][sky_type - 1]["description"]
    else:
        return None


def get_sky_gradation(sky_type):
    if len(np.shape(sky_type)) == 0 or np.shape(np.squeeze(sky_type))[0] == 1:
        if 1 <= sky_type <= 15:
            return STANDARD_PARAMETERS["type"][sky_type - 1]["gradation"]
        else:
            return -1
    else:
        return gradpar.predict(np.reshape(sky_type, (-1, 5))[:, :2]).squeeze() + 1


def get_sky_indicatrix(sky_type):
    if len(np.shape(sky_type)) == 0 or np.shape(np.squeeze(sky_type))[0] == 1:
        if 1 <= sky_type <= 15:
            return STANDARD_PARAMETERS["type"][sky_type - 1]["indicatrix"]
        else:
            return -1
    else:
        return indipar.predict(np.reshape(sky_type, (-1, 5))[:, 2:]).squeeze() + 1


def get_sky_type(gradation, indicatrix=None):
    """
    
    :param gradation: the gradation type or a vector with the luminance parameters (when indicatrix=None) 
    :param indicatrix: the indicatrix type
    :return: 
    """
    if indicatrix is None:
        indicatrix = get_sky_indicatrix(gradation)
        gradation = get_sky_gradation(gradation)

    if not (1 <= gradation <= 6):
        return -1
    if not (1 <= indicatrix <= 6):
        return -2

    for sky_type, value in enumerate(STANDARD_PARAMETERS["type"]):
        if value["gradation"] == gradation and value["indicatrix"] == indicatrix:
            return sky_type + 1

    return 0


def spectral_power(lam):
    """
    Relative spectral power distribution.
    Note: The wavelengths are to be taken as being in standard air (dry air at 15C and 101325 Pa, containing 0.03% by
     volume of carbon dioxide.
    :param lam: the wavelength in nanometres [300nm, 830nm]
    :return: the spectral power of the correspondent wavelength
    """
    a = 100. * np.power(560. / lam, 5)
    b = np.exp((1.435 * np.power(10, 7)) / (2848 * 560)) - 1
    c = np.exp((1.435 * np.power(10, 7)) / (2848 * lam)) - 1
    return a * b / c


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
    """
    The Rayleigh distribution function. Input 'x' is non negative number and sigma is mode of the distribution.
    :param x: non negative random variable
    :param sigma: the mode of the distribution
    :return: the rayleigh distribution value
    """
    # make sure the input is not negative
    x = np.absolute(x)
    return (x / np.square(sigma)) * np.exp(-np.square(x) / (2 * np.square(sigma)))


def degree_of_polarisation(x, h_max=.8):
    return h_max * np.square(np.sin(x)) / (1. + np.square(np.cos(x)))


def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def get_seville_observer():
    seville = ephem.Observer()
    seville.lat = '37.392509'
    seville.lon = '-5.983877'
    return seville
