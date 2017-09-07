"""
ciexyz.py - Spectral response curves for 1931 CIE XYZ 2 degree field of view matching functions.

Description:

This module provides the CIE standard XYZ color matching functions.
The 1931 tabulation, for a 2 degree field of view, is used in preference to the 10 degree 1964 set,
as is conventional in computer graphics.

The matching functions are stored internally at 1 nm increments, and linear interpolation is
used for any wavelength in between.

ColorPy attempts to scale the matching functions so that:
A spectrum, constant with wavelength, over the range 360 nm to 830 nm, with a total intensity
equal to the (assumed) physical intensity of the monitor, will sample with Y = 1.0.

This scaling corresponds with that in colormodels.py, which assumes Y = 1.0 at full white.

NOTE - I suspect that the scaling is not quite correct.  I think it is at least close.

Ideally, we would like the spectrum of the actual monitor display, at full white, which is not
independent of wavelength, to sample to Y = 1.0.

Constants and Functions:

start_wl_nm, end_wl_nm - Default starting and ending range of wavelengths, in nm, as integers.
delta_wl_nm            - Default wavelength spacing, in nm, as a float.

DEFAULT_DISPLAY_INTENSITY - Default assumed intensity of monitor display, in W/m^2

def init (monitor_intensity = DEFAULT_DISPLAY_INTENSITY) -
    Initialization of color matching curves.  Called at module startup with default arguments.
    This can be called again to change the assumed display intensity.

def empty_spectrum () -
    Get a black (no intensity) ColorPy spectrum.

    This is a 2D numpy array, with one row for each wavelength in the visible range,
    360 nm to 830 nm, with a spacing of delta_wl_nm (1.0 nm), and two columns.
    The first column is filled with the wavelength [nm].
    The second column is filled with 0.0.  It should later be filled with the intensity.

    The result can be passed to xyz_from_spectrum() to convert to an xyz color.

def xyz_from_wavelength (wl_nm) -
    Given a wavelength (nm), return the corresponding xyz color, for unit intensity.

def xyz_from_spectrum (spectrum) -
    Determine the xyz color of the spectrum.

    The spectrum is assumed to be a 2D numpy array, with a row for each wavelength,
    and two columns.  The first column should hold the wavelength (nm), and the
    second should hold the light intensity.  The set of wavelengths can be arbitrary,
    it does not have to be the set that empty_spectrum() returns.

def get_normalized_spectral_line_colors (
    brightness = 1.0,
    num_purples = 0,
    dwl_angstroms = 10):
    Get an array of xyz colors covering the visible spectrum.
    Optionally add a number of 'purples', which are colors interpolated between the color
    of the lowest wavelength (violet) and the highest (red).

    brightness - Desired maximum rgb component of each color.  Default 1.0.  (Maxiumum displayable brightness)
    num_purples - Number of colors to interpolate in the 'purple' range.  Default 0.  (No purples)
    dwl_angstroms - Wavelength separation, in angstroms (0.1 nm).  Default 10 A. (1 nm spacing)

References:

Wyszecki and Stiles, Color Science: Concepts and Methods, Quantitative Data and Formulae,
    2nd edition, John Wiley, 1982. Wiley Classics Library Edition 2000. ISBN 0-471-39918-3.

CVRL Color and Vision Database - http://cvrl.ioo.ucl.ac.uk/index.htm - (accessed 17 Sep 2008)
    Color and Vision Research Laboratories.
    Provides a set of data sets related to color vision.
    ColorPy uses the tables from this site for the 1931 CIE XYZ matching functions,
    and for Illuminant D65, both at 1 nm wavelength increments.

CIE Standards - http://cvrl.ioo.ucl.ac.uk/cie.htm - (accessed 17 Sep 2008)
    CIE standards as maintained by CVRL.
    The 1931 CIE XYZ and D65 tables that ColorPy uses were obtained from the following files, linked here:
        http://cvrl.ioo.ucl.ac.uk/database/data/cmfs/ciexyz31_1.txt
        http://cvrl.ioo.ucl.ac.uk/database/data/cie/Illuminantd65.txt

CIE International Commission on Illumination - http://www.cie.co.at/ - (accessed 17 Sep 2008)
    Official website of the CIE.
    There are tables of the standard functions (matching functions, illuminants) here:
        http://www.cie.co.at/main/freepubs.html
        http://www.cie.co.at/publ/abst/datatables15_2004/x2.txt
        http://www.cie.co.at/publ/abst/datatables15_2004/y2.txt
        http://www.cie.co.at/publ/abst/datatables15_2004/z2.txt
        http://www.cie.co.at/publ/abst/datatables15_2004/sid65.txt
    ColorPy does not use these specific files.

Charles Poynton - Frequently asked questions about Gamma and Color,
    posted to comp.graphics.algorithms, 25 Jan 1995.

License:

Copyright (C) 2008 Mark Kness

Author - Mark Kness - mkness@alumni.utexas.net

This file is part of ColorPy.

ColorPy is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version.

ColorPy is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with ColorPy.  If not, see <http://www.gnu.org/licenses/>.
"""
import math, numpy
import yaml

import colormodels

# Assumed physical brightness of the monitor [W/m^2]
#   80 cd/m^2 * 20.3 mW/cd (assuming light at 556 nm)
DEFAULT_DISPLAY_INTENSITY = 1.624

# For reference, the physical luminance of several interesting objects ...
# All values in cd/m^2, where 1 cd = 1 candle = 20.3 milliwatts of light at 5560 A
# Typical monitor at full blast = 80 cd/m^2  [Poynton, Color FAQ, p.4]
# Candle                        = 5000
# 40W Frosted Light Bulb        = 25000
# Clear sky                     = 4000
# Moon                          = 2500
# Sun                           = 1.6 x 10^9
#
# an advertised LCD display (2008) = 300 cd/m^2

# table of 1931 CIE XYZ matching functions.
# data from: http://cvrl.ioo.ucl.ac.uk/database/data/cmfs/ciexyz31_1.txt
# massaged into this format.
with open("/home/balder/projects/skylight/colorpy/data/CIEXYZ-1931-table.yaml", 'r') as f:
    _CIEXYZ_1931_table = yaml.load(f)

# Public - default range of wavelengths in spectra (nm).
# start_wl_nm and end_wl_nm are integers, delta_wl_nm is a float.
start_wl_nm = None
end_wl_nm = None
delta_wl_nm = None

# Private - tables of spectral curves
_wavelengths = None
_xyz_colors = None
_xyz_deltas = None


def init(display_intensity=DEFAULT_DISPLAY_INTENSITY):
    """Initialize the spectral sampling curves."""
    # Expect that the table ranges from 360 to 830
    global start_wl_nm, end_wl_nm, delta_wl_nm
    table_size = len(_CIEXYZ_1931_table)
    start_wl_nm = 360
    end_wl_nm = 830
    delta_wl_nm = 1.0
    first = _CIEXYZ_1931_table.keys()[0]
    last = _CIEXYZ_1931_table.keys()[-1]
    assert (first == start_wl_nm), 'Expecting first wavelength as %d but instead is %d' % (start_wl_nm, first)
    assert (last == end_wl_nm), 'Expecting last wavelength as %d but instead is %d' % (end_wl_nm, last)
    assert (table_size == 471), 'Expecting 471 wavelength, each 1 nm from 360 to 830 nm, instead table size is %d' % (
        table_size)
    # Assume that the color for the wl just before and after the table (359 and 831) are zero.
    # Also assume linear interpolation of the values for in-between nanometer wavelengths.
    # Construct arrays, with elements for each wavelength, as the xyz color,
    # and the change in color to the next largest nanometer.
    # We will add an (empty) entry for 359 nm and 831 nm.
    global _wavelengths, _xyz_colors, _xyz_deltas
    create_table_size = table_size + 2
    _wavelengths = numpy.empty((create_table_size), int)
    _xyz_colors = numpy.empty((create_table_size, 3))
    _xyz_deltas = numpy.empty((create_table_size, 3))
    # fill in first row as 359 nm with zero color
    _wavelengths[0] = start_wl_nm - 1
    _xyz_colors[0] = colormodels.xyz_color(0.0, 0.0, 0.0)
    # fill in last row as 831 nm with zero color
    _wavelengths[create_table_size - 1] = end_wl_nm + 1
    _xyz_colors[create_table_size - 1] = colormodels.xyz_color(0.0, 0.0, 0.0)
    # fill in the middle rows from the source data
    for i in range(0, len(_CIEXYZ_1931_table)):
        wl = _CIEXYZ_1931_table.keys()[i]
        x, y, z = _CIEXYZ_1931_table[wl]
        _wavelengths[i + 1] = wl
        _xyz_colors[i + 1] = colormodels.xyz_color(x, y, z)
    # get the integrals of each curve
    integral = numpy.zeros(3)
    for i in range(0, create_table_size - 1):
        d_integral = 0.5 * (_xyz_colors[i] + _xyz_colors[i + 1]) * delta_wl_nm
        integral += d_integral
    # scale the sampling curves so that:
    #   A spectrum, constant with wavelength, with total intensity equal to the
    #   physical intensity of the monitor, will sample with Y = 1.0.
    # This scaling corresponds with that in colormodels, which assumes Y = 1.0 at full white.
    # Ideally, we would like the spectrum of the actual monitor display, at full white,
    #   to sample to Y = 1.0, not the constant with wavelength spectrum that is assumed here.
    num_wl = table_size
    scaling = num_wl / (integral[1] * display_intensity)
    _xyz_colors *= scaling
    # now calculate all the deltas
    for i in range(0, create_table_size - 1):
        _xyz_deltas[i] = _xyz_colors[i + 1] - _xyz_colors[i]
    _xyz_deltas[create_table_size - 1] = colormodels.xyz_color(0.0, 0.0, 0.0)


#

def empty_spectrum():
    """Get a black (no intensity) ColorPy spectrum.

    This is a 2D numpy array, with one row for each wavelength in the visible range,
    360 nm to 830 nm, with a spacing of delta_wl_nm (1.0 nm), and two columns.
    The first column is filled with the wavelength [nm].
    The second column is filled with 0.0.  It should later be filled with the intensity.

    The result can be passed to xyz_from_spectrum() to convert to an xyz color.
    """
    wl_nm_range = range(start_wl_nm, end_wl_nm + 1)
    num_wl = len(wl_nm_range)
    spectrum = numpy.zeros((num_wl, 2))
    for i in range(0, num_wl):
        spectrum[i][0] = float(wl_nm_range[i])
    return spectrum


def xyz_from_wavelength(wl_nm):
    """Given a wavelength (nm), return the corresponding xyz color, for unit intensity."""
    # separate wl_nm into integer and fraction
    int_wl_nm = math.floor(wl_nm)
    frac_wl_nm = wl_nm - float(int_wl_nm)
    # skip out of range (invisible) wavelengths
    if (int_wl_nm < start_wl_nm - 1) or (int_wl_nm > end_wl_nm + 1):
        return colormodels.xyz_color(0.0, 0.0, 0.0)
    # get index into main table
    index = int(int_wl_nm - start_wl_nm + 1)
    # apply linear interpolation to get the color
    return _xyz_colors[index] + frac_wl_nm * _xyz_deltas[index]


def xyz_from_spectrum(spectrum):
    """Determine the xyz color of the spectrum.

    The spectrum is assumed to be a 2D numpy array, with a row for each wavelength,
    and two columns.  The first column should hold the wavelength (nm), and the
    second should hold the light intensity.  The set of wavelengths can be arbitrary,
    it does not have to be the set that empty_spectrum() returns."""
    shape = numpy.shape(spectrum)
    (num_wl, num_col) = shape
    assert num_col == 2, 'Expecting 2D array with each row: wavelength [nm], specific intensity [W/unit solid angle]'
    # integrate
    rtn = colormodels.xyz_color(0.0, 0.0, 0.0)
    for i in range(0, num_wl):
        wl_nm_i = spectrum[i][0]
        specific_intensity_i = spectrum[i][1]
        xyz = xyz_from_wavelength(wl_nm_i)
        rtn += specific_intensity_i * xyz
    return rtn


def get_normalized_spectral_line_colors(
        brightness=1.0,
        num_purples=0,
        dwl_angstroms=10):
    """Get an array of xyz colors covering the visible spectrum.
    Optionally add a number of 'purples', which are colors interpolated between the color
    of the lowest wavelength (violet) and the highest (red).

    brightness - Desired maximum rgb component of each color.  Default 1.0.  (Maxiumum displayable brightness)
    num_purples - Number of colors to interpolate in the 'purple' range.  Default 0.  (No purples)
    dwl_angstroms - Wavelength separation, in angstroms (0.1 nm).  Default 10 A. (1 nm spacing)
    """
    # get range of wavelengths, in angstroms, so that we can have finer resolution than 1 nm
    wl_angstrom_range = range(10 * start_wl_nm, 10 * (end_wl_nm + 1), dwl_angstroms)
    # get total point count
    num_spectral = len(wl_angstrom_range)
    num_points = num_spectral + num_purples
    xyzs = numpy.empty((num_points, 3))
    # build list of normalized color x,y values proceeding along each wavelength
    i = 0
    for wl_A in wl_angstrom_range:
        wl_nm = wl_A * 0.1
        xyz = xyz_from_wavelength(wl_nm)
        colormodels.xyz_normalize(xyz)
        xyzs[i] = xyz
        i += 1
    # interpolate from end point to start point (filling in the purples)
    first_xyz = xyzs[0]
    last_xyz = xyzs[num_spectral - 1]
    for ipurple in range(0, num_purples):
        t = float(ipurple) / float(num_purples - 1)
        omt = 1.0 - t
        xyz = t * first_xyz + omt * last_xyz
        colormodels.xyz_normalize(xyz)
        xyzs[i] = xyz
        i += 1
    # scale each color to have the max rgb component equal to the desired brightness
    for i in range(0, num_points):
        rgb = colormodels.brightest_rgb_from_xyz(xyzs[i], brightness)
        xyzs[i] = colormodels.xyz_from_rgb(rgb)
    # done
    return xyzs


def get_normalized_spectral_line_colors_annotated(
        brightness=1.0,
        num_purples=0,
        dwl_angstroms=10):
    """Get an array of xyz colors covering the visible spectrum.
    Optionally add a number of 'purples', which are colors interpolated between the color
    of the lowest wavelength (violet) and the highest (red).
    A text string describing the color is supplied for each color.

    brightness - Desired maximum rgb component of each color.  Default 1.0.  (Maxiumum displayable brightness)
    num_purples - Number of colors to interpolate in the 'purple' range.  Default 0.  (No purples)
    dwl_angstroms - Wavelength separation, in angstroms (0.1 nm).  Default 10 A. (1 nm spacing)
    """
    # get range of wavelengths, in angstroms, so that we can have finer resolution than 1 nm
    wl_angstrom_range = range(10 * start_wl_nm, 10 * (end_wl_nm + 1), dwl_angstroms)
    # get total point count
    num_spectral = len(wl_angstrom_range)
    num_points = num_spectral + num_purples
    xyzs = numpy.empty((num_points, 3))
    names = []
    # build list of normalized color x,y values proceeding along each wavelength
    i = 0
    for wl_A in wl_angstrom_range:
        wl_nm = wl_A * 0.1
        xyz = xyz_from_wavelength(wl_nm)
        colormodels.xyz_normalize(xyz)
        xyzs[i] = xyz
        name = '%.1f nm' % wl_nm
        names.append(name)
        i += 1
    # interpolate from end point to start point (filling in the purples)
    first_xyz = xyzs[0]
    last_xyz = xyzs[num_spectral - 1]
    for ipurple in range(0, num_purples):
        t = float(ipurple) / float(num_purples - 1)
        omt = 1.0 - t
        xyz = t * first_xyz + omt * last_xyz
        colormodels.xyz_normalize(xyz)
        xyzs[i] = xyz
        name = '%03d purple' % math.floor(1000.0 * t + 0.5)
        names.append(name)
        i += 1
    # scale each color to have the max rgb component equal to the desired brightness
    for i in range(0, num_points):
        rgb = colormodels.brightest_rgb_from_xyz(xyzs[i], brightness)
        xyzs[i] = colormodels.xyz_from_rgb(rgb)
    # done
    return xyzs, names


# Initialize at module startup
init()
