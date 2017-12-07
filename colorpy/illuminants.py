"""
illuminants.py - Definitions of some standard illuminants.

Description:

Illuminants are spectrums, normalized so that Y = 1.0.

Spectrums are 2D numpy arrays, with one row for each wavelength,
with the first column holding the wavelength in nm, and the
second column the intensity.

The spectrums have a wavelength increment of 1 nm.

Functions:

init () -
    Initialize CIE Illuminant D65.  This runs on module startup.

get_illuminant_D65 () -
    Get CIE Illuminant D65, as a spectrum, normalized to Y = 1.0.

    CIE standard illuminant D65 represents a phase of natural daylight
    with a correlated color temperature of approximately 6504 K.  (Wyszecki, p. 144)

    In the interest of standardization the CIE recommends that D65 be used
    whenever possible.  Otherwise, D55 or D75 are recommended.  (Wyszecki, p. 145)

    (ColorPy does not currently provide D55 or D75, however.)

get_illuminant_A () -
    Get CIE Illuminant A, as a spectrum, normalized to Y = 1.0.
    This is actually a blackbody illuminant for T = 2856 K.  (Wyszecki, p. 143)

get_blackbody_illuminant (T_K) -
    Get the spectrum of a blackbody at the given temperature, normalized to Y = 1.0.

get_constant_illuminant () -
    Get an illuminant, with spectrum constant over wavelength, normalized to Y = 1.0.

scale_illuminant (illuminant, scaling) -
    Scale the illuminant intensity by the specfied factor.

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
import ciexyz
import blackbody
import plots
import yaml

# table of CIE Illuminant D65 spectrum.
# data from: http://cvrl.ioo.ucl.ac.uk/database/data/cie/Illuminantd65.txt
# massaged into this format.
with open("/home/thor/projects/skylight/colorpy/data/Illuminant-D65.yaml", 'r') as f:
    _Illuminant_D65_table = yaml.load(f)
_Illuminant_D65 = None


def init():
    """Initialize CIE Illuminant D65.  This runs on module startup."""
    first_wl = _Illuminant_D65_table.keys()[0]
    # for now, only consider the part in the normal visible range (360-830 nm)
    first_index = ciexyz.start_wl_nm - first_wl
    table_first = _Illuminant_D65_table.keys()[first_index]
    assert (table_first == 360), 'Mismatch finding 360 nm entry in D65 table'
    global _Illuminant_D65
    _Illuminant_D65 = ciexyz.empty_spectrum()
    num_wl, num_cols = _Illuminant_D65.shape
    for i in range(0, num_wl):
        table = _Illuminant_D65_table.keys()[first_index + i]
        _Illuminant_D65[i][1] = _Illuminant_D65_table[table]
    # normalization - illuminant is scaled so that Y = 1.0
    xyz = ciexyz.xyz_from_spectrum(_Illuminant_D65)
    scaling = 1.0 / xyz[1]
    _Illuminant_D65[:, 1] *= scaling


#
# Get any of the available illuminants - D65, A, any blackbody, or a constant spectrum.
# ColorPy does not currently provide D55 or D75.
#

def get_illuminant_D65():
    """Get CIE Illuminant D65, as a spectrum, normalized to Y = 1.0.

    CIE standard illuminant D65 represents a phase of natural daylight
    with a correlated color temperature of approximately 6504 K.  (Wyszecki, p. 144)

    In the interest of standardization the CIE recommends that D65 be used
    whenever possible.  Otherwise, D55 or D75 are recommended.  (Wyszecki, p. 145)

    (ColorPy does not currently provide D55 or D75, however.)"""
    illuminant = _Illuminant_D65.copy()
    return illuminant


def get_illuminant_A():
    """Get CIE Illuminant A, as a spectrum, normalized to Y = 1.0.
    This is actually a blackbody illuminant for T = 2856 K.  (Wyszecki, p. 143)"""
    illuminant = get_blackbody_illuminant(2856.0)
    return illuminant


def get_blackbody_illuminant(T_K):
    """Get the spectrum of a blackbody at the given temperature, normalized to Y = 1.0."""
    illuminant = blackbody.blackbody_spectrum(T_K)
    xyz = ciexyz.xyz_from_spectrum(illuminant)
    if xyz[1] != 0.0:
        scaling = 1.0 / xyz[1]
        illuminant[:, 1] *= scaling
    return illuminant


def get_constant_illuminant():
    """Get an illuminant, with spectrum constant over wavelength, normalized to Y = 1.0."""
    illuminant = ciexyz.empty_spectrum()
    (num_wl, num_cols) = illuminant.shape
    for i in range(0, num_wl):
        illuminant[i][1] = 1.0
    xyz = ciexyz.xyz_from_spectrum(illuminant)
    if xyz[1] != 0.0:
        scaling = 1.0 / xyz[1]
        illuminant[:, 1] *= scaling
    return illuminant


# Scale an illuminant by an arbitrary factor

def scale_illuminant(illuminant, scaling):
    """Scale the illuminant intensity by the specfied factor."""
    illuminant[:, 1] *= scaling
    return illuminant


# Initialize at module startup
init()


# Figures - Plot some of the illuminants

def figures():
    """Plot spectra for several illuminants."""
    # D65
    plots.spectrum_plot(
        get_illuminant_D65(), 'CIE Illuminant D65', 'Illuminant-D65')
    # A
    plots.spectrum_plot(
        get_illuminant_A(), 'CIE Illuminant A', 'Illuminant-A')
    # Constant
    plots.spectrum_plot(
        get_constant_illuminant(), 'Constant Illuminant', 'Illuminant-Const')
    # Blackbody (5778)
    plots.spectrum_plot(
        get_blackbody_illuminant(5778.0), '5778 K Illuminant', 'Illuminant-5778')
