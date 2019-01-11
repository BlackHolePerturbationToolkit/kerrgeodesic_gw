r"""
Astronomical data.

Mass of Sgr A* in kg::

    sage: from kerrgeodesic_gw import astro_data
    sage: astro_data.MSgrA  # tol 1.0e-13
    8.152768e+36

Mass of Sgr A* in millions of solar masses::

    sage: astro_data.MSgrA / (1.e6*astro_data.Msol)  # tol 1.0e-13
    4.10000000000000

Mass of Sgr A* in meters (geometrized units)::

    sage: astro_data.MSgrA_m  # tol 1.0e-13
    6054176614.583217

Mass of Sgr A* in seconds (geometrized units)::

    sage: astro_data.MSgrA_s  # tol 1.0e-13
    20.19455944613262

"""
#******************************************************************************
#       Copyright (C) 2018 Eric Gourgoulhon <eric.gourgoulhon@obspm.fr>
#
#  Distributed under the terms of the GNU General Public License (GPL)
#  as published by the Free Software Foundation; either version 2 of
#  the License, or (at your option) any later version.
#                  http://www.gnu.org/licenses/
#******************************************************************************

import scipy.constants as constants

# Fundamental constants in SI units
c = constants.c
G = constants.G

# Astronomical constants in SI units
pc = constants.parsec
yr = constants.year

# Solar mass in SI units (kg)
Msol = 1.98848e30
# Solar mass in meters
Msol_m = G*Msol/c**2
# Solar mass in seconds
Msol_s = G*Msol/c**3

# Sgr A* mass in SI units (kg)
MSgrA = 4.1e6*Msol
# Sgr A* mass in meters
MSgrA_m = G*MSgrA/c**2
# Sgr A* mass in seconds
MSgrA_s = G*MSgrA/c**3
# Distance to Sgr A* in SI units (m)
dSgrA = 8.12e3*pc

# M32 mass in SI units (kg)
MM32 = 2.5e6*Msol
# M32 mass in meters
MM32_m = G*MM32/c**2
# M32 mass in seconds
MM32_s = G*MM32/c**3
# Distance to M32 in SI units (m)
dM32 = 7.9e5*pc

