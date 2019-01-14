r"""
Astronomical data.

Having imported the ``astro_data`` module, type

- ``astro_data.<TAB>``, where ``<TAB>`` stands for the tabulation key,
  to get the list of available data
- ``astro_data.??`` to see the sources of the numerical values

EXAMPLES:

Fundamental constants (in SI units)::

    sage: from kerrgeodesic_gw import astro_data
    sage: astro_data.G
    6.67408e-11
    sage: astro_data.c
    299792458.0

Solar mass in kg::

    sage: astro_data.Sun_mass_kg
    1.98848e+30

Solar mass in meters (geometrized units)::

    sage: astro_data.Sun_mass_m  # tol 1.0e-13
    1476.6284425812723

Solar mass in seconds (geometrized units)::

    sage: astro_data.Sun_mass_s  # tol 1.0e-13
    4.925502303934786e-06

Mass of Sgr A* in solar masses::

    sage: astro_data.SgrA_mass_sol
    4100000.0

Mass of Sgr A* in kg::

    sage: astro_data.SgrA_mass_kg  # tol 1.0e-13
    8.152768e+36

Mass of Sgr A* in meters (geometrized units)::

    sage: astro_data.SgrA_mass_m  # tol 1.0e-13
    6054176614.583217

Mass of Sgr A* in seconds (geometrized units)::

    sage: astro_data.SgrA_mass_s  # tol 1.0e-13
    20.19455944613262

Distance to Sgr A* in parsecs and meters, respectively::

    sage: astro_data.SgrA_distance_pc
    8120.0
    sage: astro_data.SgrA_distance_m  # tol 1.0e-13
    2.5055701960202522e+20

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
from sage.symbolic.constants import pi

# Fundamental constants in SI units
c = constants.c
G = constants.G

# Astronomical constants in SI units
pc = constants.parsec
yr = constants.year
au = constants.au

# Sun
Sun_mass_kg = 1.98848e30 # Particle Data Group, PRD 98, 030001 (2018); http://pdg.lbl.gov/
Sun_mass_m = G*Sun_mass_kg/c**2
Sun_mass_s = Sun_mass_m/c
Sun_eq_radius_m = 6.957e8  # IAU Resolution B3; arXiv:1510.07674
Sun_mean_density_SI = Sun_mass_kg/(4*pi.n()/3*Sun_eq_radius_m**3)
# aliases:
Msol = Sun_mass_kg
Msol_m = Sun_mass_m
Msol_s = Sun_mass_s

# Earth
Earth_mass_kg = 5.9724e24 # Particle Data Group, PRD 98, 030001 (2018); http://pdg.lbl.gov/
Earth_mass_m = G*Earth_mass_kg/c**2
Earth_mass_s = Earth_mass_m/c
Earth_mass_sol = Earth_mass_kg/Sun_mass_kg
Earth_eq_radius_m = 6.3781e6 # IAU Resolution B3; arXiv:1510.07674
Earth_mean_density_SI = Earth_mass_kg/(4*pi.n()/3*Earth_eq_radius_m**3)

# Jupiter
Jupiter_mass_kg = 1.89819e27 # https://ssd.jpl.nasa.gov/?planet_phys_par
Jupiter_mass_m = G*Jupiter_mass_kg/c**2
Jupiter_mass_s = Jupiter_mass_m/c
Jupiter_mass_sol = Jupiter_mass_kg/Sun_mass_kg
Jupiter_eq_radius_m = 7.1492e7 # IAU Resolution B3; arXiv:1510.07674
Jupiter_mean_radius_m = 6.9911e7 # https://en.wikipedia.org/wiki/Jupiter
Jupiter_mean_density_SI = Jupiter_mass_kg/(4*pi.n()/3*Jupiter_mean_radius_m**3)

# Brown dwarf of minimal radius
#  Source: dashed line (t=5 Gyr) in Fig. 1 of Chabrier et al. (2009)
#          https://doi.org/10.1063/1.3099078
brown_dwarf1_mass_kg = 65*Jupiter_mass_kg
brown_dwarf1_mass_sol = brown_dwarf1_mass_kg/Sun_mass_kg
brown_dwarf1_radius_m = 10**(-0.11)*Jupiter_mean_radius_m
brown_dwarf1_mean_density_SI = brown_dwarf1_mass_kg/(4*pi.n()/
                                                     3*brown_dwarf1_radius_m**3)
brown_dwarf1_mean_density_sol = brown_dwarf1_mean_density_SI/Sun_mean_density_SI

# Sagittarius A*
SgrA_mass_sol = 4.1e6 # Table A.1, GRAVITY col., A&A 615, L15 (2018)
SgrA_mass_kg = SgrA_mass_sol*Sun_mass_kg
SgrA_mass_m = G*SgrA_mass_kg/c**2
SgrA_mass_s = SgrA_mass_m/c
SgrA_distance_pc = 8.12e3 # Table A.1, GRAVITY col., A&A 615, L15 (2018)
SgrA_distance_m = SgrA_distance_pc*pc
# aliases:
MSgrA = SgrA_mass_kg
MSgrA_m = SgrA_mass_m
MSgrA_s = SgrA_mass_s
dSgrA = SgrA_distance_m

# M32
M32_mass_sol = 2.5e6 # Table 6 of Nguyen et al., ApJ 858, 118 (2018)
M32_mass_kg = M32_mass_sol*Sun_mass_kg
M32_mass_m = G*M32_mass_kg/c**2
M32_mass_s = M32_mass_m/c
M32_distance_pc = 7.9e5 # Table 4 of Nguyen et al., ApJ 858, 118 (2018)
M32_distance_m = M32_distance_pc*pc
