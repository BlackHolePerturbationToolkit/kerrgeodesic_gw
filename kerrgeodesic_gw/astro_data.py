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

    sage: astro_data.solar_mass_kg
    1.98848e+30

Solar mass in meters (geometrized units)::

    sage: astro_data.solar_mass_m  # tol 1.0e-13
    1476.6284425812723

Solar mass in seconds (geometrized units)::

    sage: astro_data.solar_mass_s  # tol 1.0e-13
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
solar_mass_kg = 1.98848e30 # Particle Data Group, PRD 98, 030001 (2018); http://pdg.lbl.gov/
solar_mass_m = G*solar_mass_kg/c**2
solar_mass_s = solar_mass_m/c
solar_eq_radius_m = 6.957e8  # IAU Resolution B3; arXiv:1510.07674
solar_mean_density_SI = solar_mass_kg/(4*pi.n()/3*solar_eq_radius_m**3)
# aliases:
Msol = solar_mass_kg
Msol_m = solar_mass_m
Msol_s = solar_mass_s

# Earth
Earth_mass_kg = 5.9724e24 # Particle Data Group, PRD 98, 030001 (2018); http://pdg.lbl.gov/
Earth_mass_m = G*Earth_mass_kg/c**2
Earth_mass_s = Earth_mass_m/c
Earth_mass_sol = Earth_mass_kg/solar_mass_kg
Earth_eq_radius_m = 6.3781e6 # IAU Resolution B3; arXiv:1510.07674
Earth_mean_density_SI = Earth_mass_kg/(4*pi.n()/3*Earth_eq_radius_m**3)
Earth_mean_density_sol = Earth_mean_density_SI/solar_mean_density_SI

# Jupiter
Jupiter_mass_kg = 1.89819e27 # https://ssd.jpl.nasa.gov/?planet_phys_par
Jupiter_mass_m = G*Jupiter_mass_kg/c**2
Jupiter_mass_s = Jupiter_mass_m/c
Jupiter_mass_sol = Jupiter_mass_kg/solar_mass_kg
Jupiter_eq_radius_m = 7.1492e7 # IAU Resolution B3; arXiv:1510.07674
Jupiter_mean_radius_m = 6.9911e7 # https://en.wikipedia.org/wiki/Jupiter
Jupiter_mean_density_SI = Jupiter_mass_kg/(4*pi.n()/3*Jupiter_mean_radius_m**3)
Jupiter_mean_density_sol = Jupiter_mean_density_SI/solar_mean_density_SI

# Brown dwarf of minimal radius
#  Source: dashed line (t=5 Gyr) in Fig. 1 of Chabrier et al. (2009)
#          https://doi.org/10.1063/1.3099078
brown_dwarf1_mass_kg = 65*Jupiter_mass_kg
brown_dwarf1_mass_sol = brown_dwarf1_mass_kg/solar_mass_kg
brown_dwarf1_radius_m = 10**(-0.11)*Jupiter_mean_radius_m
brown_dwarf1_mean_density_SI = brown_dwarf1_mass_kg/(4*pi.n()/
                                                     3*brown_dwarf1_radius_m**3)
brown_dwarf1_mean_density_sol = brown_dwarf1_mean_density_SI/solar_mean_density_SI

# B8V star
#  Source: Table 1 of Silaj et al., ApJ 795, 82 (2014)
#          https://doi.org/10.1088/0004-637X/795/1/82
#          (reproduced in https://en.wikipedia.org/wiki/B-type_main-sequence_star)
B8Vstar_mass_sol = 3.8
B8Vstar_mass_kg = B8Vstar_mass_sol*solar_mass_kg
B8Vstar_radius_sol = 3.0
B8Vstar_radius_m = B8Vstar_radius_sol*solar_eq_radius_m
B8Vstar_mean_density_sol = B8Vstar_mass_sol/B8Vstar_radius_sol**3
B8Vstar_mean_density_SI = B8Vstar_mean_density_sol*solar_mean_density_SI

# A0V star
#  Source: Table 1 of Adelman, Proc. IAU Symp. 224, 1 (2004)
#          https://doi.org/10.1017/S1743921304004314
A0Vstar_mass_sol = 2.40
A0Vstar_mass_kg = A0Vstar_mass_sol*solar_mass_kg
A0Vstar_radius_sol = 1.87
A0Vstar_radius_m = A0Vstar_radius_sol*solar_eq_radius_m
A0Vstar_mean_density_sol = A0Vstar_mass_sol/A0Vstar_radius_sol**3
A0Vstar_mean_density_SI = A0Vstar_mean_density_sol*solar_mean_density_SI

# M3V star
#  Source: Table 1 of Kaltenegger & Traub, ApJ 698 519 (2009)
#          https://doi.org/10.1088/0004-637X/698/1/519
#          (reproduced in https://en.wikipedia.org/wiki/Red_dwarf)
M3Vstar_mass_sol = 0.36
M3Vstar_mass_kg = M3Vstar_mass_sol*solar_mass_kg
M3Vstar_radius_sol = 0.39
M3Vstar_radius_m = M3Vstar_radius_sol*solar_eq_radius_m
M3Vstar_mean_density_sol = M3Vstar_mass_sol/M3Vstar_radius_sol**3
M3Vstar_mean_density_SI = M3Vstar_mean_density_sol*solar_mean_density_SI

# M4V star
#  Source: mass: Table 1 of Kaltenegger & Traub, ApJ 698 519 (2009)
#          https://doi.org/10.1088/0004-637X/698/1/519
#          (reproduced in https://en.wikipedia.org/wiki/Red_dwarf)
#          radius: Fig. 1 of Chabrier et al., A&A 472, L17 (2007)
M4Vstar_mass_sol = 0.20
M4Vstar_mass_kg = M4Vstar_mass_sol*solar_mass_kg
M4Vstar_radius_sol = 0.22
M4Vstar_radius_m = M4Vstar_radius_sol*solar_eq_radius_m
M4Vstar_mean_density_sol = M4Vstar_mass_sol/M4Vstar_radius_sol**3
M4Vstar_mean_density_SI = M4Vstar_mean_density_sol*solar_mean_density_SI

# Sagittarius A*
SgrA_mass_sol = 4.1e6  # Table A.1, GRAVITY col., A&A 615, L15 (2018)
SgrA_mass_kg = SgrA_mass_sol*solar_mass_kg
SgrA_mass_m = G*SgrA_mass_kg/c**2
SgrA_mass_s = SgrA_mass_m/c
SgrA_distance_pc = 8.12e3  # Table A.1, GRAVITY col., A&A 615, L15 (2018)
SgrA_distance_m = SgrA_distance_pc*pc
# aliases:
MSgrA = SgrA_mass_kg
MSgrA_m = SgrA_mass_m
MSgrA_s = SgrA_mass_s
dSgrA = SgrA_distance_m

# M32
M32_mass_sol = 2.5e6  # Table 6 of Nguyen et al., ApJ 858, 118 (2018)
M32_mass_kg = M32_mass_sol*solar_mass_kg
M32_mass_m = G*M32_mass_kg/c**2
M32_mass_s = M32_mass_m/c
M32_distance_pc = 7.9e5  # Table 4 of Nguyen et al., ApJ 858, 118 (2018)
M32_distance_m = M32_distance_pc*pc

# M87
M87_mass_sol = 6.2e9  # Gebhardt et al., ApJ 729, 119 (2010)
M87_mass_kg = M87_mass_sol*solar_mass_kg
M87_mass_m = G*M87_mass_kg/c**2
M87_mass_s = M87_mass_m/c
M87_distance_pc = 1.67e7  # Bird et al., A&A 524, A71 (2010)
                          # Blakeslee et al., ApJ 694, 556 (2009)
M87_distance_m = M87_distance_pc*pc

