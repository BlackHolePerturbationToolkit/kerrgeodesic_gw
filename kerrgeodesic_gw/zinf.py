r"""
Functions `Z^\infty_{\ell m}(r)` giving the amplitude of the gravitational
radiation emitted by a particle on a circular orbit at radius `r` about
a Kerr black hole.

REFERENCES:

- \S. A. Teukolsky, Astrophys. J. **185**, 635 (1973)
- \S. Detweiler, Astrophys. J. **225**, 687 (1978)
- \M. Shibata, Phys. Rev. D **50**, 6297 (1994)
- \D. Kennefick, Phys. Rev. D **58**, 064012 (1998)
- \S. A. Hughes, Phys. Rev. D **61**, 084004 (2000) [:doi:`10.1103/PhysRevD.61.084004`]
- \E. Gourgoulhon, A. Le Tiec, F. Vincent, N. Warburton, Astron. Astrophys.
  **627**, A92 (2019) [:doi:`10.1051/0004-6361/201935406`]; :arXiv:`1903.02049`

"""
#******************************************************************************
#       Copyright (C) 2018 Eric Gourgoulhon <eric.gourgoulhon@obspm.fr>
#       Copyright (C) 2018 Niels Warburton <niels.warburton@ucd.ie>
#
#  Distributed under the terms of the GNU General Public License (GPL)
#  as published by the Free Software Foundation; either version 2 of
#  the License, or (at your option) any later version.
#                  http://www.gnu.org/licenses/
#******************************************************************************

import os
from sage.calculus.interpolation import spline
from sage.functions.other import sqrt
from sage.functions.log import ln
from sage.rings.real_double import RDF
from sage.rings.complex_double import CDF
from sage.symbolic.constants import pi
from sage.symbolic.all import i as I

_cached_splines = {}

def Zinf_Schwarzchild_PN(l, m, r):
    r"""
    Amplitude factor of the mode `(\ell,m)` for a Schwarzschild BH at the 1.5PN
    level.

    The 1.5PN formulas are taken from E. Poisson, Phys. Rev. D **47**, 1497
    (1993), :doi:`10.1103/PhysRevD.47.1497`.

    INPUT:

    - ``l`` -- integer >= 2; the harmonic degree `\ell`
    - ``m`` -- integer within the range ``[-l, l]``; the azimuthal number `m`
    - ``r`` -- areal radius of the orbit (in units of `M`, the BH mass)

    OUTPUT:

    - coefficient `Z^\infty_{\ell m}(r)` (in units of `M^{-2}`)

    EXAMPLES::

        sage: from kerrgeodesic_gw import Zinf_Schwarzchild_PN
        sage: Zinf_Schwarzchild_PN(2, 2, 6.)  # tol 1.0e-13
        -0.00981450418730346 + 0.003855681972781947*I
        sage: Zinf_Schwarzchild_PN(5, 3, 6.)  # tol 1.0e-13
        -6.958527913913504e-05*I

    """
    if m < 0:
        return (-1)**l * Zinf_Schwarzchild_PN(l, -m, r).conjugate()
    v = 1./sqrt(r)
    if l == 2:
        b = RDF(sqrt(pi/5.)/r**4)
        if m == 1:
            return CDF(4./3.*I*b*v*(1 - 17./28.*v**2))
        if m == 2:
            return CDF(-16*b*(1 - 107./42.*v**2
                        + (2*pi + 4*I*(3*ln(2*v) - 0.839451001765134))*v**3))
    if l == 3:
        b = RDF(sqrt(pi/7.)/r**(4.5))
        if m == 1:
            return CDF(I/3.*b/sqrt(10.)*(1 - 8./3.*v**2))
        if m == 2:
            return CDF(-16./3.*b*v)
        if m == 3:
            return CDF(-81*I*b/sqrt(8.)*(1 - 4*v**2))
    if l == 4:
        b = RDF(sqrt(pi)/r**5)
        if m == 1:
            return CDF(I/105.*b/sqrt(2)*v)
        if m == 2:
            return CDF(-16./63.*b)
        if m == 3:
            return CDF(-81./5.*I*b/sqrt(14.)*v)
        if m == 4:
            return CDF(512./9.*b/sqrt(7.))
    if l == 5:
        b = RDF(sqrt(pi)/r**(5.5))
        if m == 1:
            return CDF(I/360.*b/sqrt(77.))
        if m == 2:
            return CDF(0)
        if m == 3:
            return CDF(-81./40.*I*b*sqrt(3./22.))
        if m == 4:
            return CDF(0)
        if m == 5:
            return CDF(3125./24.*I*b*sqrt(5./66.))
    raise NotImplementedError("{} not implemented".format((l, m)))

def Zinf(a, l, m, r, algorithm='spline'):
    r"""
    Amplitude factor of the mode `(\ell,m)`.

    The factor `Z^\infty_{\ell m}(r)` is obtained by spline interpolation
    of tabulated numerical solutions of the radial component of the
    Teukolsky equation.

    INPUT:

    - ``a`` -- BH angular momentum parameter (in units of `M`, the BH mass)
    - ``l`` -- integer >= 2; the harmonic degree `\ell`
    - ``m`` -- integer within the range ``[-l, l]``; the azimuthal number `m`
    - ``r`` -- areal radius of the orbit (in units of `M`)
    - ``algorithm`` -- (default: ``'spline'``) string describing the
      computational method; allowed values are

      - ``'spline'``: spline interpolation of tabulated data
      - ``'1.5PN'`` (only for ``a=0``): 1.5-post-Newtonian expansion following
        E. Poisson, Phys. Rev. D **47**, 1497 (1993)
        [:doi:`10.1103/PhysRevD.47.1497`], with a minus one factor
        accounting for a different convention for the metric signature.

    OUTPUT:

    - coefficient `Z^\infty_{\ell m}(r)` (in units of `M^{-2}`)

    EXAMPLES::

        sage: from kerrgeodesic_gw import Zinf
        sage: Zinf(0.98, 2, 2, 1.7)  # tol 1.0e-13
        -0.04302234478778856 + 0.28535368610053824*I
        sage: Zinf(0., 2, 2, 10.)  # tol 1.0e-13
        0.0011206407919254163 - 0.0003057608384581628*I
        sage: Zinf(0., 2, 2, 10., algorithm='1.5PN')  # tol 1.0e-13
        0.0011971529546749354 - 0.0003551610880408921*I

    """
    if m < 0:
        return (-1)**l * Zinf(a, l, -m, r, algorithm=algorithm).conjugate()
    if algorithm == '1.5PN':
        if a == 0:
            # the factor (-1) below accounts for a difference of signature with
            # Poisson (1993):
            return -Zinf_Schwarzchild_PN(l, m, r)
        raise ValueError("a must be zero for algorithm='1.5PN'")
    a = RDF(a)
    param = (a, l, m)
    if param in _cached_splines:
        splines = _cached_splines[param]
    else:
        file_name = "data/Zinf_a{:.1f}.dat".format(float(a)) if a <= 0.9 \
                    else "data/Zinf_a{:.2f}.dat".format(float(a))
        file_name = os.path.join(os.path.dirname(__file__), file_name)
        r_high_l = 20. if a <= 0.9 else 10.
        with open(file_name, "r") as data_file:
            lm_values = []        # l values up to 10 (for r <= r_high_l)
            lm_values_low = []    # l values up to 5 only (for r > r_high_l)
            for ld in range(2, 6):
                for md in range(1, ld+1):
                    lm_values_low.append((ld, md))
            for ld in range(2, 11):
                for md in range(1, ld+1):
                    lm_values.append((ld, md))
            Zreal = {}
            Zimag = {}
            for (ld, md) in lm_values:
                Zreal[(ld, md)] = []
                Zimag[(ld, md)] = []
            for line in data_file:
                items = line.split('\t')
                rd = RDF(items.pop(0))
                if rd <= r_high_l:
                    for (ld, md) in lm_values:
                        Zreal[(ld, md)].append((rd, RDF(items.pop(0))))
                        Zimag[(ld, md)].append((rd, RDF(items.pop(0))))
                else:
                    for (ld, md) in lm_values_low:
                        Zreal[(ld, md)].append((rd, RDF(items.pop(0))))
                        Zimag[(ld, md)].append((rd, RDF(items.pop(0))))
        for (ld, md) in lm_values:
            _cached_splines[(a,) + (ld, md)] = (spline(Zreal[(ld, md)]),
                                                spline(Zimag[(ld, md)]))
        if param not in _cached_splines:
            raise ValueError("Zinf: case (a, l, m) = {} not implemented".format(param))
        splines = _cached_splines[param]
    # The factor (-1)**(l+m) below accounts for a difference of convention
    # in the C++ code used to produce the data files
    return (-1)**(l+m)*CDF(splines[0](r), splines[1](r))

def _lmax(a, r):
    r"""
    Utility function giving the maximal value of the harmonic index `\ell`
    for tabulated values of `Z^\infty_{\ell m}(r)`.

    INPUT:

    - ``a`` -- BH angular momentum parameter (in units of `M`, the BH mass)
    - ``r`` -- areal radius of the orbit (in units of `M`)

    OUTPUT:

    - maximal value of `\ell` for which `Z^\infty_{\ell m}(r)` is tabulated

    EXAMPLES::

        sage: from kerrgeodesic_gw.zinf import _lmax
        sage: _lmax(0, 6.)
        10
        sage: _lmax(0, 40.)
        5
        sage: _lmax(0.9, 12.)
        10
        sage: _lmax(0.95, 12.)
        5

    """
    r_high_l = 20. if a <= 0.9 else 10.
    l_max = 10 if r <= r_high_l else 5
    return l_max
