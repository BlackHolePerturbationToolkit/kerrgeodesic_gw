r"""
Spin-weighted spherical harmonics `{}_s Y_l^m(\theta,\phi)`

"""
#******************************************************************************
#       Copyright (C) 2018 Eric Gourgoulhon <eric.gourgoulhon@obspm.fr>
#
#  Distributed under the terms of the GNU General Public License (GPL)
#  as published by the Free Software Foundation; either version 2 of
#  the License, or (at your option) any later version.
#                  http://www.gnu.org/licenses/
#******************************************************************************

from sage.rings.integer_ring import ZZ
from sage.rings.real_double import RDF
from sage.rings.real_mpfr import RealField_class
from sage.functions.other import binomial, factorial, sqrt
from sage.functions.log import exp
from sage.functions.trig import cos, sin
from sage.symbolic.constants import pi
from sage.symbolic.all import i as I
from sage.symbolic.expression import Expression

_cached_functions = {}
_cached_values = {}

def _compute_sw_spherical_harm(s, l, m, theta, phi, condon_shortley=True,
                               numerical=None):
    r"""
    Compute the spin-weighted spherical harmonic of spin weight ``s`` and
    indices ``(l,m)`` as a callable symbolic expression in (theta,phi)

    INPUT:

    - ``s`` -- integer; the spin weight
    - ``l`` -- non-negative integer; the harmonic degree
    - ``m`` -- integer within the range ``[-l, l]``; the azimuthal number
    - ``theta`` -- colatitude angle
    - ``phi`` -- azimuthal angle
    - ``condon_shortley`` -- (default: ``True``) determines whether the
      Condon-Shortley phase of `(-1)^m` is taken into account (see below)
    - ``numerical`` -- (default: ``None``) determines whether a symbolic or
      a numerical computation of a given type is performed; allowed values are

      - ``None``: a symbolic computation is performed
      - ``RDF``: Sage's machine double precision floating-point numbers
        (``RealDoubleField``)
      - ``RealField(n)``, where ``n`` is a number of bits: Sage's
        floating-point numbers with an arbitrary precision; note that ``RR`` is
        a shortcut for ``RealField(53)``.
      - ``float``: Python's floating-point numbers


    OUTPUT:

    - `{}_s Y_l^m(\theta,\phi)` either

      - as a symbolic expression if ``numerical`` is ``None``
      - or a pair of floating-point numbers, each of them being of the type
        corresponding to ``numerical`` and representing respectively the
        real and imaginary parts of `{}_s Y_l^m(\theta,\phi)`

    ALGORITHM:

    The spin-weighted spherical harmonic is evaluated according to Eq. (3.1)
    of J. N. Golberg et al., J. Math. Phys. **8**, 2155 (1967)
    [:doi:`10.1063/1.1705135`], with an extra `(-1)^m` factor (the so-called
    *Condon-Shortley phase*) if ``condon_shortley`` is ``True``, the actual
    formula being then the one given in
    :wikipedia:`Spin-weighted_spherical_harmonics#Calculating`

    TESTS::

        sage: from kerrgeodesic_gw.spin_weighted_spherical_harm import _compute_sw_spherical_harm
        sage: theta, phi = var("theta phi")
        sage: _compute_sw_spherical_harm(-2, 2, 1, theta, phi)
        1/4*(sqrt(5)*cos(theta) + sqrt(5))*e^(I*phi)*sin(theta)/sqrt(pi)

    """
    if abs(s)>l:
        return ZZ(0)
    if abs(theta) < 1.e-6:     # TODO: fix the treatment of small theta values
        if theta < 0:          #       possibly with exact formula for theta=0
            theta = -1.e-6     #
        else:                  #
            theta = 1.e-6      #
    cott2 = cos(theta/2)/sin(theta/2)
    res = 0
    for r in range(l-s+1):
        res += (-1)**(l-r-s) * (binomial(l-s, r) * binomial(l+s, r+s-m)
                                * cott2**(2*r+s-m))
    res *= sin(theta/2)**(2*l)
    ff = factorial(l+m)*factorial(l-m)*(2*l+1) / (factorial(l+s)*factorial(l-s))
    if numerical:
        pre = sqrt(numerical(ff)/numerical(pi))/2
    else:
        pre = sqrt(ff)/(2*sqrt(pi))
    res *= pre
    if condon_shortley:
        res *= (-1)**m
    if numerical:
        return (numerical(res*cos(m*phi)), numerical(res*sin(m*phi)))
    # Symbolic case:
    res = res.simplify_full()
    res = res.reduce_trig()    # get rid of cos(theta/2) and sin(theta/2)
    res = res.simplify_trig()  # further trigonometric simplifications
    res *= exp(I*m*phi)
    return res


def spin_weighted_spherical_harmonic(s, l, m, theta, phi,
                                     condon_shortley=True, cached=True,
                                     numerical=None):
    r"""
    Return the spin-weighted spherical harmonic of spin weight ``s`` and
    indices ``(l,m)``.

    INPUT:

    - ``s`` -- integer; the spin weight
    - ``l`` -- non-negative integer; the harmonic degree
    - ``m`` -- integer within the range ``[-l, l]``; the azimuthal number
    - ``theta`` -- colatitude angle
    - ``phi`` -- azimuthal angle
    - ``condon_shortley`` -- (default: ``True``) determines whether the
      Condon-Shortley phase of `(-1)^m` is taken into account (see below)
    - ``cached`` -- (default: ``True``) determines whether the result shall be
      cached; setting ``cached`` to ``False`` forces a new computation, without
      caching the result
    - ``numerical`` -- (default: ``None``) determines whether a symbolic or
      a numerical computation of a given type is performed; allowed values are

      - ``None``: the type of computation is deduced from the type of ``theta``
      - ``RDF``: Sage's machine double precision floating-point numbers
        (``RealDoubleField``)
      - ``RealField(n)``, where ``n`` is a number of bits: Sage's
        floating-point numbers with an arbitrary precision; note that ``RR`` is
        a shortcut for ``RealField(53)``.
      - ``float``: Python's floating-point numbers

    OUTPUT:

    - the value of `{}_s Y_l^m(\theta,\phi)`, either as a symbolic expression
      or as floating-point complex number of the type determined by
      ``numerical``

    ALGORITHM:

    The spin-weighted spherical harmonic is evaluated according to Eq. (3.1)
    of J. N. Golberg et al., J. Math. Phys. **8**, 2155 (1967)
    [:doi:`10.1063/1.1705135`], with an extra `(-1)^m` factor (the so-called
    *Condon-Shortley phase*) if ``condon_shortley`` is ``True``, the actual
    formula being then the one given in
    :wikipedia:`Spin-weighted_spherical_harmonics#Calculating`

    EXAMPLES::

        sage: from kerrgeodesic_gw import spin_weighted_spherical_harmonic
        sage: theta, phi = var('theta phi')
        sage: spin_weighted_spherical_harmonic(-2, 2, 1, theta, phi)
        1/4*(sqrt(5)*cos(theta) + sqrt(5))*e^(I*phi)*sin(theta)/sqrt(pi)
        sage: spin_weighted_spherical_harmonic(-2, 2, 1, theta, phi,
        ....:                                  condon_shortley=False)
        -1/4*(sqrt(5)*cos(theta) + sqrt(5))*e^(I*phi)*sin(theta)/sqrt(pi)
        sage: spin_weighted_spherical_harmonic(-2, 2, 1, pi/3, pi/4)
        (3/32*I + 3/32)*sqrt(5)*sqrt(3)*sqrt(2)/sqrt(pi)


    Evaluation as floating-point numbers: the type of the output is deduced
    from the input::

        sage: spin_weighted_spherical_harmonic(-2, 2, 1, 1.0, 2.0)  # tol 1.0e-13
        -0.170114676286891 + 0.371707349012686*I
        sage: parent(_)
        Complex Field with 53 bits of precision
        sage: spin_weighted_spherical_harmonic(-2, 2, 1, RDF(2.0), RDF(3.0))  # tol 1.0e-13
        -0.16576451879564585 + 0.023629159118690464*I
        sage: parent(_)
        Complex Double Field
        sage: spin_weighted_spherical_harmonic(-2, 2, 1, float(3.0), float(4.0))  # tol 1.0e-13
        (-0.0002911423884400524-0.00033709085352998027j)
        sage: parent(_)
        <type 'complex'>

    Computation with arbitrary precision are possible (here 100 bits)::

        sage: R100 = RealField(100); R100
        Real Field with 100 bits of precision
        sage: spin_weighted_spherical_harmonic(-2, 2, 1, R100(1.5), R100(2.0))  # tol 1.0e-28
        -0.14018136537676185317636108802 + 0.30630187143465275236861476906*I

    Even when the entry is symbolic, numerical evaluation can be enforced via
    the argument ``numerical``. For instance, setting ``numerical`` to ``RDF``
    (SageMath's Real Double Field)::

        sage: spin_weighted_spherical_harmonic(-2, 2, 1, pi/3, pi/4, numerical=RDF)  # tol 1.0e-13
        0.2897056515173923 + 0.28970565151739225*I
        sage: parent(_)
        Complex Double Field

    One can also use ``numerical=RR`` (SageMath's Real Field with precision set
    to 53 bits)::

        sage: spin_weighted_spherical_harmonic(-2, 2, 1, pi/3, pi/4, numerical=RR)   # tol 1.0e-13
        0.289705651517392 + 0.289705651517392*I
        sage: parent(_)
        Complex Field with 53 bits of precision

    Another option is to use Python floats::

        sage: spin_weighted_spherical_harmonic(-2, 2, 1, pi/3, pi/4, numerical=float)  # tol 1.0e-13
        (0.28970565151739225+0.2897056515173922j)
        sage: parent(_)
        <type 'complex'>

    One can go beyond double precision, for instance using 100 bits of
    precision::

        sage: spin_weighted_spherical_harmonic(-2, 2, 1, pi/3, pi/4,
        ....:                                  numerical=RealField(100))  # tol 1.0e-28
        0.28970565151739218525664455148 + 0.28970565151739218525664455148*I
        sage: parent(_)
        Complex Field with 100 bits of precision

    """
    global _cached_functions, _cached_values
    s = ZZ(s)  # ensure that we are dealing with Sage integers
    l = ZZ(l)
    m = ZZ(m)
    if abs(s)>l:
        return ZZ(0)
    if (isinstance(theta, Expression) and theta.variables() == (theta,) and
            isinstance(phi, Expression) and phi.variables() == (phi,)):
        # Evaluation as a symbolic function
        if cached:
            param = (s, l, m, condon_shortley)
            if param not in _cached_functions:
                _cached_functions[param] = _compute_sw_spherical_harm(s, l, m,
                    theta, phi, condon_shortley=condon_shortley,
                    numerical=None).function(theta, phi)
            return _cached_functions[param](theta, phi)
        return _compute_sw_spherical_harm(s, l, m, theta, phi,
                                          condon_shortley=condon_shortley,
                                          numerical=None)
    # Numerical or symbolic evaluation
    param = (s, l, m, theta, phi, condon_shortley, str(numerical))
    if not cached or param not in _cached_values:
        # a new computation is required
        if numerical:
            # theta and phi are enforced to be of the type defined by numerical
            theta = numerical(theta)
            phi = numerical(phi)
        else:
            # the type of computation is deduced from that of theta
            if type(theta) is float:
                numerical = float
            elif (theta.parent() is RDF or
                  isinstance(theta.parent(), RealField_class)):
                numerical = theta.parent()
        res = _compute_sw_spherical_harm(s, l, m, theta, phi,
                                         condon_shortley=condon_shortley,
                                         numerical=numerical)
        if numerical is RDF or isinstance(numerical, RealField_class):
            res = numerical.complex_field()(res)
        elif numerical is float:
            res = complex(*res)
        if cached:
            _cached_values[param] = res
        return res
    return _cached_values[param]
