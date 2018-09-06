r"""
Gravitational radiation by a particle on a circular orbit around a Kerr
black hole

The gravitational wave emitted by a particle of mass `\mu` in a circular orbit
around a Kerr black hole of mass `M` and angular momentum parameter `a`
is given by the formula:

.. MATH::

    h_+ - i h_\times = \frac{2\mu}{r} \,
    \sum_{\ell=2}^{\infty} \sum_{m=-\ell}^\ell
    \frac{Z^\infty_{\ell m}(r_0)}{(m\omega_0)^2}
    \, _{-2}S^{am\omega_0}_{\ell m}(\theta,\phi)
    \, e^{-i m \phi_0} e^{- i m \omega_0 (t-r_*)}

where

- `h_+ = h_+(t,r,\theta,\phi)` and `h_\times = h_\times(t,r,\theta,\phi)`,
  `(t,r,\theta,\phi)` being the Boyer-Lindquist coordinates of the observer
- `r_*` is the tortoise coordinate corresponding to `r`
- `r_0` is the Boyer-Lindquist radius of the particle's orbit
- `\phi_0` is some constant phase factor
- `\omega_0` is the orbital angular velocity
- `Z^\infty_{\ell m}(r_0)` is numerically provided by the function
  :func:`~kerrgeodesic_gw.zinf.Zinf`
- `_{-2}S^{am\omega_0}_{\ell m}(\theta,\phi)` is the spin-weighted spheroidal
  harmonic of weight `-2` (cf.
  :func:`~kerrgeodesic_gw.spin_weighted_spheroidal_harm.spin_weighted_spheroidal_harmonic`)

The Fourier-series expansion of the waveform `(h_+,h_\times)` received at the
location `(t,r,\theta,\phi)` is

.. MATH::

    h_{+,\times}(t, r, \theta, \phi) = \sum_{m=1}^{+\infty}
    \left[ A_m^{+,\times}(r,\theta) \cos(m\psi)
         + B_m^{+,\times}(r,\theta)\sin(m\psi) \right] ,

where

.. MATH::

    \psi := \omega_0 (t - r_*) - \phi + \phi_0 ,

`\omega_0` being the orbital frequency of the particle and `r_*` the tortoise
coordinate corresponding to `r`.

Note that the dependence of the Fourier coefficients `A_m^{+,\times}(r,\theta)`
and `B_m^{+,\times}(r,\theta)` with respect to `r` is simply `\mu/r`, where
`\mu` is the particle's mass. The dependence with respect to `\theta` is more
complicated and involves both the radius `r_0` of the particle's orbit and the
BH parameters `(M,a)`.

The functions :func:`h_plus_particle` and :func:`h_cross_particle`
defined below compute the rescaled waveforms

.. MATH::

    {\bar h}_+ := \frac{r}{\mu} \, h_+
    \quad\mbox{and}\quad
    {\bar h}_\times := \frac{r}{\mu} \, h_\times

while the functions :func:`h_fourier_mode_plus` and
:func:`h_fourier_mode_cross` compute the rescaled Fourier coefficients

.. MATH::

    {\bar A}_m^{+,\times}(\theta) := \frac{r}{\mu} A_m^{+,\times}(r,\theta)
    \quad\mbox{and}\quad
    {\bar B}_m^{+,\times}(\theta) := \frac{r}{\mu} B_m^{+,\times}(r,\theta)

REFERENCES:

- \S. A. Teukolsky, Astrophys. J. **185**, 635 (1973)
- \S. Detweiler, Astrophys. J. **225**, 687 (1978)
- \M. Shibata, Phys. Rev. D **50**, 6297 (1994)
- \D. Kennefick, Phys. Rev. D **58**, 064012 (1998)
- \S. A. Hughes, Phys. Rev. D **61**, 084004 (2000) [:doi:`10.1103/PhysRevD.61.084004`]
- our paper

"""
#******************************************************************************
#       Copyright (C) 2018 Eric Gourgoulhon <eric.gourgoulhon@obspm.fr>
#
#  Distributed under the terms of the GNU General Public License (GPL)
#  as published by the Free Software Foundation; either version 2 of
#  the License, or (at your option) any later version.
#                  http://www.gnu.org/licenses/
#******************************************************************************

from sage.rings.real_double import RDF
from sage.functions.trig import cos, sin
from .spin_weighted_spherical_harm import spin_weighted_spherical_harmonic
from .spin_weighted_spheroidal_harm import spin_weighted_spheroidal_harmonic
from .zinf import Zinf

def h_fourier_mode_plus(m, a, r0, theta, l_max=10, algorithm_Zinf='spline'):
    r"""
    Return the Fourier mode of a given order ``m`` of the rescaled `h_+`-part
    of the gravitational wave emitted by a particle in circular orbit around
    a Kerr black hole.

    The rescaled Fourier mode of order ``m`` received at the location
    `(t,r,\theta,\phi)` is

    .. MATH::

        \frac{r}{\mu} h_m^+ = {\bar A}_m^+ \cos(m\psi)
                                + {\bar B}_m^+ \sin(m\psi)

    where `\mu` is the particle mass and `\psi := \omega_0 (t-r_*) - \phi`,
    `\omega_0` being the orbital frequency of the particle and `r_*` the
    tortoise coordinate corresponding to `r`.

    INPUT:

    - ``m`` -- positive integer defining the Fourier mode
    - ``a`` -- BH angular momentum parameter (in units of `M`, the BH mass)
    - ``r0`` -- Boyer-Lindquist radius of the orbit (in units of `M`)
    - ``theta`` -- Boyer-Lindquist colatitute `\theta` of the observer
    - ``l_max`` -- (default: 5) upper bound in the summation over the harmonic
      degree `\ell`
    - ``algorithm_Zinf`` -- (default: ``'spline'``) string describing the
      computational method for `Z^\infty_{\ell m}(r_0)`; allowed values are

      - ``'spline'``: spline interpolation of tabulated data
      - ``'1.5PN'`` (only for ``a=0``): 1.5-post-Newtonian expansion following
        E. Poisson, Phys. Rev. D **47**, 1497 (1993)
        [:doi:`10.1103/PhysRevD.47.1497`]

    OUTPUT:

    - tuple `({\bar A}_m^+, {\bar B}_m^+)` (cf. the above expression
      for `(r/\mu) h_m^+`)

    EXAMPLES:

    Let us consider the case `a=0` first (Schwarzschild black hole), with
    `m=2`::

        sage: from kerrgeodesic_gw import h_fourier_mode_plus
        sage: a = 0
        sage: h_fourier_mode_plus(2, a, 8., pi/2)  # tol 1.0e-13
        (0.2014580652208302, -0.06049343736886148)
        sage: h_fourier_mode_plus(2, a, 8., pi/2, l_max=5)  # tol 1.0e-13
        (0.20146097329552273, -0.060495372034569186)
        sage: h_fourier_mode_plus(2, a, 8., pi/2, l_max=5, algorithm_Zinf='1.5PN')  # tol 1.0e-13
        (0.2204617125753912, -0.0830484439639611)

    Values of `m` different from 2::

        sage: h_fourier_mode_plus(3, a, 20., pi/2)  # tol 1.0e-13
        (-0.005101595598729037, -0.021302121442654077)
        sage: h_fourier_mode_plus(3, a, 20., pi/2, l_max=5, algorithm_Zinf='1.5PN')  # tol 1.0e-13
        (0.0, -0.016720919174427588)
        sage: h_fourier_mode_plus(1, a, 8., pi/2)  # tol 1.0e-13
        (-0.014348477201223874, -0.05844679575244101)
        sage: h_fourier_mode_plus(0, a, 8., pi/2)
        (0, 0)

    The case `a/M=0.95` (rapidly rotating Kerr black hole)::

        sage: a = 0.95
        sage: h_fourier_mode_plus(2, a, 8., pi/2)  # tol 1.0e-13
        (0.182748773431646, -0.05615306925896938)
        sage: h_fourier_mode_plus(8, a, 8., pi/2)  # tol 1.0e-13
        (-4.724709221209198e-05, 0.0006867183495228116)
        sage: h_fourier_mode_plus(2, a, 2., pi/2)  # tol 1.0e-13
        (0.1700402877617014, 0.33693580916655747)
        sage: h_fourier_mode_plus(8, a, 2., pi/2)  # tol 1.0e-13
        (-0.009367442995129153, -0.03555092085651877)

    """
    if m == 0:
        return (0, 0)
    a = RDF(a)  # RDF = Real Double Field
    # m times the orbital angular velocity
    m_omega0 = RDF(m / (r0**1.5 + a))
    am_omega0 = a*m_omega0
    # Sum over l
    Aplus, Bplus = 0, 0
    for l in range(max(m, 2), l_max+1):
        # NB: we call spin_weighted_spher*_harmonic with phi=0, so that the
        # outcome (an element of CDF) should be real; to avoid any round-off
        # error, we call the method real() on the outcome, thereby getting an
        # element of RDF instead of a real in CDF.
        if a == 0:
            Slm = spin_weighted_spherical_harmonic(-2, l, m, theta, 0,
                                                   numerical=RDF).real()
            Slmm = spin_weighted_spherical_harmonic(-2, l, -m, theta, 0,
                                                    numerical=RDF).real()
        else:
            Slm = spin_weighted_spheroidal_harmonic(-2, l, m, am_omega0, theta,
                                                    0, cached=True).real()
            Slmm = spin_weighted_spheroidal_harmonic(-2, l, -m, -am_omega0,
                                                     theta, 0, cached=True).real()
        Zlm = Zinf(a, l, m, r0, algorithm=algorithm_Zinf)
        reZlm = Zlm.real()
        imZlm = Zlm.imag()
        Splus = Slm + (-1)**l *Slmm
        Aplus += reZlm*Splus
        Bplus += imZlm*Splus
    pre = RDF(2)/(m_omega0*m_omega0)
    Aplus *= pre
    Bplus *= pre
    return (Aplus, Bplus)

def h_fourier_mode_cross(m, a, r0, theta, l_max=10, algorithm_Zinf='spline'):
    r"""
    Return the Fourier mode of a given order ``m`` of the rescaled
    `h_\times`-part of the gravitational wave emitted by a particle in
    circular orbit around a Kerr black hole.

    The rescaled Fourier mode of order ``m`` received at the location
    `(t,r,\theta,\phi)` is

    .. MATH::

        \frac{r}{\mu} h_m^\times = {\bar A}_m^\times \cos(m\psi)
                                + {\bar B}_m^\times \sin(m\psi)

    where `\mu` is the particle mass and `\psi := \omega_0 (t-r_*) - \phi`,
    `\omega_0` being the orbital frequency of the particle and `r_*` the
    tortoise coordinate corresponding to `r`.

    INPUT:

    - ``m`` -- positive integer defining the Fourier mode
    - ``a`` -- BH angular momentum parameter (in units of `M`, the BH mass)
    - ``r0`` -- Boyer-Lindquist radius of the orbit (in units of `M`)
    - ``theta`` -- Boyer-Lindquist colatitute `\theta` of the observer
    - ``l_max`` -- (default: 5) upper bound in the summation over the harmonic
      degree `\ell`
    - ``algorithm_Zinf`` -- (default: ``'spline'``) string describing the
      computational method for `Z^\infty_{\ell m}(r_0)`; allowed values are

      - ``'spline'``: spline interpolation of tabulated data
      - ``'1.5PN'`` (only for ``a=0``): 1.5-post-Newtonian expansion following
        E. Poisson, Phys. Rev. D **47**, 1497 (1993)
        [:doi:`10.1103/PhysRevD.47.1497`]

    OUTPUT:

    - tuple `({\bar A}_m^\times, {\bar B}_m^\times)` (cf. the above expression
      for `(r/\mu) h_m^\times`)

    EXAMPLES:

    Let us consider the case `a=0` first (Schwarzschild black hole), with
    `m=2`::

        sage: from kerrgeodesic_gw import h_fourier_mode_cross
        sage: a = 0

    `h_m^\times` is always zero in the direction `\theta=\pi/2`::

        sage: h_fourier_mode_cross(2, a, 8., pi/2)  # tol 1.0e-13
        (3.444996575846961e-17, 1.118234985040581e-16)

    Let us then evaluate `h_m^\times` in the direction `\theta=\pi/4`::

        sage: h_fourier_mode_cross(2, a, 8., pi/4)  # tol 1.0e-13
        (0.09841144532628172, 0.31201728756415015)
        sage: h_fourier_mode_cross(2, a, 8., pi/4, l_max=5)  # tol 1.0e-13
        (0.09841373523119075, 0.31202073689061305)
        sage: h_fourier_mode_cross(2, a, 8., pi/4, l_max=5, algorithm_Zinf='1.5PN')  # tol 1.0e-13
        (0.11744823578781578, 0.34124272645755677)

    Values of `m` different from 2::

        sage: h_fourier_mode_cross(3, a, 20., pi/4)  # tol 1.0e-13
        (0.022251439699635174, -0.005354134279052387)
        sage: h_fourier_mode_cross(3, a, 20., pi/4, l_max=5, algorithm_Zinf='1.5PN')  # tol 1.0e-13
        (0.017782177999686274, 0.0)
        sage: h_fourier_mode_cross(1, a, 8., pi/4)  # tol 1.0e-13
        (0.03362589948237155, -0.008465651545641889)
        sage: h_fourier_mode_cross(0, a, 8., pi/4)
        (0, 0)

    The case `a/M=0.95` (rapidly rotating Kerr black hole)::

        sage: a = 0.95
        sage: h_fourier_mode_cross(2, a, 8., pi/4)  # tol 1.0e-13
        (0.08843838892991202, 0.28159329265206867)
        sage: h_fourier_mode_cross(8, a, 8., pi/4)  # tol 1.0e-13
        (-0.00014588821622195107, -8.179557811364057e-06)
        sage: h_fourier_mode_cross(2, a, 2., pi/4)  # tol 1.0e-13
        (-0.6021994882885746, 0.3789513303450391)
        sage: h_fourier_mode_cross(8, a, 2., pi/4)  # tol 1.0e-13
        (0.01045760329050054, -0.004986913120370192)

    """
    if m == 0:
        return (0, 0)
    a = RDF(a)  # RDF = Real Double Field
    # m times the orbital angular velocity
    m_omega0 = RDF(m / (r0**1.5 + a))
    am_omega0 = a*m_omega0
    # Sum over l
    Across, Bcross = 0, 0
    for l in range(max(m, 2), l_max+1):
        # NB: we call spin_weighted_spher*_harmonic with phi=0, so that the
        # outcome (an element of CDF) should be real; to avoid any round-off
        # error, we call the method real() on the outcome, thereby getting an
        # element of RDF instead of a real in CDF.
        if a == 0:
            Slm = spin_weighted_spherical_harmonic(-2, l, m, theta, 0,
                                                   numerical=RDF).real()
            Slmm = spin_weighted_spherical_harmonic(-2, l, -m, theta, 0,
                                                    numerical=RDF).real()
        else:
            Slm = spin_weighted_spheroidal_harmonic(-2, l, m, am_omega0, theta,
                                                    0, cached=True).real()
            Slmm = spin_weighted_spheroidal_harmonic(-2, l, -m, -am_omega0,
                                                     theta, 0, cached=True).real()
        Zlm = Zinf(a, l, m, r0, algorithm=algorithm_Zinf)
        reZlm = Zlm.real()
        imZlm = Zlm.imag()
        Scross = Slm - (-1)**l *Slmm
        Across -= imZlm*Scross
        Bcross += reZlm*Scross
    pre = RDF(2)/(m_omega0*m_omega0)
    Across *= pre
    Bcross *= pre
    return (Across, Bcross)

def h_plus_particle(a, r0, phi0, u, theta, phi, l_max=10, m_min=1,
                    algorithm_Zinf='spline'):
    r"""
    Return the rescaled `h_+`-part of the gravitational radiation emitted by
    a particle in circular orbit around a Kerr black hole.

    INPUT:

    - ``a`` -- BH angular momentum parameter (in units of `M`, the BH mass)
    - ``r0`` -- Boyer-Lindquist radius of the orbit (in units of `M`)
    - ``phi0`` -- phase factor
    - ``u`` -- retarded time coordinate of the observer (in units of `M`):
      `u = t - r_*`, where `t` is the Boyer-Lindquist time coordinate and `r_*`
      is the tortoise coordinate
    - ``theta`` -- Boyer-Lindquist colatitute  `\theta` of the observer
    - ``phi`` -- Boyer-Lindquist azimuthal coordinate `\phi`  of the observer
    - ``l_max`` -- (default: 10) upper bound in the summation over the harmonic
      degree `\ell`
    - ``m_min`` -- (default: 1) lower bound in the summation over the Fourier
      mode `m`
    - ``algorithm_Zinf`` -- (default: ``'spline'``) string describing the
      computational method for `Z^\infty_{\ell m}(r_0)`; allowed values are

      - ``'spline'``: spline interpolation of tabulated data
      - ``'1.5PN'`` (only for ``a=0``): 1.5-post-Newtonian expansion following
        E. Poisson, Phys. Rev. D **47**, 1497 (1993)
        [:doi:`10.1103/PhysRevD.47.1497`]

    OUTPUT:

    - the rescaled waveform  `(r / \mu) h_+`, where `\mu` is the particle's
      mass and `r` is the Boyer-Lindquist radial coordinate of the observer

    EXAMPLES:

    Let us consider the case `a=0` (Schwarzschild black hole) and `r_0=6 M`
    (emission from the ISCO)::

        sage: from kerrgeodesic_gw import h_plus_particle
        sage: a = 0
        sage: h_plus_particle(a, 6., 0., 0., pi/2, 0.)  # tol 1.0e-13
        0.1536656546005028
        sage: h_plus_particle(a, 6., 0., 0., pi/2, 0., l_max=5)  # tol 1.0e-13
        0.157759938177291
        sage: h_plus_particle(a, 6., 0., 0., pi/2, 0., l_max=5, algorithm_Zinf='1.5PN')  # tol 1.0e-13
        0.22583887001798497

    For an orbit of larger radius (`r_0=12 M`), the 1.5-post-Newtonian
    approximation is in better agreement with the exact computation::

        sage: h_plus_particle(a, 12., 0., 0., pi/2, 0.)  # tol 1.0e-13
        0.11031251832047866
        sage: h_plus_particle(a, 12., 0., 0., pi/2, 0., l_max=5, algorithm_Zinf='1.5PN')  # tol 1.0e-13
        0.12935832450325302

    A plot of the waveform generated by a particle orbiting at the ISCO::

        sage: hp = lambda u: h_plus_particle(a, 6., 0., u, pi/2, 0.)
        sage: plot(hp, (0, 200.), axes_labels=[r'$(t-r_*)/M$', r'$r h_+/\mu$'],
        ....:      gridlines=True, frame=True, axes=False)
        Graphics object consisting of 1 graphics primitive

    .. PLOT::

        from kerrgeodesic_gw import h_plus_particle
        hp = lambda u: h_plus_particle(0., 6., 0., u, pi/2, 0.)
        g = plot(hp, (0, 200.), axes_labels=[r'$(t-r_*)/M$', r'$r h_+/\mu$'], \
                 gridlines=True, frame=True, axes=False)
        sphinx_plot(g)

    Case `a/M=0.95` (rapidly rotating Kerr black hole)::

        sage: a = 0.95
        sage: h_plus_particle(a, 2., 0., 0., pi/2, 0.)  # tol 1.0e-13
        0.20326150400852214

    Assessing the importance of the mode `m=1`::

        sage: h_plus_particle(a, 2., 0., 0., pi/2, 0., m_min=2)  # tol 1.0e-13
        0.21845811047370495

    """
    # Orbital angular velocity:
    omega0 = RDF(1. / (r0**1.5 + a))
    # Phase angle:
    psi = omega0*u - phi + phi0
    # Sum over the Fourier modes:
    hplus = 0
    for m in range(m_min, l_max+1):
        hm = h_fourier_mode_plus(m, a, r0, theta, l_max=l_max,
                                 algorithm_Zinf=algorithm_Zinf)
        mpsi = m*psi
        hplus += hm[0]*cos(mpsi) + hm[1]*sin(mpsi)
    return hplus

def h_cross_particle(a, r0, phi0, u, theta, phi, l_max=10, m_min=1,
                     algorithm_Zinf='spline'):
    r"""
    Return the rescaled `h_\times`-part of the gravitational radiation emitted
    by a particle in circular orbit around a Kerr black hole.

    INPUT:

    - ``a`` -- BH angular momentum parameter (in units of `M`, the BH mass)
    - ``r0`` -- Boyer-Lindquist radius of the orbit (in units of `M`)
    - ``phi0`` -- phase factor
    - ``u`` -- retarded time coordinate of the observer (in units of `M`):
      `u = t - r_*`, where `t` is the Boyer-Lindquist time coordinate and `r_*`
      is the tortoise coordinate
    - ``theta`` -- Boyer-Lindquist colatitute  `\theta` of the observer
    - ``phi`` -- Boyer-Lindquist azimuthal coordinate `\phi`  of the observer
    - ``l_max`` -- (default: 10) upper bound in the summation over the harmonic
      degree `\ell`
    - ``m_min`` -- (default: 1) lower bound in the summation over the Fourier
      mode `m`
    - ``algorithm_Zinf`` -- (default: ``'spline'``) string describing the
      computational method for `Z^\infty_{\ell m}(r_0)`; allowed values are

      - ``'spline'``: spline interpolation of tabulated data
      - ``'1.5PN'`` (only for ``a=0``): 1.5-post-Newtonian expansion following
        E. Poisson, Phys. Rev. D **47**, 1497 (1993)
        [:doi:`10.1103/PhysRevD.47.1497`]

    OUTPUT:

    - the rescaled waveform  `(r / \mu) h_\times`, where `\mu` is the
      particle's mass and `r` is the Boyer-Lindquist radial coordinate of the
      observer

    EXAMPLES:

    Let us consider the case `a=0` (Schwarzschild black hole) and `r_0=6 M`
    (emission from the ISCO). For `\theta=\pi/2`, we have `h_\times=0`::

        sage: from kerrgeodesic_gw import h_cross_particle
        sage: a = 0
        sage: h_cross_particle(a, 6., 0., 0., pi/2, 0.)  # tol 1.0e-13
        1.0041370414185673e-16

    while for `\theta=\pi/4`, we have::

        sage: h_cross_particle(a, 6., 0., 0., pi/4, 0.)  # tol 1.0e-13
        0.275027796440582
        sage: h_cross_particle(a, 6., 0., 0., pi/4, 0., l_max=5)  # tol 1.0e-13
        0.2706516303570341
        sage: h_cross_particle(a, 6., 0., 0., pi/4, 0., l_max=5, algorithm_Zinf='1.5PN')  # tol 1.0e-13
        0.2625307460899205

    For an orbit of larger radius (`r_0=12 M`), the 1.5-post-Newtonian
    approximation is in better agreement with the exact computation::

        sage: h_cross_particle(a, 12., 0., 0., pi/4, 0.)
        0.1050751824554463
        sage: h_cross_particle(a, 12., 0., 0., pi/4, 0., l_max=5, algorithm_Zinf='1.5PN')  # tol 1.0e-13
        0.10244926162224487

    A plot of the waveform generated by a particle orbiting at the ISCO::

        sage: hc = lambda u: h_cross_particle(a, 6., 0., u, pi/4, 0.)
        sage: plot(hc, (0, 200.), axes_labels=[r'$(t-r_*)/M$', r'$r h_\times/\mu$'],
        ....:      gridlines=True, frame=True, axes=False)
        Graphics object consisting of 1 graphics primitive

    .. PLOT::

        from kerrgeodesic_gw import h_cross_particle
        hc = lambda u: h_cross_particle(0, 6., 0., u, pi/4, 0.)
        g = plot(hc, (0, 200.), axes_labels=[r'$(t-r_*)/M$', r'$r h_\times/\mu$'], \
                 gridlines=True, frame=True, axes=False)
        sphinx_plot(g)

    Case `a/M=0.95` (rapidly rotating Kerr black hole)::

        sage: a = 0.95
        sage: h_cross_particle(a, 2., 0., 0., pi/4, 0.)  # tol 1.0e-13
        -0.2681353673743396

    Assessing the importance of the mode `m=1`::

        sage: h_cross_particle(a, 2., 0., 0., pi/4, 0., m_min=2)  # tol 1.0e-13
        -0.3010579420748449

    """
    # Orbital angular velocity:
    omega0 = RDF(1. / (r0**1.5 + a))
    # Phase angle:
    psi = omega0*u - phi + phi0
    # Sum over the Fourier modes:
    hcross = 0
    for m in range(m_min, l_max+1):
        hm = h_fourier_mode_cross(m, a, r0, theta, l_max=l_max,
                                  algorithm_Zinf=algorithm_Zinf)
        mpsi = m*psi
        hcross += hm[0]*cos(mpsi) + hm[1]*sin(mpsi)
    return hcross

