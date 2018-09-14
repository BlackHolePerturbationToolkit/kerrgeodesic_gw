r"""
Gravitational radiation by a blob of matter on a circular orbit around a Kerr
black hole


This module implements the following functions:

- :func:`h_plus_blob`: evaluates `r h_+`
- :func:`h_cross_blob`: evaluates `r h_\times`
- :func:`surface_density_toy_model`: `\Sigma(\bar{r},\bar{\phi})` as an
  indicator function
- :func:`surface_density_gaussian`: `\Sigma(\bar{r},\bar{\phi})` as a Gaussian
  profile

"""
#******************************************************************************
#       Copyright (C) 2018 Eric Gourgoulhon <eric.gourgoulhon@obspm.fr>
#
#  Distributed under the terms of the GNU General Public License (GPL)
#  as published by the Free Software Foundation; either version 2 of
#  the License, or (at your option) any later version.
#                  http://www.gnu.org/licenses/
#******************************************************************************
from sage.functions.trig import cos, sin
from sage.functions.log import exp
from sage.functions.other import sqrt
from scipy.integrate import dblquad
from .gw_particle import h_plus_particle, h_cross_particle

def h_plus_blob(u, theta, phi, a, surf_dens, param_surf_dens, integ_range,
                l_max=10, m_min=1, epsabs=1e-6, epsrel=1e-6):
    r"""
    Return the rescaled `h_+`-part of the gravitational radiation emitted by
    a matter blob in circular orbit around a Kerr black hole.

    INPUT:

    - ``u`` -- retarded time coordinate of the observer (in units of `M`, the
      BH mass): `u = t - r_*`, where `t` is the Boyer-Lindquist time coordinate
      and `r_*` is the tortoise coordinate
    - ``theta`` -- Boyer-Lindquist colatitute  `\theta` of the observer
    - ``phi`` -- Boyer-Lindquist azimuthal coordinate `\phi`  of the observer
    - ``a`` -- BH angular momentum parameter (in units of `M`)
    - ``surf_dens`` -- surface density function `\Sigma`; must take three
      arguments: ``(r_bar, phi_bar, param_surf_dens)``, where

      - ``r_bar`` is the Boyer-Lindquist radial coordinate `\bar{r}` in the
        matter blob
      - ``phi_bar`` is the Boyer-Lindquist azimuthal coordinate `\bar{\phi}` in
        the matter blob
      - ``param_surf_dens`` are parameters defining the function
        `\Sigma(\bar{r},\bar{\phi})`

    - ``param_surf_dens`` -- parameters to be passed as the third argument to
      the function ``surf_dens`` (see above)
    - ``integ_range`` -- tuple `(\bar{r}_1, \bar{r}_2, \bar{\phi}_1, \bar{\phi}_2)`
      defining the integration range
    - ``l_max`` -- (default: 10) upper bound in the summation over the harmonic
      degree `\ell`
    - ``m_min`` -- (default: 1) lower bound in the summation over the Fourier
      mode `m`
    - ``epsabs`` -- (default: 1e-6) absolute tolerance passed directly to the
      inner 1-D quadrature integration
    - ``epsrel`` -- (default: 1e-6) relative tolerance of the inner 1-D
      integrals

    OUTPUT:

    - a pair ``(rh, err)``, where ``rh`` is the rescaled waveform `r h_+`,
      `r` being the Boyer-Lindquist radial coordinate of the observer, and
      ``err`` is an estimate of the absolute error in the computation of the
      integral

    EXAMPLES:

    Gravitational emission from a constant density blob close to the ISCO of
    a Schwarzschild black hole::

        sage: from kerrgeodesic_gw import h_plus_blob, surface_density_toy_model
        sage: a = 0
        sage: param_surf_dens = [6.5, 0, 0.6, 0.1]
        sage: integ_range = [6.3, 6.7, -0.04, 0.04]
        sage: h_plus_blob(0., pi/2, 0., a, surface_density_toy_model,  # tol 1.0e-13
        ....:             param_surf_dens, integ_range)
        (0.03688373245628765, 9.872900530109585e-10)
        sage: h_plus_blob(0., pi/2, 0., a, surface_density_toy_model,  # tol 1.0e-13
        ....:             param_surf_dens, integ_range, l_max=5)
        (0.037532424224875585, 1.3788099591183387e-09)

    """
    # The integrand:
    def f_plus(phib, rb, u, theta, phi, a, param_surf_dens, l_max, m_min):
        a2 = a*a
        sqrt_gam = sqrt(rb*(rb**3 + a2*rb + 2*a2) / (rb**2 - 2*rb + a2))
        return float(h_plus_particle(a, rb, u, theta, phi, phi0=phib,
                                     l_max=l_max, m_min=m_min)) * \
               surf_dens(rb, phib, param_surf_dens) * sqrt_gam
    #
    rb_min = integ_range[0]
    rb_max = integ_range[1]
    phib_min = integ_range[2]
    phib_max = integ_range[3]
    u = float(u)
    theta = float(theta)
    phi = float(phi)
    a = float(a)
    return dblquad(f_plus, rb_min, rb_max, lambda x: phib_min,
                   lambda x: phib_max,
                   args=(u, theta, phi, a, param_surf_dens, l_max, m_min),
                   epsabs=epsabs, epsrel=epsrel)

def h_cross_blob(u, theta, phi, a, surf_dens, param_surf_dens, integ_range,
                 l_max=10, m_min=1, epsabs=1e-6, epsrel=1e-6):
    r"""
    Return the rescaled `h_\times`-part of the gravitational radiation emitted
    by a matter blob in circular orbit around a Kerr black hole.

    INPUT:

    - ``u`` -- retarded time coordinate of the observer (in units of `M`, the
      BH mass): `u = t - r_*`, where `t` is the Boyer-Lindquist time coordinate
      and `r_*` is the tortoise coordinate
    - ``theta`` -- Boyer-Lindquist colatitute  `\theta` of the observer
    - ``phi`` -- Boyer-Lindquist azimuthal coordinate `\phi`  of the observer
    - ``a`` -- BH angular momentum parameter (in units of `M`)
    - ``surf_dens`` -- surface density function `\Sigma`; must take three
      arguments: ``(r_bar, phi_bar, param_surf_dens)``, where

      - ``r_bar`` is the Boyer-Lindquist radial coordinate `\bar{r}` in the
        matter blob
      - ``phi_bar`` is the Boyer-Lindquist azimuthal coordinate `\bar{\phi}` in
        the matter blob
      - ``param_surf_dens`` are parameters defining the function
        `\Sigma(\bar{r},\bar{\phi})`

    - ``param_surf_dens`` -- parameters to be passed as the third argument to
      the function ``surf_dens`` (see above)
    - ``integ_range`` -- tuple `(\bar{r}_1, \bar{r}_2, \bar{\phi}_1, \bar{\phi}_2)`
      defining the integration range
    - ``l_max`` -- (default: 10) upper bound in the summation over the harmonic
      degree `\ell`
    - ``m_min`` -- (default: 1) lower bound in the summation over the Fourier
      mode `m`
    - ``epsabs`` -- (default: 1e-6) absolute tolerance passed directly to the
      inner 1-D quadrature integration
    - ``epsrel`` -- (default: 1e-6) relative tolerance of the inner 1-D
      integrals

    OUTPUT:

    - a pair ``(rh, err)``, where ``rh`` is the rescaled waveform `r h_\times`,
      `r` being the Boyer-Lindquist radial coordinate of the observer, and
      ``err`` is an estimate of the absolute error in the computation of the
      integral

    EXAMPLES:

    Gravitational emission from a constant density blob close to the ISCO of
    a Schwarzschild black hole::

        sage: from kerrgeodesic_gw import h_cross_blob, surface_density_toy_model
        sage: a = 0
        sage: param_surf_dens = [6.5, 0, 0.6, 0.1]
        sage: integ_range = [6.3, 6.7, -0.04, 0.04]
        sage: h_cross_blob(0., pi/4, 0., a, surface_density_toy_model,  # tol 1.0e-13
        ....:             param_surf_dens, integ_range)
        (0.06203815455135455, 1.972915343495317e-09)
        sage: h_cross_blob(0., pi/4, 0., a, surface_density_toy_model,  # tol 1.0e-13
        ....:             param_surf_dens, integ_range, l_max=5)
        (0.06121422594295032, 1.924590678064715e-09)

    """
    # The integrand:
    def f_cross(phib, rb, u, theta, phi, a, param_surf_dens, l_max, m_min):
        a2 = a*a
        sqrt_gam = sqrt(rb*(rb**3 + a2*rb + 2*a2) / (rb**2 - 2*rb + a2))
        return float(h_cross_particle(a, rb, u, theta, phi, phi0=phib,
                                      l_max=l_max, m_min=m_min)) * \
               surf_dens(rb, phib, param_surf_dens) * sqrt_gam
    #
    rb_min = integ_range[0]
    rb_max = integ_range[1]
    phib_min = integ_range[2]
    phib_max = integ_range[3]
    u = float(u)
    theta = float(theta)
    phi = float(phi)
    a = float(a)
    return dblquad(f_cross, rb_min, rb_max, lambda x: phib_min,
                   lambda x: phib_max,
                   args=(u, theta, phi, a, param_surf_dens, l_max, m_min),
                   epsabs=epsabs, epsrel=epsrel)

def surface_density_toy_model(r, phi, param):
    r"""

    Surface density of the toy model matter blob.

    INPUT:

    - ``r`` -- Boyer-Lindquist radial coordinate `\bar{r}` in the matter blob
    - ``phi`` -- Boyer-Lindquist azimuthal coordinate `\bar{\phi}` in the
      matter blob
    - ``param`` -- list of parameters defining the position and extent of the
      matter blob:

      - ``param[0]``: mean radius `r_0` (Boyer-Lindquist coordinate)
      - ``param[1]``: mean azimuthal angle `\phi_0` (Boyer-Lindquist
        coordinate)
      - ``param[2]``: radial extent `\lambda`
      - ``param[3]``: opening angle `\Delta\phi`

    OUTPUT:

    - surface density `\Sigma(\bar{r}, \bar{\phi})`

    EXAMPLES::

        sage: from kerrgeodesic_gw import surface_density_toy_model
        sage: param = [6.5, 0, 0.6, 0.1]
        sage: surface_density_toy_model(6.5, 0, param)
        1.0
        sage: surface_density_toy_model(6.7, -0.04, param)
        1.0
        sage: surface_density_toy_model(6.3, 0.04, param)
        1.0
        sage: surface_density_toy_model(7, -0.06, param)
        0.0
        sage: surface_density_toy_model(5., 0.06, param)
        0.0

    3D representation: `z=\Sigma(\bar{r}, \bar{\phi})` in terms of
    `x:=\bar{r}\cos\bar\phi` and `y:=\bar{r}\sin\bar\phi`::

        sage: s_plot = lambda r, phi: surface_density_toy_model(r, phi, param)
        sage: r, phi, z = var('r phi z')
        sage: plot3d(s_plot, (r, 6, 8), (phi, -0.4, 0.4),
        ....:        transformation=(r*cos(phi), r*sin(phi), z))
        Graphics3d Object

    .. PLOT::

        from kerrgeodesic_gw import surface_density_toy_model
        param = [6.5, 0, 0.6, 0.1]
        s_plot = lambda r, phi: surface_density_toy_model(r, phi, param)
        r, phi, z = var('r phi z')
        g = plot3d(s_plot, (r, 6, 8), (phi, -0.4, 0.4), \
                   transformation=(r*cos(phi), r*sin(phi), z))
        sphinx_plot(g)

    """
    r0, phi0 = param[0], param[1]
    Dr, Dphi = param[2], param[3]
    if abs(r-r0)<Dr/float(2) and abs(phi-phi0)<Dphi/float(2):
        return float(1)
    return float(0)

def surface_density_gaussian(r, phi, param):
    r"""

    Surface density of a matter blob with a Gaussian profile

    INPUT:

    - ``r`` -- Boyer-Lindquist radial coordinate `\bar{r}` in the matter blob
    - ``phi`` -- Boyer-Lindquist azimuthal coordinate `\bar{\phi}` in the
      matter blob
    - ``param`` -- list of parameters defining the position and width of
      matter blob:

      - ``param[0]``: mean radius `r_0` (Boyer-Lindquist coordinate)
      - ``param[1]``: mean azimuthal angle `\phi_0` (Boyer-Lindquist
        coordinate)
      - ``param[2]``: width `\lambda` of the Gaussian profile

    OUTPUT:

    - surface density `\Sigma(\bar{r}, \bar{\phi})`

    EXAMPLES::

        sage: from kerrgeodesic_gw import surface_density_gaussian
        sage: param = [6.5, 0., 0.3]
        sage: surface_density_gaussian(6.5, 0, param)
        1.0
        sage: surface_density_gaussian(8., 0, param)  # tol 1.0e-13
        1.3887943864964021e-11
        sage: surface_density_gaussian(6.5, pi/16, param)  # tol 1.0e-13
        1.4901161193847656e-08

    3D representation: `z=\Sigma(\bar{r}, \bar{\phi})` in terms of
    `x:=\bar{r}\cos\bar\phi` and `y:=\bar{r}\sin\bar\phi`::

        sage: s_plot = lambda r, phi: surface_density_gaussian(r, phi, param)
        sage: r, phi, z = var('r phi z')
        sage: plot3d(s_plot, (r, 6, 8), (phi, -0.4, 0.4),
        ....:        transformation=(r*cos(phi), r*sin(phi), z))
        Graphics3d Object

    .. PLOT::

        from kerrgeodesic_gw import surface_density_gaussian
        param = param = [6.5, 0., 0.3]
        s_plot = lambda r, phi: surface_density_gaussian(r, phi, param)
        r, phi, z = var('r phi z')
        g = plot3d(s_plot, (r, 6, 8), (phi, -0.4, 0.4), \
                   transformation=(r*cos(phi), r*sin(phi), z))
        sphinx_plot(g)

    """
    r0, phi0, lam = param[0], param[1], param[2]
    return float(exp(-((r - r0*cos(phi-phi0))**2 +
                       (r0*sin(phi-phi0))**2)/lam**2))
