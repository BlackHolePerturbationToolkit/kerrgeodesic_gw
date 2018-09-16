r"""
Gravitational radiation by a blob of matter on a circular orbit around a Kerr
black hole


This module implements the following functions:

- :func:`h_blob`: evaluates `r h_+` or `r h_\times`
- :func:`h_blob_signal`: time sequence of `r h_+/\mu` or `r h_\times/\mu`
- :func:`surface_density_toy_model`: `\Sigma(\bar{r},\bar{\phi})` as an
  indicator function
- :func:`surface_density_gaussian`: `\Sigma(\bar{r},\bar{\phi})` as a Gaussian
  profile
- :func:`blob_mass`: mass of the blob of matter, by integration of
  `\Sigma(\bar{r},\bar{\phi})`

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

def h_blob(u, theta, phi, a, surf_dens, param_surf_dens, integ_range,
           mode='+', l_max=10, m_min=1, epsabs=1e-6, epsrel=1e-6):
    r"""
    Return the rescaled value of `h_+` or `h_\times` (depending on the
    parameter ``mode``) for the gravitational radiation emitted by
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
    - ``mode`` -- (default: ``'+'``) string determining which GW polarization
      mode is considered; allowed values are ``'+'`` and ``'x'``, for
      respectively `h_+` and `h_\times`
    - ``l_max`` -- (default: 10) upper bound in the summation over the harmonic
      degree `\ell`
    - ``m_min`` -- (default: 1) lower bound in the summation over the Fourier
      mode `m`
    - ``epsabs`` -- (default: 1e-6) absolute tolerance passed directly to the
      inner 1-D quadrature integration
    - ``epsrel`` -- (default: 1e-6) relative tolerance of the inner 1-D
      integrals

    OUTPUT:

    - a pair ``(rh, err)``, where ``rh`` is the rescaled waveform `r h_+`
      (resp. `r h_\times`) if ``mode`` = ``'+'`` (resp. ``'x'``) , `r` being
      the Boyer-Lindquist radial coordinate of the observer, and ``err`` is an
      estimate of the absolute error in the computation of the integral

    EXAMPLES:

    Gravitational emission `h_+` from a constant density blob close to the ISCO
    of a Schwarzschild black hole::

        sage: from kerrgeodesic_gw import h_blob, surface_density_toy_model
        sage: a = 0
        sage: param_surf_dens = [6.5, 0, 0.6, 0.1]
        sage: integ_range = [6.3, 6.7, -0.04, 0.04]
        sage: h_blob(0., pi/2, 0., a, surface_density_toy_model,  # tol 1.0e-13
        ....:        param_surf_dens, integ_range)
        (0.03688373245628765, 9.872900530109585e-10)
        sage: h_blob(0., pi/2, 0., a, surface_density_toy_model,  # tol 1.0e-13
        ....:        param_surf_dens, integ_range, l_max=5)
        (0.037532424224875585, 1.3788099591183387e-09)

    The `h_\times` part at `\theta=\pi/4`::

        sage: h_blob(0., pi/4, 0., a, surface_density_toy_model,  # tol 1.0e-13
        ....:        param_surf_dens, integ_range, mode='x')
        (0.06203815455135455, 1.972915343495317e-09)
        sage: h_blob(0., pi/4, 0., a, surface_density_toy_model,  # tol 1.0e-13
        ....:        param_surf_dens, integ_range, mode='x', l_max=5)
        (0.06121422594295032, 1.924590678064715e-09)

    """
    # The integrand:
    if mode == '+':
        h_particle = h_plus_particle
    elif mode == 'x':
        h_particle = h_cross_particle
    else:
        raise ValueError("mode must be either '+' or 'x'")
    def ff(phib, rb, u, theta, phi, a, param_surf_dens, l_max, m_min):
        a2 = a*a
        sqrt_gam = sqrt(rb*(rb**3 + a2*rb + 2*a2) / (rb**2 - 2*rb + a2))
        return float(h_particle(a, rb, u, theta, phi, phi0=phib, l_max=l_max,
                                m_min=m_min)) * \
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
    return dblquad(ff, rb_min, rb_max, lambda x: phib_min, lambda x: phib_max,
                   args=(u, theta, phi, a, param_surf_dens, l_max, m_min),
                   epsabs=epsabs, epsrel=epsrel)

def h_blob_signal(u_min, u_max, theta, phi, a, surf_dens, param_surf_dens,
                  integ_range, mode='+', nb_points=100, l_max=10, m_min=1,
                  epsabs=1e-6, epsrel=1e-6, store=None, verbose=True):
    r"""
    Return a time sequence of `h_+` or `h_\times` (depending on the parameter
    ``mode``) describing the gravitational radiation from a matter
    blob orbiting a Kerr black hole.

    INPUT:

    - ``u_min`` -- lower bound of the retarded time coordinate of the observer
      (in units of `M`, the BH mass): `u = t - r_*`, where `t` is the
      Boyer-Lindquist time coordinate and `r_*` is the tortoise coordinate
    - ``u_max`` -- upper bound of the retarded time coordinate of the observer
      (in units of `M`)
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
    - ``mode`` -- (default: ``'+'``) string determining which GW polarization
      mode is considered; allowed values are ``'+'`` and ``'x'``, for
      respectively `h_+` and `h_\times`
    - ``nb_points`` -- (default: 100) number of points in the interval
      ``(u_min, u_max)``
    - ``l_max`` -- (default: 10) upper bound in the summation over the harmonic
      degree `\ell`
    - ``m_min`` -- (default: 1) lower bound in the summation over the Fourier
      mode `m`
    - ``epsabs`` -- (default: 1e-6) absolute tolerance passed directly to the
      inner 1-D quadrature integration
    - ``epsrel`` -- (default: 1e-6) relative tolerance of the inner 1-D
      integrals
    - ``store`` -- (default: ``None``) string containing a file name for
      storing the time sequence; if ``None``, no storage is attempted
    - ``verbose`` -- (default: ``True``) boolean determining whether to monitor
      the progress of the sequence computation

    OUTPUT:

    - a list of ``nb_points`` pairs `(u, r h_+/\mu)` or  `(u, r h_\times/\mu)`
      (depending on ``mode``),  where `\mu` is the blob's mass and
      `r` is the Boyer-Lindquist radial coordinate of the observer

    """
    mu_mass = blob_mass(a, surf_dens, param_surf_dens, integ_range,
                        epsabs=epsabs, epsrel=epsrel)[0]
    signal = []
    du = (u_max - u_min)/(nb_points-1)
    for i in range(nb_points):
        u = u_min + du*i
        h, err = h_blob(u, theta, phi, a, surf_dens, param_surf_dens,
                        integ_range, mode=mode, l_max=l_max,
                        m_min=m_min, epsabs=epsabs, epsrel=epsrel)
        h = h / mu_mass
        signal.append((u, h))
        if verbose:
            #!# print("i={}  u={}  h={}  error={}".format(i, u, h, err), end="\r")
            print("i={}  u={}  h={}  error={}".format(i, u, h, err))
    if verbose:
        print("")
    if store:
        with open(store, "w") as output_file:
            for u, h in signal:
                output_file.write("{}\t{}\n".format(u, h))
    return signal


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
      - ``param[4]`` (optional): amplitude `\Sigma_0`; if not provided,
        then `\Sigma_0=1` is used

    OUTPUT:

    - surface density `\Sigma(\bar{r}, \bar{\phi})`

    EXAMPLES:

    Use with the default amplitude (`\Sigma_0=1`)::

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

    Use with a non-default amplitude (`\Sigma_0=2`)::

        sage: sigma0 = 2.
        sage: param = [6.5, 0, 0.6, 0.1, sigma0]
        sage: surface_density_toy_model(6.5, 0, param)
        2.0

    """
    r0, phi0 = param[0], param[1]
    Dr, Dphi = param[2], param[3]
    Sigma0 = float(param[4]) if len(param) == 5 else float(1)
    if abs(r-r0)<Dr/float(2) and abs(phi-phi0)<Dphi/float(2):
        return Sigma0
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
      - ``param[3]`` (optional): amplitude `\Sigma_0`; if not provided,
        then `\Sigma_0=1` is used

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

    Use with a non-default amplitude (`\Sigma_0=10^{-5}`)::

        sage: sigma0 = 1.e-5
        sage: param = [6.5, 0., 0.3, sigma0]
        sage: surface_density_gaussian(6.5, 0, param)
        1e-05

    """
    r0, phi0, lam = param[0], param[1], param[2]
    Sigma0 = param[3] if len(param) == 4 else float(1)
    return float(Sigma0*exp(-((r - r0*cos(phi-phi0))**2 +
                              (r0*sin(phi-phi0))**2)/lam**2))

def blob_mass(a, surf_dens, param_surf_dens, integ_range, epsabs=1e-8,
              epsrel=1e-8):
    r"""
    Compute the mass of a blob of matter by integrating its surface density.

    INPUT:

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
    - ``epsabs`` -- (default: 1e-8) absolute tolerance of the inner 1-D
      quadrature integration
    - ``epsrel`` -- (default: 1e-8) relative tolerance of the inner 1-D
      integrals

    OUTPUT:

    - a pair ``(mu, err)``, where ``mu`` is the mass of the matter blob and
      ``err`` is an estimate of the absolute error in the computation of the
      integral

    EXAMPLES::

        sage: from kerrgeodesic_gw import blob_mass, surface_density_gaussian
        sage: param = [6.5, 0., 0.3]
        sage: integ_range = [6, 7, -0.1, 0.1]
        sage: blob_mass(0., surface_density_gaussian, param, integ_range)  # tol 1.0e-13
        (0.3328779622200767, 5.671806593829388e-10)
        sage: blob_mass(0., surface_density_gaussian, param, integ_range,  # tol 1.0e-13
        ....:           epsabs=1.e-3)
        (0.33287796222007715, 1.109366597806814e-06)

    """
    def func_mass(phib, rb, a, param_surf_dens):
        r"""
        Integrand of the double integral for the blob mass `\mu`
        """
        a2 = a*a
        sqrt_gam = sqrt(rb*(rb**3 + a2*rb + 2*a2) / (rb**2 - 2*rb + a2))
        return surf_dens(rb, phib, param_surf_dens) * sqrt_gam
    #
    rb_min = integ_range[0]
    rb_max = integ_range[1]
    phib_min = integ_range[2]
    phib_max = integ_range[3]
    return dblquad(func_mass, rb_min, rb_max, lambda x: phib_min,
                   lambda x: phib_max, args=(a, param_surf_dens),
                   epsabs=epsabs, epsrel=epsrel)
