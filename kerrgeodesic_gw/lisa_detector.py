r"""
Functions relative to the LISA detector.

"""
#******************************************************************************
#       Copyright (C) 2018 Eric Gourgoulhon <eric.gourgoulhon@obspm.fr>
#
#  Distributed under the terms of the GNU General Public License (GPL)
#  as published by the Free Software Foundation; either version 2 of
#  the License, or (at your option) any later version.
#                  http://www.gnu.org/licenses/
#******************************************************************************

import os
from sage.calculus.interpolation import spline
from sage.functions.other import sqrt
from sage.functions.log import log
from sage.functions.trig import cos
from sage.rings.real_double import RDF
from sage.symbolic.constants import pi

_sensitivity_spline = None
_psd_spline = None

def strain_sensitivity(freq):
    r"""
    Return LISA strain spectral sensitivity at a given frequency.

    The strain spectral sensitivity is the square root of the effective
    noise power spectral density (cf. :func:`power_spectral_density`).

    INPUT:

    - ``freq`` -- frequency `f` (in `\mathrm{Hz}`)

    OUTPUT:

    - strain sensitivity `S(f)^{1/2}` (in `\mathrm{Hz}^{-1/2}`)

    EXAMPLES::

        sage: from kerrgeodesic_gw import lisa_detector
        sage: hn = lisa_detector.strain_sensitivity
        sage: hn(1.e-1)  # tol 1.0e-13
        5.82615031500758e-20
        sage: hn(1.e-2)  # tol 1.0e-13
        1.654806317072275e-20
        sage: hn(1.e-3)  # tol 1.0e-13
        1.8082609253700212e-19

    ::

        sage: plot_loglog(hn, (1e-5, 1), plot_points=2000, ymin=1e-20, ymax=1e-14,
        ....:             axes_labels=[r"$f\ [\mathrm{Hz}]$",
        ....:                          r"$S(f)^{1/2} \ \left[\mathrm{Hz}^{-1/2}\right]$"],
        ....:             gridlines='minor', frame=True, axes=False)
        Graphics object consisting of 1 graphics primitive

    .. PLOT::

        from kerrgeodesic_gw import lisa_detector
        hn = lisa_detector.strain_sensitivity
        g = plot_loglog(hn, (1e-5, 1), plot_points=2000, ymin=1e-20, ymax=1e-14, \
                        axes_labels=[r"$f\ [\mathrm{Hz}]$", \
                                     r"$S(f)^{1/2} \ \left[\mathrm{Hz}^{-1/2}\right]$"], \
                        gridlines='minor', frame=True, axes=False)
        sphinx_plot(g)

    """
    global _sensitivity_spline
    if not _sensitivity_spline:
        data = []
        file_name = os.path.join(os.path.dirname(__file__),
                                 "data/Sensitivity_LISA_SciRD1806_Alloc.dat")
        with open(file_name, "r") as data_file:
            for dline in data_file:
                f, s = dline.split('\t')
                data.append((log(RDF(f), 10), log(sqrt(RDF(s)), 10)))
        _sensitivity_spline = spline(data)
    if freq<1.e-5 or freq>1.:
        raise ValueError("frequency {} Hz is out of range".format(freq))
    freq = RDF(freq)
    return RDF(10)**(_sensitivity_spline(log(freq, 10)))

def power_spectral_density(freq):
    r"""
    Return the effective power spectral density (PSD) of the detector noise
    at a given frequency.

    INPUT:

    - ``freq`` -- frequency `f` (in `\mathrm{Hz}`)

    OUTPUT:

    - effective power spectral density `S(f)` (in `\mathrm{Hz}^{-1}`)

    EXAMPLES::

        sage: from kerrgeodesic_gw import lisa_detector
        sage: Sn = lisa_detector.power_spectral_density
        sage: Sn(1.e-1)  # tol 1.0e-13
        3.3944027493062926e-39
        sage: Sn(1.e-2)  # tol 1.0e-13
        2.738383947022306e-40
        sage: Sn(1.e-3)  # tol 1.0e-13
        3.269807574220045e-38

    """
    global _psd_spline
    if not _psd_spline:
        data = []
        file_name = os.path.join(os.path.dirname(__file__),
                                 "data/Sensitivity_LISA_SciRD1806_Alloc.dat")
        with open(file_name, "r") as data_file:
            for dline in data_file:
                f, s = dline.split('\t')
                data.append((log(RDF(f), 10), log(RDF(s), 10)))
        _psd_spline = spline(data)
    if freq<1.e-5 or freq>1.:
        raise ValueError("frequency {} Hz is out of range".format(freq))
    freq = RDF(freq)
    return RDF(10)**(_psd_spline(log(freq, 10)))

def power_spectral_density_RCLfit(freq):
    r"""
    Return the effective power spectral density (PSD) of the detector noise
    at a given frequency, according to the analytical fit by Robson, Cornish
    and Liu, :arxiv:`1803.01944`

    INPUT:

    - ``freq`` -- frequency `f` (in `\mathrm{Hz}`)

    OUTPUT:

    - effective power spectral density `S(f)` (in `\mathrm{Hz}^{-1}`)

    EXAMPLES::

        sage: from kerrgeodesic_gw import lisa_detector
        sage: Sn = lisa_detector.power_spectral_density_RCLfit
        sage: Sn(1.e-1)  # tol 1.0e-13
        2.12858262120861e-39
        sage: Sn(1.e-2)  # tol 1.0e-13
        1.44307343517977e-40
        sage: Sn(1.e-3)  # tol 1.0e-13
        1.63410027259543e-38

    """
    p_oms = 2.25e-22 * (1 + (2.e-3/freq)**4)
    p_acc = 9.e-30 * (1 +(4.e-4/freq)**2) * (1 + (freq/8.e-3)**4)
    L = 2.5e9
    f_star = 1.909e-2
    p_n = (p_oms + 2*(1 + cos(freq/f_star)**2)*p_acc/(2*RDF(pi)*freq)**4)/L**2
    return 10./3.*p_n*(1 + 0.6*(freq/f_star)**2)
