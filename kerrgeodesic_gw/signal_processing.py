r"""
Signal processing

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
from scipy.fftpack import fft, fftfreq, fftshift
from scipy.integrate import quad
from sage.calculus.interpolation import spline
from sage.functions.other import sqrt
from sage.rings.real_double import RDF

def fourier(signal):
    r"""
    Compute the Fourier transform of a time signal.

    The *Fourier transform* of the time signal `h(t)` is defined by

    .. MATH::

        \tilde{h}(f) = \int_{-\infty}^{+\infty} h(t)\,
                     \mathrm{e}^{-2\pi \mathrm{i} f t} \, \mathrm{d}t

    INPUT:

    - ``signal`` -- list of pairs `(t, h(t))`, where `t` is the time and `h(t)`
      is the signal at `t`. *NB*: the sampling in `t` must be uniform

    OUTPUT:

    - a numerical approximation (FFT) of the Fourier transform, as a list
      of pairs `(f,\tilde{h}(f))`

    EXAMPLES:

    Fourier transform of the `h_+` signal from a particle orbiting at the
    ISCO of a Schwarzschild black hole::

        sage: from kerrgeodesic_gw import h_particle_signal, fourier
        sage: a, r0 = 0., 6.
        sage: theta, phi = pi/2, 0
        sage: sig = h_particle_signal(a, r0, theta, phi, 0., 800., nb_points=200)
        sage: sig[:5]  # tol 1.0e-13
        [(0.000000000000000, 0.1536656546005028),
         (4.02010050251256, 0.03882960191740132),
         (8.04020100502512, -0.0791773803745108),
         (12.0603015075377, -0.18368354016995483),
         (16.0804020100502, -0.25796165580196795)]
        sage: ft = fourier(sig)
        sage: ft[:5]   # tol 1.0e-13
        [(-0.12437500000000001, (1.0901773159466113+0j)),
         (-0.12313125000000001, (1.0901679813096925+0.021913447210468399j)),
         (-0.12188750000000001, (1.0901408521337845+0.043865034665617614j)),
         (-0.12064375000000001, (1.0900987733011678+0.065894415355341171j)),
         (-0.11940000000000001, (1.0900473072498438+0.088044567109304195j))]

    Plot of the norm of the Fourier transform::

        sage: ft_norm = [(f, abs(hf)) for (f, hf) in ft]
        sage: line(ft_norm, axes_labels=[r'$f M$', r'$|\tilde{h}_+(f)| r/(\mu M)$'],
        ....:      gridlines=True, frame=True, axes=False)
        Graphics object consisting of 1 graphics primitive

    .. PLOT::

        from kerrgeodesic_gw import h_particle_signal, fourier
        a, r0 = 0., 6.
        theta, phi = pi/2, 0
        sig = h_particle_signal(a, r0, theta, phi, 0., 800., nb_points=200)
        ft = fourier(sig)
        ft_norm = [(f, abs(hf)) for (f, hf) in ft]
        g = line(ft_norm, \
                 axes_labels=[r'$f M$', r'$|\tilde{h}_+(f)| r/(\mu M)$'], \
                 gridlines=True, frame=True, axes=False)
        sphinx_plot(g)

    The first peak is at the orbital frequency of the particle::

        sage: f0 = n(1/(2*pi*r0^(3/2))); f0
        0.0108291222393566

    while the highest peak is at twice this frequency (`m=2` mode).

    """
    t = [s[0] for s in signal]
    h = [s[1] for s in signal]
    dt = t[1] - t[0]
    hf = dt*fftshift(fft(h))
    f = fftshift(fftfreq(len(signal), d=dt))
    return zip(f, hf)

def read_signal(filename, dirname=None):
    r"""
    Read a signal from a data file.

    INPUT:

    - ``filename`` -- string; name of the file
    - ``dirname`` -- (default: None) string; name of directory where the file
      is located

    OUTPUT:

    - signal as a list of pairs `(t, h(t))`

    EXAMPLES::

        sage: from kerrgeodesic_gw import save_signal, read_signal
        sage: sig0 = [(RDF(i/10), RDF(sin(i/5))) for i in range(5)]
        sage: from sage.misc.temporary_file import tmp_filename
        sage: filename = tmp_filename(ext='.dat')
        sage: save_signal(sig0, filename)
        sage: sig = read_signal(filename)
        sage: sig   # tol 1.0e-13
        [(0.0, 0.0),
         (0.1, 0.19866933079506122),
         (0.2, 0.3894183423086505),
         (0.3, 0.5646424733950354),
         (0.4, 0.7173560908995228)]

    A test::

        sage: sig == sig0
        True

    """
    sig = []
    if dirname:
        file_name = os.path.join(dirname, filename)
    else:
        file_name = filename
    with open(file_name, "r") as data_file:
        for dline in data_file:
            t, h = dline.split('\t')
            sig.append((RDF(t), RDF(h)))
    return sig

def save_signal(sig, filename, dirname=None):
    r"""
    Write a signal in a data file.

    INPUT:

    - ``sig`` -- signal as a list of pairs `(t, h(t))`
    - ``filename`` -- string; name of the file
    - ``dirname`` -- (default: None) string; name of directory where the file
      is located

    EXAMPLES::

        sage: from kerrgeodesic_gw import save_signal, read_signal
        sage: sig = [(RDF(i/10), RDF(sin(i/5))) for i in range(5)]
        sage: sig   # tol 1.0e-13
        [(0.0, 0.0),
         (0.1, 0.19866933079506122),
         (0.2, 0.3894183423086505),
         (0.3, 0.5646424733950354),
         (0.4, 0.7173560908995228)]
        sage: from sage.misc.temporary_file import tmp_filename
        sage: filename = tmp_filename(ext='.dat')
        sage: save_signal(sig, filename)

    A test::

        sage: sig1 = read_signal(filename)
        sage: sig1 == sig
        True

    """
    if dirname:
        file_name = os.path.join(dirname, filename)
    else:
        file_name = filename
    with open(file_name, "w") as output_file:
        for t, h in sig:
            output_file.write("{}\t{}\n".format(t, h))

def signal_to_noise(signal, time_scale, psd, fmin, fmax, scale=1,
                    quad_epsrel=1.e-6, quad_limit=500):
    r"""
    Evaluate the signal-to-noise ratio of a signal observed in a
    detector of a given power spectral density.

    The *signal-to-noise ratio* `\rho` of the time signal `h(t)` is
    computed according to the formula

    .. MATH::
       :label: rho_snr

        \rho^2 = 4 \int_{0}^{+\infty} \frac{|\tilde{h}(f)|^2}{S_n(f)}
                    \, \mathrm{d}f

    where `\tilde{h}(f)` is the Fourier transform of `h(t)` (see
    :func:`fourier`) and `S_n(f)` is the detector's one-sided noise power
    spectral density (see e.g. :func:`.lisa_detector.power_spectral_density`).

    INPUT:

    - ``signal`` -- list of pairs `(t, h(t))`, where `t` is the time and `h(t)`
      is the signal at `t`. *NB*: the sampling in `t` must be uniform
    - ``time_scale`` -- value of `t` unit in terms of `S_n(f)` unit; if `S_n(f)`
      is provided in `\mathrm{Hz}^{-1}`, then `time_scale` must be the unit of
      `t` in ``signal`` expressed in seconds.
    - ``psd`` -- function with a single argument (`f`) representing the
      detector's one-sided noise power spectral density `S_n`
    - ``fmin`` -- lower bound used instead of `0` in the integral :eq:`rho_snr`
    - ``fmax`` -- upper bound used instead of `+\infty` in the integral
      :eq:`rho_snr`
    - ``scale`` -- (default: ``1``) scale factor by which `h(t)` must be
      multiplied to get the actual signal
    - ``quad_epsrel`` -- (default: ``1.e-6``) relative error tolerance in the
      computation of the integral :eq:`rho_snr`
    - ``quad_limit`` -- (default: ``500``) upper bound on the number of
      subintervals used in the adaptive algorithm to compute the integral
      (this corresponds to the argument ``limit`` of SciPy's function
      `quad <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html#scipy.integrate.quad/>`_)

    EXAMPLES:

    Let us evaluate the SNR of the gravitational signal generated by a 1-solar
    mass object orbiting at the ISCO of Sgr A* observed by LISA during 1 day.
    We need the following functions::

        sage: from kerrgeodesic_gw import (h_particle_signal, signal_to_noise,
        ....:                              lisa_detector, astro_data)

    We model Sgr A* as a Schwarzschild black hole and we consider that the
    signal is constituted by the mode `h_+(t)` observed in the orbital plane::

        sage: a, r0 = 0., 6.
        sage: theta, phi = pi/2, 0
        sage: tmax = 24*3600/astro_data.MSgrA_s  # 1 day in units of Sgr A* mass
        sage: h = h_particle_signal(a, r0, theta, phi, 0., tmax, mode='+',
        ....:                       nb_points=4000)

    The signal-to-noise ratio is then computed as::

        sage: time_scale = astro_data.MSgrA_s  # Sgr A* mass in seconds
        sage: psd = lisa_detector.power_spectral_density
        sage: fmin, fmax = 1e-5, 2e-2
        sage: mu_ov_r = astro_data.Msol_m / astro_data.dSgrA  # mu/r
        sage: snr = signal_to_noise(h, time_scale, psd, fmin, fmax,
        ....:                       scale=mu_ov_r)
        sage: snr  # tol 1.0e-13
        5382.189120880952

    """
    sig = [(t*time_scale, h) for (t, h) in signal]
    hf = fourier(sig)
    hf2 = [(f, abs(h)**2) for (f, h) in hf]
    hf2_pos = [(f, h) for (f, h) in hf2 if f >=0]
    hf2_func = spline(hf2_pos)
    def hf2_ov_Sn(f):
        return hf2_func(f) / psd(f)
    integ = quad(hf2_ov_Sn, fmin, fmax, epsrel=quad_epsrel, limit=quad_limit)
    return 2*sqrt(integ[0])*scale
