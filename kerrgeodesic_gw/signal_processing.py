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
from sage.rings.real_double import RDF

def fourier(signal):
    r"""
    Compute the Fourier transform of a time signal.

    The *Fourier transform* of the time signal `h(t)` is defined by

    .. MATH::

        \tilde{h}(f) = \int_{-\infty}^{+\infty} h(t)\,
                     \mathrm{e}^{-2\pi \mathrm{i} f t} \, \mathrm{d}t

    INPUT:

    - ``signal``: list of pairs `(t, h(t))`, where `t` is the time and `h(t)`
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
