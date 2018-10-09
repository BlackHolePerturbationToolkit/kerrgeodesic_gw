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
      is the signal at `t`.

    OUTPUT:

    - a numerical approximation (FFT) of the Fourier transform, as a list
      of pairs `(f,\tilde{h}(f))`

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

    """
    if dirname:
        file_name = os.path.join(dirname, filename)
    else:
        file_name = filename
    with open(file_name, "w") as output_file:
        for t, h in sig:
            output_file.write("{}\t{}\n".format(t, h))
