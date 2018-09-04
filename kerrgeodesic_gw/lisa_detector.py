r"""
Functions relative to the LISA detector.

"""
import os
from sage.calculus.interpolation import spline
from sage.functions.other import sqrt
from sage.functions.log import log
from sage.rings.real_double import RDF

_sensitivity_spline = None

def strain_sensitivity(freq):
    r"""
    Return LISA strain sensitivity at a given frequency.

    INPUT:

    - ``freq`` -- frequency `f` (in `\mathrm{Hz}`)

    OUTPUT:

    - strain sensitivity `S(f)^{1/2}` (in `\mathrm{Hz}^{-1/2}`)

    EXAMPLES::

        sage: from kerrgeodesic_gw import lisa_detector
        sage: lisa_detector.strain_sensitivity(1.e-1)  # tol 1.0e-13
        5.82615031500758e-20
        sage: lisa_detector.strain_sensitivity(1.e-2)  # tol 1.0e-13
        1.654806317072275e-20
        sage: lisa_detector.strain_sensitivity(1.e-3)  # tol 1.0e-13
        1.8082609253700212e-19

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
