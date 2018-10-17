r"""
Spin-weighted spheroidal harmonics

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

from sage.functions.other import abs, ceil, sqrt
from sage.functions.generalized import sgn
from sage.matrix.constructor import matrix
from sage.rings.integer_ring import ZZ
from sage.rings.real_double import RDF
from sage.symbolic.constants import pi
from sage.symbolic.expression import Expression
from .spin_weighted_spherical_harm import spin_weighted_spherical_harmonic

_eigenvectors = dict()

def kHat(s, l, m, gamma):
    if l == 0 and m  == 0:
         return gamma*gamma/RDF(3)
    else:
         return -(l*(1 + l)) + (2*gamma*m*s**2)/(l + l**2) + (gamma**2*(1 +
                (2*(l + l**2 - 3*m**2)*(l + l**2 - 3*s**2))/
                (l*(-3 + l + 8*l**2 + 4*l**3))))/RDF(3)

def k2(s, l, m, gamma):
    return (gamma**2*sqrt(RDF((1 + l - m)*(2 + l - m)*(1 + l + m)*(2 + l + m)
                              *(1 + l - s)*(2 + l - s)*(1 + l + s)*(2 + l + s))
                          /RDF((1 + 2*l)*(5 + 2*l))))/RDF((1 + l)*(2 + l)*(3 + 2*l))

def kTilde2(s, l, m, gamma):
    if l == 0 and m == 0:
        return -2*gamma*s*sqrt((1 - s**2)/RDF(3))
    else:
        return (-2*gamma*(2*l + l**2 + gamma*m)*s
                *sqrt(((1 + 2*l + l**2 - m**2)*(1 + 2*l + l**2 - s**2))
                      /RDF(3 + 8*l + 4*l**2)))/RDF(l*(2 + 3*l + l**2))


def _compute_eigenvector(s, l, m, gam, verbose=False, min_nmax=8):
    r"""
    Compute the eigenvector and the eigenvalue corresponding to (s, l, m, gam)

    INPUT:

    - ``s`` -- integer; the spin weight
    - ``l`` -- non-negative integer; the harmonic degree
    - ``m`` -- integer within the range ``[-l, l]``; the azimuthal number
    - ``gam`` -- spheroidicity parameter
    - ``verbose`` -- (default: ``False``) determines whether some details of the
      computation are printed out
    - ``min_nmax`` -- (default: 8) integer; floor for the evaluation of the
      parameter ``nmax``, which sets the highest degree of the spherical
      harmonic expansion as ``l+nmax``.

    """
    nmax = ceil(abs(3*gam/2 - gam*gam/250)) + min_nmax  # FIXME : improve the estimate of nmax
    if nmax%2 == 0:
        nmax += 1
    lmin = max(abs(s), abs(m))
    nmin = min(l-lmin, nmax)
    size = nmax + nmin + 1
    mat = matrix(RDF, size, size)
    for i in range(1, size+1):
        mat[i-1, i-1] = - kHat(s, l - nmin - 1 + i, m, gam)
        if i > 2:
            mat[i-1, i-3] = - k2(s, l - nmin - 3 + i, m, gam)
        if i > 1:
            mat[i-1, i-2] = - kTilde2(s, l - nmin + i - 2, m, gam)
        if (i < size):
            mat[i-1, i] = - kTilde2(s, l - nmin + i - 1, m, gam)
        if (i < size-1):
            mat[i-1, i+1] = - k2(s, l - nmin + i + -1, m, gam)
    if verbose:
        print("nmax: {}".format(nmax))
        print("lmin: {}".format(lmin))
        print("nmin: {}".format(nmin))
        print("size: {}".format(size))
        # show(mat)
    # Computation of the eigenvalues and eigenvectors:
    evlist = mat.eigenvectors_right()  # list of triples, each triple being
                                       # (eigenvalue, [eigenvector], 1)
    sevlist = sorted(evlist, key=lambda x: x[0], reverse=True)
    egval, egvec, mult =  sevlist[-(nmin + 1)]
    egvec = egvec[0]  # since egvec is the single-element list [eigenvector]
    if verbose:
        print("eigenvalue: {}".format(egval))
        print("eigenvector: {}".format(egvec))
        check = mat*egvec - egval*egvec
        print("check: {}".format(check))
    lamb = egval - s*(s + 1) - 2*m*gam + gam*gam
    return lamb, egvec, nmin, nmax, lmin

def spin_weighted_spheroidal_eigenvalue(s, l, m, gamma, verbose=False,
                                        cached=True, min_nmax=8):
    r"""
    Return the spin-weighted oblate spheroidal eigenvalue of spin weight
    ``s``, degree ``l``, azimuthal order ``m`` and spheroidicity ``gamma``.

    INPUT:

    - ``s`` -- integer; the spin weight
    - ``l`` -- non-negative integer; the harmonic degree
    - ``m`` -- integer within the range ``[-l, l]``; the azimuthal number
    - ``gamma`` -- spheroidicity parameter `\gamma`
    - ``verbose`` -- (default: ``False``) determines whether some details of the
      computation are printed out
    - ``cached`` -- (default: ``True``) determines whether the eigenvalue and
      the eigenvectors shall be cached; setting ``cached`` to ``False`` forces
      a new computation, without caching the result
    - ``min_nmax`` -- (default: 8) integer; floor for the evaluation of the
      parameter ``nmax``, which sets the highest degree of the spherical
      harmonic expansion as ``l+nmax``.

    OUTPUT:

    - eigenvalue `\lambda` related to the eigenvalue `\mathcal{E}_{\ell m}` of
      the spheroidal harmonic by

    .. MATH::

        \lambda = \mathcal{E}_{\ell m} - 2 m \gamma + \gamma^2 - s(s+1)

    ALGORITHM:

    The method is adapted from that exposed in Appendix A of S.A. Hughes,
    Phys. Rev. D **61**, 084004 (2000) [:doi:`10.1103/PhysRevD.61.084004`].

    EXAMPLES::

        sage: from kerrgeodesic_gw import spin_weighted_spheroidal_eigenvalue
        sage: spin_weighted_spheroidal_eigenvalue(-2, 2, 1, 1.2)  # tol 1.0e-13
        0.5167945263162421
        sage: spin_weighted_spheroidal_eigenvalue(-2, 2, 1, 1.2, cached=False)  # tol 1.0e-13
        0.5167945263162421
        sage: spin_weighted_spheroidal_eigenvalue(-2, 2, 1, 0)
        4.0

    """
    global _eigenvectors
    s = ZZ(s)  # ensure that we are dealing with Sage integers
    l = ZZ(l)
    m = ZZ(m)
    gamma = RDF(gamma)  # all computations are performed with RDF
    param = (s, l, m, gamma)
    if cached:
        if param not in _eigenvectors:
            _eigenvectors[param] = _compute_eigenvector(*param, verbose=verbose,
                                                        min_nmax=min_nmax)
        eigenval = _eigenvectors[param][0]
    else:
        eigenval = _compute_eigenvector(*param, verbose=verbose,
                                        min_nmax=min_nmax)[0]
    return eigenval

def spin_weighted_spheroidal_harmonic(s, l, m, gamma, theta, phi,
                                      verbose=False, cached=True, min_nmax=8):
    r"""
    Return the spin-weighted oblate spheroidal harmonic of spin weight ``s``,
    degree ``l``, azimuthal order ``m`` and spheroidicity ``gamma``.

    INPUT:

    - ``s`` -- integer; the spin weight
    - ``l`` -- non-negative integer; the harmonic degree
    - ``m`` -- integer within the range ``[-l, l]``; the azimuthal number
    - ``gamma`` -- spheroidicity parameter
    - ``theta`` -- colatitude angle
    - ``phi`` -- azimuthal angle
    - ``verbose`` -- (default: ``False``) determines whether some details of
      the computation are printed out
    - ``cached`` -- (default: ``True``) determines whether the eigenvectors
      shall be cached; setting ``cached`` to ``False`` forces a new
      computation, without caching the result
    - ``min_nmax`` -- (default: 8) integer; floor for the evaluation of the
      parameter ``nmax``, which sets the highest degree of the spherical
      harmonic expansion as ``l+nmax``.

    OUTPUT:

    - value of `{}_s S_{lm}^\gamma(\theta,\phi)`

    ALGORITHM:

    The spin-weighted oblate spheroidal harmonics are computed by an expansion
    over spin-weighted *spherical* harmonics, the coefficients of the expansion
    being obtained by solving an eigenvalue problem, as exposed in Appendix A
    of S.A. Hughes, Phys. Rev. D **61**, 084004 (2000)
    [:doi:`10.1103/PhysRevD.61.084004`].

    EXAMPLES::

        sage: from kerrgeodesic_gw import spin_weighted_spheroidal_harmonic
        sage: spin_weighted_spheroidal_harmonic(-2, 2, 2, 1.1, pi/2, 0)  # tol 1.0e-13
        0.08702532727529422
        sage: spin_weighted_spheroidal_harmonic(-2, 2, 2, 1.1, pi/3, pi/3)  # tol 1.0e-13
        -0.14707166027821453 + 0.25473558795537715*I
        sage: spin_weighted_spheroidal_harmonic(-2, 2, 2, 1.1, pi/3, pi/3, cached=False)   # tol 1.0e-13
        -0.14707166027821453 + 0.25473558795537715*I
        sage: spin_weighted_spheroidal_harmonic(-2, 2, 2, 1.1, pi/3, pi/4)  # tol 1.0e-13
        1.801108380050024e-17 + 0.2941433205564291*I
        sage: spin_weighted_spheroidal_harmonic(-2, 2, -1, 1.1, pi/3, pi/3, cached=False)  # tol 1.0e-13
        0.11612826056899399 - 0.20114004750009495*I

    Test that the relation
    `{}_s S_{lm}^\gamma(\theta,\phi) = (-1)^{l+s}\, {}_s S_{l,-m}^{-\gamma}(\pi-\theta,-\phi)`
    [cf. Eq. (2.3) of :arxiv:`1810.00432`], which is used to evaluate
    `{}_s S_{lm}^\gamma(\theta,\phi)` when `m<0` and ``cached`` is ``True``,
    is correctly implemented::

        sage: spin_weighted_spheroidal_harmonic(-2, 2, -2, 1.1, pi/3, pi/3)  # tol 1.0e-13
        -0.04097260436590737 - 0.07096663248016997*I
        sage: abs(_ - spin_weighted_spheroidal_harmonic(-2, 2, -2, 1.1, pi/3, pi/3,
        ....:                                           cached=False)) < 1.e-13
        True
        sage: spin_weighted_spheroidal_harmonic(-2, 3, -1, 1.1, pi/3, pi/3)  # tol 1.0e-13
        0.1781880511506843 - 0.3086307578946672*I
        sage: abs(_ - spin_weighted_spheroidal_harmonic(-2, 3, -1, 1.1, pi/3, pi/3,
        ....:                                           cached=False)) < 1.e-13
        True

    """
    global _eigenvectors
    if m<0 and cached:
        # We use the symmetry formula
        #  {}_s S_{lm}^\gamma(\theta,\phi) =
        #                  (-1)^{l+s} {}_s S_{l,-m}^{-\gamma}(\pi-\theta,-\phi)
        # cf. Eq. (2.3) of  https://arxiv.org/abs/1810.00432
        return (-1)**(l+s)*spin_weighted_spheroidal_harmonic(s, l, -m, -gamma,
                                pi-theta, -phi, verbose=verbose, cached=cached,
                                min_nmax=min_nmax)
    s = ZZ(s)  # ensure that we are dealing with Sage integers
    l = ZZ(l)
    m = ZZ(m)
    gamma = RDF(gamma)  # all computations are performed with RDF
    param = (s, l, m, gamma)
    if cached:
        if param not in _eigenvectors:
            _eigenvectors[param] = _compute_eigenvector(*param, verbose=verbose,
                                                        min_nmax=min_nmax)
        data = _eigenvectors[param]
    else:
        data = _compute_eigenvector(*param, verbose=verbose, min_nmax=min_nmax)
    egvec = data[1]
    nmin = data[2]
    nmax = data[3]
    lmin = data[4]
    # If neither theta nor phi is symbolic, we convert both of them to RDF:
    if not ((isinstance(theta, Expression) and theta.variables())
            or (isinstance(phi, Expression) and phi.variables())):
        theta = RDF(theta)
        phi = RDF(phi)
    resu = RDF(0)
    for k in range(-nmin, nmax+1):
        resu += egvec[k + nmin] \
                 * spin_weighted_spherical_harmonic(s, l+k, m, theta, phi,
                                                    condon_shortley=True)
    resu *= sgn(egvec[min(l - lmin + 1, (nmax + nmin)/2 + 1)-1])
    return resu
