r"""
Geodesics of Kerr spacetime are implemented via the class :class:`KerrGeodesic`.

"""
#******************************************************************************
#       Copyright (C) 2020 Eric Gourgoulhon <eric.gourgoulhon@obspm.fr>
#
#  Distributed under the terms of the GNU General Public License (GPL)
#  as published by the Free Software Foundation; either version 2 of
#  the License, or (at your option) any later version.
#                  http://www.gnu.org/licenses/
#******************************************************************************

from sage.functions.trig import cos, sin, acos
from sage.functions.other import sqrt
from sage.symbolic.expression import Expression
from sage.symbolic.relation import solve
from sage.symbolic.ring import SR
from sage.rings.real_mpfr import RealField, RR
from sage.plot.circle import circle
from sage.plot.plot3d.shapes import Cylinder
from sage.plot.plot3d.shapes2 import sphere
from sage.repl.rich_output.pretty_print import pretty_print
from sage.misc.table import table
from sage.manifolds.differentiable.integrated_curve import IntegratedGeodesic

class KerrGeodesic(IntegratedGeodesic):
    r"""
    Geodesic of Kerr spacetime.

    Geodesics are computed by solving the geodesic equation via
    the generic
    `SageMath geodesic integrator <https://doc.sagemath.org/html/en/reference/manifolds/sage/manifolds/differentiable/integrated_curve.html>`_.

    INPUT:

    - ``parent`` --
      `IntegratedGeodesicSet <https://doc.sagemath.org/html/en/reference/manifolds/sage/manifolds/differentiable/manifold_homset.html>`_, the set of curves `\mathrm{Hom_{geodesic}}(I, M)` to
      which the geodesic belongs
    - ``affine_parameter`` -- symbolic variable to be used for the affine
      parameter of the geodesic
    - ``initial_point`` -- point of Kerr spacetime from which the geodesic
      is to be integrated
    - ``pt0`` -- (default: ``None``) Boyer-Lindquist component `p^t` of the
      initial 4-momentum vector
    - ``pr0`` -- (default: ``None``) Boyer-Lindquist component `p^r` of the
      initial 4-momentum
      vector
    - ``pth0`` -- (default: ``None``) Boyer-Lindquist component `p^\theta` of
      the initial 4-momentum vector
    - ``pph0`` -- (default: ``None``) Boyer-Lindquist component `p^\phi` of
      the initial 4-momentum vector
    - ``mu`` -- (default: ``None``) mass `\mu` of the particle
    - ``E`` -- (default: ``None``) conserved energy `E` of the particle
    - ``L`` -- (default: ``None``) conserved angular momemtum `L` of the
      particle
    - ``Q`` -- (default: ``None``) Carter constant `Q` of the particle
    - ``r_increase`` -- (default: ``True``) boolean; if ``True``, the initial
      value of `p^r=\mathrm{d}r/\mathrm{d}\lambda` determined from the integral
      of motions is positive or zero, otherwise, `p^r` is negative
    - ``th_increase`` -- (default: ``True``) boolean; if ``True``, the initial
      value of `p^\theta=\mathrm{d}\theta/\mathrm{d}\lambda` determined from
      the integral of motions is positive or zero, otherwise, `p^\theta` is
      negative
    - ``chart`` -- (default: ``None``) chart on the spacetime manifold in terms
      of which the geodesic equations are expressed; if ``None`` the default
      chart (Boyer-Lindquist coordinates) is assumed
    - ``name`` -- (default: ``None``) string; symbol given to the geodesic
    - ``latex_name`` -- (default: ``None``) string; LaTeX symbol to denote
      the geodesic; if none is provided, ``name`` will be used
    - ``a_num`` -- (default: ``None``) numerical value of the Kerr spin
      parameter `a` (required for a numerical integration)
    - ``m_num`` -- (default: ``None``) numerical value of the Kerr mass
      parameter `m` (required for a numerical integration)
    - ``verbose`` -- (default: ``False``) boolean; determines whether some
      information is printed during the construction of the geodesic

    EXAMPLES:

    We construct first the Kerr spacetime::

        sage: from kerrgeodesic_gw import KerrBH
        sage: a = var('a')
        sage: M = KerrBH(a); M
        Kerr spacetime M
        sage: BLchart = M.boyer_lindquist_coordinates(); BLchart
        Chart (M, (t, r, th, ph))

    We pick an initial spacetime point for the geodesic::

        sage: init_point = M((0, 6, pi/2, 0), name='P')

    and introduce the affine parameter `\lambda` along the geodesic as a
    symbolic variable (we cannot name it ``lambda``, since this is reserved
    keyword in Python)::

        sage: lamb = var('lamb', latex_name=r'\lambda')

    A geodesic is constructed by providing the range of the affine
    parameter, the initial point and either (i) the Boyer-Lindquist components
    `(p^t_0, p^r_0, p^\theta_0, p^\phi_0)` of the initial 4-momentum vector
    `p_0 = \left. \frac{\mathrm{d}x}{\mathrm{d}\lambda}\right| _{\lambda=0}`,
    (ii) the four integral of motions `(\mu, E, L, Q)` or (iii) some of the
    components of `p_0` along with with some integrals of motion. We shall
    also specify some numerical value for the Kerr spin parameter `a`.
    Here, we choose `\lambda\in[0, 300m]`, the option (ii) and `a=0.998 m`,
    where `m` in the black hole mass::

        sage: geod = M.geodesic((lamb, 0, 300), init_point, mu=1, E=0.883,
        ....:                   L=1.982, Q=0.467, a_num=0.998)
        sage: geod
        Geodesic of the Kerr spacetime M

    The numerical integration of the geodesic equation is performed via
    :meth:`integrate`, by providing the step in `\delta\lambda` in units of
    `m`::

        sage: geod.integrate(step=0.005)

    We can then plot the geodesic::

        sage: geod.plot()
        Graphics3d Object

    .. PLOT::

        from kerrgeodesic_gw import KerrBH
        a = var('a')
        M = KerrBH(a)
        M.boyer_lindquist_coordinates()
        init_point = M((0, 6, pi/2, 0), name='p0')
        lamb = var('lamb', latex_name=r'\lambda')
        geod = M.geodesic((lamb, 0, 300), init_point, mu=1, E=0.883, \
                          L=1.982, Q=0.467, a_num=0.998)
        geod.integrate(step=0.005)
        graph = geod.plot(label_axes=True)  # True is required for jmol
        graph._extra_kwds['aspect_ratio'] = 1 # for jmol
        sphinx_plot(graph)

    Actually, many options can be passed to :meth:`plot`. For instance to
    a get a 3D spacetime diagram::

        sage: geod.plot(coordinates='txy')
        Graphics3d Object

    .. PLOT::

        from kerrgeodesic_gw import KerrBH
        a = var('a')
        M = KerrBH(a)
        M.boyer_lindquist_coordinates()
        init_point = M((0, 6, pi/2, 0), name='p0')
        lamb = var('lamb', latex_name=r'\lambda')
        geod = M.geodesic((lamb, 0, 300), init_point, mu=1, E=0.883, \
                          L=1.982, Q=0.467, a_num=0.998)
        geod.integrate(step=0.005)
        graph = geod.plot(coordinates='txy', label_axes=True)
        sphinx_plot(graph)

    or to get the trace of the geodesic in the `(x,y)` plane::

        sage: geod.plot(coordinates='xy', plot_points=2000)
        Graphics object consisting of 2 graphics primitives

    .. PLOT::

        from kerrgeodesic_gw import KerrBH
        a = var('a')
        M = KerrBH(a)
        M.boyer_lindquist_coordinates()
        init_point = M((0, 6, pi/2, 0), name='p0')
        lamb = var('lamb', latex_name=r'\lambda')
        geod = M.geodesic((lamb, 0, 300), init_point, mu=1, E=0.883, \
                          L=1.982, Q=0.467, a_num=0.998)
        geod.integrate(step=0.005)
        graph = geod.plot(coordinates='xy', plot_points=2000)
        sphinx_plot(graph)

    or else to get the trace in the `(x,z)` plane::

        sage: geod.plot(coordinates='xz')
        Graphics object consisting of 2 graphics primitives

    .. PLOT::

        from kerrgeodesic_gw import KerrBH
        a = var('a')
        M = KerrBH(a)
        M.boyer_lindquist_coordinates()
        init_point = M((0, 6, pi/2, 0), name='p0')
        lamb = var('lamb', latex_name=r'\lambda')
        geod = M.geodesic((lamb, 0, 300), init_point, mu=1, E=0.883, \
                          L=1.982, Q=0.467, a_num=0.998)
        geod.integrate(step=0.005)
        graph = geod.plot(coordinates='xz')
        sphinx_plot(graph)

    As a curve, the geodesic is a map from an interval of `\mathbb{R}` to the
    spacetime `M`::

        sage: geod.display()
        (0, 300) --> M
        sage: geod.domain()
        Real interval (0, 300)
        sage: geod.codomain()
        Kerr spacetime M

    It maps values of `\lambda` to points of spacetime::

        sage: geod(0)
        Point on the Kerr spacetime M
        sage: geod(0).coordinates()  # coordinates in the default chart  # tol 1.0e-13
        (0.0, 6.0, 1.5707963267948966, 0.0)
        sage: BLchart(geod(0))       # equivalent to above   # tol 1.0e-13
        (0.0, 6.0, 1.5707963267948966, 0.0)
        sage: geod(300).coordinates()   # tol 1.0e-13
        (553.4637298235969, 3.7035524920469878, 1.6613833561392117, 84.62814586190511)

    The initial 4-momentum vector `p_0` is returned by the method
    :meth:`initial_tangent_vector()`::

        sage: p0 = geod.initial_tangent_vector(); p0
        Tangent vector p at Point P on the Kerr spacetime M
        sage: p0 in M.tangent_space(init_point)
        True
        sage: p0.display()  # tol 1.0e-13
        p = 1.29225788954106 d/dt + 0.00438084990626460 d/dr
            + 0.0189826106258554 d/dth + 0.0646134478134985 d/dph
        sage: p0[:]  # tol 1.0e-13
        [1.29225788954106, 0.00438084990626460, 0.0189826106258554, 0.0646134478134985]

    For instance, the components `p^t_0` and `p^\phi_0` are recovered by::

        sage: p0[0], p0[3]  # tol 1.0e-13
        (1.29225788954106, 0.0646134478134985)

    Let us check that the scalar square of `p_0` is `-1`, i.e. is consistent
    with the mass parameter `\mu = 1` used in the construction of the
    geodesic::

        sage: g = M.metric()
        sage: g.at(init_point)(p0, p0).subs(a=0.998)  # tol 1.0e-13
        -1.00000000000000

    The 4-momentum vector `p` at a any value of the affine parameter `\lambda`,
    e.g. `\lambda=200m`, is obtained by::

        sage: p = geod.tangent_vector_eval_at(200); p
        Tangent vector at Point on the Kerr spacetime M
        sage: p in M.tangent_space(geod(200))
        True
        sage: p.display()  # tol 1.0e-13
        1.3165926052039156 d/dt - 0.07370434461768362 d/dr
         - 0.010911954123803514 d/dth + 0.07600209953046694 d/dph

    The particle mass `\mu` computed at a given value of  `\lambda` is returned
    by the method :meth:`evaluate_mu`::

        sage: geod.evaluate_mu(0)  # tol 1.0e-13
        1.00000000000000

    Of course, it should be conserved along the geodesic; actually it is, up
    to the numerical accuracy::

        sage: geod.evaluate_mu(300)  # tol 1.0e-13
        1.0000117991483337

    Similarly, the conserved energy `E`, conserved angular momentum `L` and
    Carter constant `Q` are computed at any value of `\lambda` by respectively
    :meth:`evaluate_E`, :meth:`evaluate_L` and :meth:`evaluate_Q`::

        sage: geod.evaluate_E(0)  # tol 1.0e-13
        0.883000000000000
        sage: geod.evaluate_L(0)  # tol 1.0e-13
        1.98200000000000
        sage: geod.evaluate_Q(0)  # tol 1.0e-13
        0.467000000000000

    Let us check that the values of `\mu`, `E`, `L` and `Q` evaluated at
    `\lambda=300 m` are equal to those at `\lambda=0` up to the numerical
    accuracy of the integration scheme::

        sage: geod.check_integrals_of_motion(300)  # tol 1.0e-13
          quantity         value            initial value       diff.      relative diff.
           $\mu$     1.0000117991483337   1.00000000000000    0.00001180     0.00001180
            $E$      0.883067996752519    0.883000000000000   0.00006800     0.00007701
            $L$       1.98248083176764    1.98200000000000    0.0004808      0.0002426
            $Q$      0.467214121733873    0.467000000000000   0.0002141      0.0004585

    Decreasing the integration step results in smaller errors::

        sage: geod.integrate(step=0.001)
        sage: geod.check_integrals_of_motion(300)  # tol 1.0e-13
          quantity         value            initial value       diff.      relative diff.
           $\mu$     1.0000023592012068   1.00000000000000     2.359e-6       2.359e-6
            $E$      0.883013605582177    0.883000000000000   0.00001361     0.00001541
            $L$       1.98209628664691    1.98200000000000    0.00009629     0.00004858
            $Q$      0.467042767529623    0.467000000000000   0.00004277     0.00009158

    """
    def __init__(self, parent, affine_parameter,
                 initial_point, pt0=None, pr0=None, pth0=None, pph0=None,
                 mu=None, E=None, L=None, Q=None, r_increase=True,
                 th_increase=True, chart=None, name=None,
                 latex_name=None, a_num=None, m_num=None, verbose=False):
        self._spacetime = parent.codomain()
        self._mu = mu
        self._E = E
        self._L = L
        self._Q = Q
        self._a = a_num
        if self._a is None:
            self._a = self._spacetime.spin()
        self._m = m_num
        if self._m is None:
            self._m = self._spacetime.mass()
        self._init_vector = self._compute_init_vector(initial_point, pt0, pr0,
                                                      pth0, pph0, r_increase,
                                                      th_increase, verbose)
        if verbose:
            print("Initial tangent vector: ")
            pretty_print(self._init_vector.display())
        metric = self._spacetime.metric()
        IntegratedGeodesic.__init__(self, parent, metric, affine_parameter,
                                    self._init_vector, chart=chart,
                                    name=name, latex_name=latex_name,
                                    verbose=verbose)

    def _repr_(self):
        r"""
        Return a string representation of ``self``.
        """
        description = "Geodesic "
        if self._name is not None:
            description += self._name + " "
        description += "of the {}".format(self._spacetime)
        return description

    def _compute_init_vector(self, point, pt0, pr0, pth0, pph0, r_increase,
                             th_increase, verbose):
        r"""
        Computes the initial 4-momentum vector `p` from the constants of motion
        """
        BLchart = self._spacetime.boyer_lindquist_coordinates()
        basis = BLchart.frame().at(point)
        r, th = BLchart(point)[1:3]
        a, m = self._a, self._m
        r2 = r**2
        a2 = a**2
        rho2 = r2 + (a*cos(th))**2
        Delta = r2 - 2*m*r + a2
        if pt0 is None:
            if (self._mu is not None and pr0 is not None and pth0 is not None
                and pph0 is not None):
                xxx = SR.var('xxx')
                v = self._spacetime.tangent_space(point)((xxx, pr0, pth0, pph0),
                                                         basis=basis)
                muv2 = - self._spacetime.metric().at(point)(v, v)
                muv2 = muv2.substitute(self._numerical_substitutions())
                solutions = solve(muv2 == self._mu**2, xxx, solution_dict=True)
                if verbose:
                    print("Solutions for p^t:")
                    pretty_print(solutions)
                for sol in solutions:
                    if sol[xxx] > 0:
                        pt0 = sol[xxx]
                        break
                else:  # pt0 <= 0 might occur in the ergoregion
                    pt0 = solutions[0][xxx]
                try:
                    pt0 = RR(pt0)
                except TypeError:  # pt0 contains some symbolic expression
                    pass
            else:
                if self._E is None:
                    raise ValueError("the constant E must be provided")
                if self._L is None:
                    raise ValueError("the constant L must be provided")
                E, L = self._E, self._L
                pt0 = ((r2 + a2)/Delta*((r2 + a2)*E - a*L)
                       + a*(L - a*E*sin(th)**2)) / rho2
        if pph0 is None:
            if self._E is None:
                raise ValueError("the constant E must be provided")
            if self._L is None:
                raise ValueError("the constant L must be provided")
            E, L = self._E, self._L
            pph0 = (L/sin(th)**2 - a*E + a/Delta*((r2 + a2)*E - a*L)) / rho2
        if pr0 is None:
            if self._E is None:
                raise ValueError("the constant E must be provided")
            if self._L is None:
                raise ValueError("the constant L must be provided")
            if self._mu is None:
                raise ValueError("the constant mu must be provided")
            if self._Q is None:
                raise ValueError("the constant Q must be provided")
            E, L, Q = self._E, self._L, self._Q
            mu2 = self._mu**2
            E2_mu2 = E**2 - mu2
            pr0 = sqrt((E2_mu2)*r**4 + 2*m*mu2*r**3
                       + (a2*E2_mu2 - L**2 - Q)*r**2
                       + 2*m*(Q + (L - a*E)**2)*r - a2*Q) / rho2
            if not r_increase:
                pr0 = - pr0
        if pth0 is None:
            if self._E is None:
                raise ValueError("the constant E must be provided")
            if self._L is None:
                raise ValueError("the constant L must be provided")
            if self._mu is None:
                raise ValueError("the constant mu must be provided")
            if self._Q is None:
                raise ValueError("the constant Q must be provided")
            E2 = self._E**2
            L2 = self._L**2
            mu2 = self._mu**2
            Q = self._Q
            pth0 = sqrt(Q + cos(th)**2*(a2*(E2 - mu2)
                                        - L2/sin(th)**2)) / rho2
            if not th_increase:
                pth0 = - pth0
        return self._spacetime.tangent_space(point)((pt0, pr0, pth0, pph0),
                                                    basis=basis, name='p')
    def initial_tangent_vector(self):
        r"""
        Return the initial tangent vector.
        """
        return self._init_vector

    def _numerical_substitutions(self):
        r"""
        Helper function to substitute the numerical values for a and m.
        """
        subs = {}
        a = self._spacetime.spin()
        if isinstance(a, Expression) and self._a != a:
            subs[a] = self._a
        m = self._spacetime.mass()
        if isinstance(m, Expression) and self._m != m:
            subs[m] = self._m
        return subs

    def integrate(self, step=None, method='odeint', solution_key=None,
                  parameters_values=None, verbose=False, **control_param):
        r"""
        Perform the numerical integration.
        """
        # Substituting a and m by their numerical values:
        if parameters_values is None:
            parameters_values = {}
        parameters_values.update(self._numerical_substitutions())
        self.solve(step=step, method=method, solution_key=solution_key,
                   parameters_values=parameters_values, verbose=verbose,
                   **control_param)
        self.interpolate(solution_key=solution_key, verbose=verbose)

    def evaluate_mu(self, affine_parameter):
        r"""
        Compute the mass parameter `\mu` at a given value of the affine
        parameter `\lambda`
        """
        if affine_parameter == self.domain().lower_bound():
            p = self._init_vector
        else:
            p = self.tangent_vector_eval_at(affine_parameter)
        point = p.parent().base_point()
        g = self._spacetime.metric().at(point)
        mu = sqrt(-g(p, p))
        return mu.substitute(self._numerical_substitutions())

    def evaluate_E(self, affine_parameter):
        r"""
        Compute the conserved energy `E` at a given value of the affine
        parameter `\lambda`.
        """
        if affine_parameter == self.domain().lower_bound():
            p = self._init_vector
        else:
            p = self.tangent_vector_eval_at(affine_parameter)
        point = p.parent().base_point()
        BLchart = self._spacetime.boyer_lindquist_coordinates()
        r, th = BLchart(point)[1:3]
        a, m = self._a, self._m
        rho2 = r**2 + (a*cos(th))**2
        p_comp = p.components(basis=BLchart.frame().at(point))
        pt = p_comp[0]
        pph = p_comp[3]
        b = 2*m*r/rho2
        E = (1 - b)*pt + b*a*sin(th)**2*pph
        return E.substitute(self._numerical_substitutions())

    def evaluate_L(self, affine_parameter):
        r"""
        Compute the conserved angular momentum about the rotation axis `L` at
        a given value of the affine parameter `\lambda`.
        """
        if affine_parameter == self.domain().lower_bound():
            p = self._init_vector
        else:
            p = self.tangent_vector_eval_at(affine_parameter)
        point = p.parent().base_point()
        BLchart = self._spacetime.boyer_lindquist_coordinates()
        r, th = BLchart(point)[1:3]
        a, m = self._a, self._m
        r2 = r**2
        a2 = a**2
        rho2 = r2 + (a*cos(th))**2
        p_comp = p.components(basis=BLchart.frame().at(point))
        pt = p_comp[0]
        pph = p_comp[3]
        bs = 2*a*m*r*sin(th)**2/rho2
        L = -bs*pt + (r2 + a2 + a*bs)*sin(th)**2*pph
        return L.substitute(self._numerical_substitutions())

    def evaluate_Q(self, affine_parameter):
        r"""
        Compute the Carter constant `Q` at a given value of the affine
        parameter `\lambda`.
        """
        if affine_parameter == self.domain().lower_bound():
            p = self._init_vector
        else:
            p = self.tangent_vector_eval_at(affine_parameter)
        point = p.parent().base_point()
        BLchart = self._spacetime.boyer_lindquist_coordinates()
        r, th = BLchart(point)[1:3]
        a = self._a
        rho4 = (r**2 + (a*cos(th))**2)**2
        mu2 = self.evaluate_mu(affine_parameter)**2
        E2 = self.evaluate_E(affine_parameter)**2
        L2 = self.evaluate_L(affine_parameter)**2
        p_comp = p.components(basis=BLchart.frame().at(point))
        pth = p_comp[2]
        Q = rho4*pth**2 + cos(th)**2*(L2/sin(th)**2 + a**2*(mu2 - E2))
        return Q.substitute(self._numerical_substitutions())

    def plot(self, coordinates='xyz', prange=None, interpolation_key=None,
             style='-', thickness=1, plot_points=1000, color='red',
             include_end_point=(True, True), end_point_offset=(0.001, 0.001),
             verbose=False, label_axes=None, plot_horizon=True,
             horizon_color='grey', display_tangent=False, color_tangent='blue',
             plot_points_tangent=10, width_tangent=1,
             scale=1, aspect_ratio='automatic', **kwds):
        r"""
        Plot the geodesic in terms of the coordinates `(t,x,y,z)` deduced from
        the Boyer-Lindquist coordinates `(t,r,\theta,\phi)` via the standard
        polar to Cartesian transformations.
        """
        bad_format_msg = ("the argument 'coordinates' must be a string of "
                          "2 or 3 characters among {t,x,y,z} denoting the "
                          "coordinates involved in the plot")
        if len(coordinates) < 2 or len(coordinates) > 3:
            raise ValueError(bad_format_msg)
        map_to_Euclidean = self._spacetime.map_to_Euclidean()
        X4 = map_to_Euclidean.codomain().cartesian_coordinates()
        t, x, y, z = X4[:]
        coord_dict = {'t': t, 'x': x, 'y': y, 'z': z}
        try:
            ambient_coords = [coord_dict[c] for c in coordinates]
        except KeyError:
            raise ValueError(bad_format_msg)
        if ambient_coords[0] == t:
            ambient_coords = ambient_coords[1:] + [t]
        kwds['axes_labels'] = [repr(c) for c in ambient_coords] # unfortunately this option
                                                                # is not forwarded to
                                                                # three.js in Sage <= 9.1
        if label_axes is None:  # the default
            if len(ambient_coords) == 2:
                label_axes = True
            else:
                label_axes = False  # since three.js has its own mechanism to
                                    # label axes
        graph = self.plot_integrated(chart=X4, mapping=map_to_Euclidean,
               ambient_coords=ambient_coords, prange=prange,
               interpolation_key=interpolation_key, style=style,
               thickness=thickness, plot_points=plot_points, color=color,
               include_end_point=include_end_point,
               end_point_offset=end_point_offset, verbose=verbose,
               label_axes=label_axes, display_tangent=display_tangent,
               color_tangent=color_tangent,
               plot_points_tangent=plot_points_tangent,
               width_tangent=width_tangent, scale=scale,
               aspect_ratio=aspect_ratio, **kwds)
        if plot_horizon:
            rH = self._spacetime.event_horizon_radius()
            rH = rH.substitute(self._numerical_substitutions())
            if len(ambient_coords) == 3:
                if t in ambient_coords:
                    BLchart = self._spacetime.boyer_lindquist_coordinates()
                    lambda_min = self.domain().lower_bound()
                    lambda_max = self.domain().upper_bound()
                    tmin = BLchart(self(lambda_min))[0]
                    tmax = BLchart(self(lambda_max))[0]
                    graph += Cylinder(rH, tmax - tmin, color=horizon_color,
                                      aspect_ratio=aspect_ratio)
                else:
                    graph += sphere(size=rH, color=horizon_color,
                                    aspect_ratio=aspect_ratio)
            if len(ambient_coords) == 2 and t not in ambient_coords:
                graph += circle((0,0), rH, color=horizon_color, thickness=2)
        return graph

    def check_integrals_of_motion(self, affine_parameter):
        r"""
        Check the constancy of the four integrals of motion
        """
        RF = RealField(16)
        lambda_min = self.domain().lower_bound()
        res = [["quantity", "value", "initial value", "diff.", "relative diff."]]
        mu = self.evaluate_mu(affine_parameter)
        mu0 = self.evaluate_mu(lambda_min)
        diff = mu - mu0
        rel_diff = RF(diff / mu0) if mu0 != 0 else "-"
        res.append([r"$\mu$", mu, mu0, RF(diff), rel_diff])
        E = self.evaluate_E(affine_parameter)
        E0 = self.evaluate_E(lambda_min)
        diff = E - E0
        rel_diff = RF(diff / E0) if E0 != 0 else "-"
        res.append([r"$E$", E, E0, RF(diff), rel_diff])
        L = self.evaluate_L(affine_parameter)
        L0 = self.evaluate_L(lambda_min)
        diff = L - L0
        rel_diff = RF(diff / L0) if L0 != 0 else "-"
        res.append([r"$L$", L, L0, RF(diff), rel_diff])
        Q = self.evaluate_Q(affine_parameter)
        Q0 = self.evaluate_Q(lambda_min)
        diff = Q - Q0
        rel_diff = RF(diff / Q0) if Q0 != 0 else "-"
        res.append([r"$Q$", Q, Q0, RF(diff), rel_diff])
        return table(res, align="center")
