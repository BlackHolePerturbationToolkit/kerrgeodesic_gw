r"""
Utilities

This module contains some utility functions. For the moment, there is only

- :func:`graphics_inset`: generates plots with insets

"""

#******************************************************************************
#       Copyright (C) 2019 Eric Gourgoulhon <eric.gourgoulhon@obspm.fr>
#
#  Distributed under the terms of the GNU General Public License (GPL)
#  as published by the Free Software Foundation; either version 2 of
#  the License, or (at your option) any later version.
#                  http://www.gnu.org/licenses/
#******************************************************************************

from matplotlib.backends.backend_agg import FigureCanvasAgg


def graphics_inset(main, inset, position, **kwds):
    r"""
    Insert a graphics object as an inset into another graphics object.

    This function compensates for the (current) lack of the inset functionality
    in SageMath graphics.

    The output is not a SageMath graphics object but a Matplotlib object, of
    type ``Figure``.

    INPUT:

    - ``main`` -- a SageMath graphics object
    - ``inset`` -- a SageMath graphics object to be inserted into ``main``
    - ``position`` -- a 4-uple ``(left, bottom, width, height)`` specifying the
      position and size of the inset (all quantities are in fractions of
      figure width and height)
    - ``kwds`` -- options passed to method ``matplotlib.figure.Figure.add_axes``

    OUTPUT:

    - a ``matplotlib.figure.Figure`` object

    EXAMPLES::

        sage: from kerrgeodesic_gw.utilities import graphics_inset
        sage: g1 = plot(x*sin(1/x), (x, -2, 2), axes_labels=[r"$x$", r"$y$"])
        sage: g2 = plot(x*sin(1/x), (x, -0.05, 0.05),
        ....:           axes_labels=[r"$x$", r"$y$"],fontsize=8, frame=True)
        sage: figure = graphics_inset(g1, g2, (0.72, 0.4, 0.25, 0.25))
        sage: figure
        <Figure size 640x480 with 2 Axes>

    .. PLOT::

        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        from kerrgeodesic_gw.utilities import graphics_inset
        g1 = plot(x*sin(1/x), (x, -2, 2), axes_labels=[r"$x$", r"$y$"])
        g2 = plot(x*sin(1/x), (x, -0.05, 0.05), \
                  axes_labels=[r"$x$", r"$y$"],fontsize=8, frame=True)
        figure = graphics_inset(g1, g2, (0.72, 0.4, 0.25, 0.25))
        mpl.rcParams['image.interpolation'] = 'bilinear'
        mpl.rcParams['image.resample'] = False
        mpl.rcParams['figure.figsize'] = [8.0, 6.0]
        mpl.rcParams['figure.dpi'] = 80
        mpl.rcParams['savefig.dpi'] = 100
        fn = tmp_filename(ext=".png")
        figure.savefig(fn)
        img = mpimg.imread(fn)
        plt.imshow(img)
        plt.axis("off")

    The output is a Matplotlib object, of type ``Figure``::

        sage: type(figure)
        <class 'matplotlib.figure.Figure'>

    It can be saved into a pdf file via the method ``savefig``::

        sage: figure.savefig("figure.pdf")

    An example with a Matplotlib keyword::

        sage: figure = graphics_inset(g1, g2, (0.72, 0.4, 0.25, 0.25),
        ....:                         facecolor='yellow')
        sage: figure
        <Figure size 640x480 with 2 Axes>

    .. PLOT::

        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        from kerrgeodesic_gw.utilities import graphics_inset
        g1 = plot(x*sin(1/x), (x, -2, 2), axes_labels=[r"$x$", r"$y$"])
        g2 = plot(x*sin(1/x), (x, -0.05, 0.05), \
                  axes_labels=[r"$x$", r"$y$"],fontsize=8, frame=True)
        figure = graphics_inset(g1, g2, (0.72, 0.4, 0.25, 0.25), \
                                facecolor='yellow')
        mpl.rcParams['image.interpolation'] = 'bilinear'
        mpl.rcParams['image.resample'] = False
        mpl.rcParams['figure.figsize'] = [8.0, 6.0]
        mpl.rcParams['figure.dpi'] = 80
        mpl.rcParams['savefig.dpi'] = 100
        fn = tmp_filename(ext=".png")
        figure.savefig(fn)
        img = mpimg.imread(fn)
        plt.imshow(img)
        plt.axis("off")


    """
    # Creation of the Matplotlib's Figure object
    # ------------------------------------------
    options = {}
    options.update(main.SHOW_OPTIONS)  # options of class Graphics for show()
    options.update(main._extra_kwds)   # options set in main
    # Sage's default style for legends:
    options['legend_options'] = dict(back_color='white', borderpad=0.6,
                                     borderaxespad=None,
                                     columnspacing=None,
                                     fancybox=False, font_family='sans-serif',
                                     font_size='medium', font_style='normal',
                                     font_variant='normal', font_weight='medium',
                                     handlelength=0.05, handletextpad=0.5,
                                     labelspacing=0.02, loc='best',
                                     markerscale=0.6, ncol=1, numpoints=2,
                                     shadow=True, title=None)
    # We get rid of options that are not relevant for main.matplotlib():
    options.pop('dpi')
    options.pop('transparent')
    options.pop('fig_tight')
    figure = main.matplotlib(**options)
    # Creation of the Matplotlib's Axes object for the inset
    # ------------------------------------------------------
    scale = inset._extra_kwds.get('scale')
    scale_dict = {}
    if scale == 'loglog':
        scale_dict['xscale'] = 'log'
        scale_dict['yscale'] = 'log'
    elif scale == 'semilogx':
        scale_dict['xscale'] = 'log'
    elif scale == 'semilogy':
        scale_dict['yscale'] = 'log'
    kwds.update(scale_dict)
    axes_inset = figure.add_axes(position, **kwds)
    # Adding the inset to the figure
    # ------------------------------
    options = {}
    options.update(inset.SHOW_OPTIONS)  # options of class Graphics for show()
    options.update(inset._extra_kwds)   # options set in inset
    # We get rid of options that are not relevant for inset.matplotlib():
    options.pop('dpi')
    options.pop('transparent')
    options.pop('fig_tight')
    # We add the previously created figure and axes_inset to the parameters:
    options.update({'figure': figure, 'sub': axes_inset})
    inset.matplotlib(**options)
    # Corrects a bug in matplotlib() method:
    figure.get_axes()[0].tick_params(labelsize=main._fontsize)
    figure.get_axes()[1].tick_params(labelsize=inset._fontsize)
    # print("inset tightbbox: {}".format(figure.get_axes()[1].get_tightbbox()))
    bg_color = axes_inset.get_facecolor()
    for label in axes_inset.get_xticklabels():
        label.set_backgroundcolor(bg_color)
    for label in axes_inset.get_yticklabels():
        label.set_backgroundcolor(bg_color)
    axes_inset.set_xlabel(axes_inset.get_xlabel(), backgroundcolor=bg_color)
    axes_inset.set_ylabel(axes_inset.get_ylabel(), backgroundcolor=bg_color)
    # Set the canvas to enable direct rendering of the figure
    figure.set_canvas(FigureCanvasAgg(figure))
    return figure
