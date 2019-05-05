r"""
Utilities
"""

from matplotlib.backends.backend_agg import FigureCanvasAgg


def graphics_inset(main, inset, position, **kwds):
    r"""
    Insert a graphics object as an inset to another one.

    INPUT:

    - ``main`` -- main SageMath graphics object
    - ``inset`` -- inset SageMath graphics object
    - ``position`` -- a 4-uple ``(left, bottom, width, height)`` specifying the
      position and size of the inset (all quantities are in fractions of
      figure width and height)
    - ``kwds`` -- options passed to method ``matplotlib.figure.Figure.add_axes``

    OUTPUT:

    - a ``matplotlib.figure.Figure`` object

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
    scale = inset._extra_kwds['scale']
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
