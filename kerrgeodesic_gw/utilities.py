r"""
Utilities
"""

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
    # Creation of the Matplotlib Figure object
    # ----------------------------------------
    options = {}
    options.update(main.SHOW_OPTIONS)  # options of class Graphics for show()
    options.update(main._extra_kwds)   # options set in main
    # We get rid of options that are not relevant for main.matplotlib():
    options.pop('dpi')
    options.pop('transparent')
    options.pop('fig_tight')
    figure = main.matplotlib(**options)
    # Creation of the Matplotlib Axes object for the inset
    # ----------------------------------------------------
    scale = inset._extra_kwds['scale']
    scale_dict = {}
    if scale == 'loglog':
        scale_dict = {'xscale': 'log', 'yscale': 'log'}
    elif scale == 'semilogx':
        scale_dict = {'xscale': 'log'}
    elif scale == 'semilogy':
        scale_dict = {'yscale': 'log'}
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
    return figure
