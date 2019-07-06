# kerrgeodesic_gw

A [SageMath](http://www.sagemath.org/) package to compute gravitational radiation from material orbiting a Kerr black hole

This package makes use of SageMath functionalities developed through the [SageManifolds](https://sagemanifolds.obspm.fr/) project and is part of the [Black Hole Peturbation Toolkit](http://bhptoolkit.org/).

## Installation

### Requirements

This package requires the Python-based free mathematics software system [SageMath](http://www.sagemath.org/) (at least version 8.2).

*NB:* the version of SageMath shipped with Ubuntu 18.04 is only 8.1; instead of
the Ubuntu package `sagemath`, install then the most recent binary for Ubuntu 18.04
from [SageMath download page](http://www.sagemath.org/download-linux.html).

### Simple installation from PyPI

It suffices to run

    sage -pip install kerrgeodesic_gw

to have the package ready to use in SageMath.
See however *install from source* below if you want to build a
local version of the documentation or modify the source files (development).

*NB:* on the [CoCalc](https://cocalc.com) cloud computing platform, you need
to add the option `--user`, i.e. open a terminal and run

    sage -pip install --user kerrgeodesic_gw


Here is the [kerrgeodesic_gw page](https://pypi.org/project/kerrgeodesic-gw/) on PyPI (the Python Package Index).

### Install from source

Download the source from the git repository:

    git clone https://github.com/BlackHolePerturbationToolkit/kerrgeodesic_gw.git

This creates a directory `kerrgeodesic_gw`.

Run

    sage -pip install --upgrade --no-index -v kerrgeodesic_gw

to install the package in SageMath.
A shortcut of the above command is provided by the `Makefile` distributed with the package:

    cd kerrgeodesic_gw
    make install

*NB:* on [CoCalc](https://cocalc.com), you need to add the option `--user`, i.e. open a terminal and run

    git clone https://github.com/BlackHolePerturbationToolkit/kerrgeodesic_gw.git
    sage -pip install --user --upgrade --no-index -v kerrgeodesic_gw

#### Install for development

If you plan to edit the package source, you should add the option `-e` to the pip install, i.e. run

    sage -pip install --upgrade --no-index -v -e kerrgeodesic_gw

or equivalently

    cd kerrgeodesic_gw
    make develop

## Usage

Once the package is installed, you can use it in SageMath, like for instance:

    sage: from kerrgeodesic_gw import spin_weighted_spherical_harmonic
    sage: theta, phi = var('theta phi')
    sage: spin_weighted_spherical_harmonic(-2, 2, 1, theta, phi)
    1/4*(sqrt(5)*cos(theta) + sqrt(5))*e^(I*phi)*sin(theta)/sqrt(pi)


## Tests

This package is configured for tests written in the documentation strings of the source files, also known as *doctests*.
You may then test the install by running, from the root of the package tree
(i.e. the directory kerrgeodesic_gw created by the `git clone`),

    sage -t kerrgeodesic_gw

You should then get the message `All tests passed!`

Alternatively, you can run (from the same directory)

    make test


## Documentation

The package documentation can be generated using SageMath's [Sphinx](http://www.sphinx-doc.org/) installation:

    cd docs
    sage -sh -c "make html"

A shorthand of the above is

    make doc

The html reference manual is then at

    kerrgeodesic_gw/docs/build/html/index.html

For the LaTeX documentation, use

    make doc-pdf

The pdf reference manual is then

    kerrgeodesic_gw/docs/build/latex/kerrgeodesic_gw.pdf

### Online documentation

- [Reference manual](https://share.cocalc.com/share/2b3f8da9-6d53-4261-b5a5-ff27b5450abb/kerrgeodesic_gw/docs/build/html/index.html)
  ([PDF](https://cocalc.com/share/2b3f8da9-6d53-4261-b5a5-ff27b5450abb/kerrgeodesic_gw/docs/build/latex/kerrgeodesic_gw.pdf))
- [Article describing the formulas implemented in the package](https://doi.org/10.1051/0004-6361/201935406) *(open access)*
- Demo notebooks:

  - [Spin-weighted spheroidal harmonics](https://nbviewer.jupyter.org/github/BlackHolePerturbationToolkit/kerrgeodesic_gw/blob/master/Notebooks/basic_kerrgeodesic_gw.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/BlackHolePerturbationToolkit/kerrgeodesic_gw/master?filepath=Notebooks/basic_kerrgeodesic_gw.ipynb)
  - [Timelike geodesic in Kerr spacetime](https://nbviewer.jupyter.org/github/BlackHolePerturbationToolkit/kerrgeodesic_gw/blob/master/Notebooks/geod_Kerr.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/BlackHolePerturbationToolkit/kerrgeodesic_gw/master?filepath=Notebooks/geod_Kerr.ipynb)
  - [Gravitational waves from circular orbits around a Kerr black hole](https://nbviewer.jupyter.org/github/BlackHolePerturbationToolkit/kerrgeodesic_gw/blob/master/Notebooks/grav_waves_circular.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/BlackHolePerturbationToolkit/kerrgeodesic_gw/master?filepath=Notebooks/grav_waves_circular.ipynb)
  - [More on gravitational waves from circular orbits](https://share.cocalc.com/share/2b3f8da9-6d53-4261-b5a5-ff27b5450abb/gw_single_particle.ipynb?viewer=share)

- For the tensor calculus functionalities of the
  [KerrBH](https://share.cocalc.com/share/2b3f8da9-6d53-4261-b5a5-ff27b5450abb/kerrgeodesic_gw/docs/build/html/kerr_spacetime.html)
  class provided by the package, see these examples:
  [Kerr 1](https://nbviewer.jupyter.org/github/sagemanifolds/SageManifolds/blob/master/Worksheets/v1.3/SM_Kerr.ipynb),
  [Kerr 2](https://nbviewer.jupyter.org/github/sagemanifolds/SageManifolds/blob/master/Worksheets/v1.3/SM_Kerr_Killing_tensor.ipynb),
  [Kerr 3](https://nbviewer.jupyter.org/github/sagemanifolds/SageManifolds/blob/master/Worksheets/v1.3/SM_Simon-Mars_Kerr.ipynb),
  and more generally [SageManifolds documentation](https://sagemanifolds.obspm.fr/documentation.html).


## Authors

- Eric Gourgoulhon
- Alexandre Le Tiec
- Frederic Vincent
- Niels Warburton

**Reference:** E. Gourgoulhon, A. Le Tiec, F. H. Vincent & N. Warburton: *Gravitational waves from bodies orbiting the Galactic center black hole and their detectability by LISA*, [A&A **627**, A92 (2019)](https://doi.org/10.1051/0004-6361/201935406) (preprint: [arXiv:1903.02049](https://arxiv.org/abs/1903.02049))
