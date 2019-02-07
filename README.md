# kerrgeodesic_gw

A [SageMath](http://www.sagemath.org/) package to compute gravitational radiation from material orbiting a Kerr black hole

## Installation

### Requirements

This package requires the Python-based free mathematics software system [SageMath](http://www.sagemath.org/) (at least version 8.2).

### Simple installation from PyPI

It suffices to run

	sage -pip install kerrgeodesic_gw

to have the package ready to use in SageMath.
See however the *install from source* section below if you want to build a
local version of the documentation or modify the source files (development)

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

- [Reference manual](https://share.cocalc.com/share/2b3f8da9-6d53-4261-b5a5-ff27b5450abb/kerrgeodesic_gw/docs/build/html/index.html) ([PDF](https://cocalc.com/share/2b3f8da9-6d53-4261-b5a5-ff27b5450abb/kerrgeodesic_gw/docs/build/latex/kerrgeodesic_gw.pdf))
- [Demo notebook](https://share.cocalc.com/share/2b3f8da9-6d53-4261-b5a5-ff27b5450abb/gw_single_particle.ipynb?viewer=share)
