# kerrgeodesic_gw

A [SageMath](http://www.sagemath.org/) package to compute gravitational radiation from material orbiting a Kerr black hole

## Installation

### Local install from source

Download the source from the git repository:

	git clone https://gitlab.obspm.fr/gourgoul/kerrgeodesic_gw.git

This creates a directory `kerrgeodesic_gw`.

Run

	sage -pip install --upgrade --no-index -v kerrgeodesic_gw

A shortcut of the above command is provided by the `Makefile` distributed with the package:

	cd kerrgeodesic_gw
	make install

*NB:* on [CoCalc](https://cocalc.com), you need to add the option `--user`, i.e. open a terminal and run

	git clone https://gitlab.obspm.fr/gourgoul/kerrgeodesic_gw.git
	sage -pip install --user --upgrade --no-index -v kerrgeodesic_gw

#### Install for development

If you plan to edit the package source, you should add the option `-e` to the pip install, i.e. run

	sage -pip install --upgrade --no-index -v -e kerrgeodesic_gw
	
or equivalently

	cd kerrgeodesic_gw
	make develop


### Install from PyPI

Simply run

	sage -pip install kerrgeodesic_gw

(not ready yet)


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

The package documentation can be generated using Sage's [Sphinx](http://www.sphinx-doc.org/) installation:

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