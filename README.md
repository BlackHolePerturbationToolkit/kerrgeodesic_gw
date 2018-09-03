# kerrgeodesic_gw

A [SageMath](http://www.sagemath.org/) package to compute gravitational radiation from material orbiting a Kerr black hole

## Installation

### Local install from source

Download the source from the git repository:

	git clone https://gitlab.obspm.fr/gourgoul/kerrgeodesic_gw.git

This creates a directory `kerrgeodesic_gw`.

Run

	sage -pip install --upgrade --no-index -v -e PATH_TO_KERRGEODESIC_GW

where `PATH_TO_KERRGEODESIC_GW` is the path to the directory `kerrgeodesic_gw` created by the `git clone` command.

You may then test the install by running all the doctests in the package source files via

	sage -t PATH_TO_KERRGEODESIC_GW
	
You should then get the message `All tests passed!`

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
