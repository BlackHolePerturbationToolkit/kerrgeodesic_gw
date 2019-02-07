## -*- encoding: utf-8 -*-
import os
import sys
from setuptools import setup
from codecs import open # To open the README file with proper encoding
from setuptools.command.test import test as TestCommand # for tests


# Get information from separate files (README, VERSION)
def readfile(filename):
    with open(filename,  encoding='utf-8') as f:
        return f.read()

# For the tests
class SageTest(TestCommand):
    def run_tests(self):
        errno = os.system("sage -t --force-lib kerrgeodesic_gw")
        if errno != 0:
            sys.exit(1)

setup(
    name="kerrgeodesic_gw",
    version=readfile("VERSION").strip(), # the VERSION file is shared with the documentation
    description='Gravitational radiation from material orbiting a Kerr black hole',
    long_description=readfile("README.md"), # get the long description from the README
    long_description_content_type='text/markdown',
    url='https://github.com/BlackHolePerturbationToolkit/kerrgeodesic_gw',
    author='Eric Gourgoulhon, Alexandre Le Tiec, Frederic Vincent, Niels Warburton',
    author_email='eric.gourgoulhon@obspm.fr', # choose a main contact email
    license='GPLv2+', # This should be consistent with the LICENCE file
    classifiers=[
      # How mature is this project? Common values are
      #   3 - Alpha
      #   4 - Beta
      #   5 - Production/Stable
      'Development Status :: 4 - Beta',
      'Intended Audience :: Science/Research',
      'Topic :: Scientific/Engineering :: Mathematics',
      'Topic :: Scientific/Engineering :: Physics',
      'Topic :: Scientific/Engineering :: Astronomy',
      'License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)',
      'Programming Language :: Python :: 2.7',
      'Programming Language :: Python :: 3',
    ], # classifiers list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
    keywords="SageMath",
    packages=['kerrgeodesic_gw'],
    package_data={'kerrgeodesic_gw': ['data/*.dat']},
    cmdclass={'test': SageTest}, # adding a special setup command for tests
    install_requires=['sphinx'],
)
