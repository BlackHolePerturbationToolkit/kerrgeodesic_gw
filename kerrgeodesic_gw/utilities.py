r"""
Utilities.

"""
#******************************************************************************
#       Copyright (C) 2018 Eric Gourgoulhon <eric.gourgoulhon@obspm.fr>
#
#  Distributed under the terms of the GNU General Public License (GPL)
#  as published by the Free Software Foundation; either version 2 of
#  the License, or (at your option) any later version.
#                  http://www.gnu.org/licenses/
#******************************************************************************

import os
from sage.rings.real_double import RDF

def read_signal(filename, dirname=None):
    r"""
    Read a data file containing some signal
    """
    td = []
    sd = []
    if dirname:
        file_name = os.path.join(dirname, filename)
    else:
        file_name = filename
    with open(file_name, "r") as data_file:
        for dline in data_file:
            t, s = dline.split('\t')
            td.append(RDF(t))
            sd.append(RDF(s))
    return td, sd

def save_signal(td, sd, filename, dirname=None):
    r"""
    Write a signal in a data file.
    """
    if dirname:
        file_name = os.path.join(dirname, filename)
    else:
        file_name = filename
    with open(file_name, "w") as output_file:
        for t, s in zip(td, sd):
            output_file.write("{}\t{}\n".format(t, s))
