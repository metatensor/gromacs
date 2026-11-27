#!/usr/bin/env python3
#
# This file is part of the GROMACS molecular simulation package.
#
# Copyright 2026- The GROMACS Authors
# and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
# Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
#
# GROMACS is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License
# as published by the Free Software Foundation; either version 2.1
# of the License, or (at your option) any later version.
#
# GROMACS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with GROMACS; if not, see
# https://www.gnu.org/licenses, or write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
#
# If you want to redistribute modifications to GROMACS, please
# consider that scientific software is very special. Version
# control is crucial - bugs must be traceable. We will be happy to
# consider code for inclusion in the official distribution, but
# derived work must not be called official GROMACS. Details are found
# in the README & COPYING files - if they are missing, get the
# official version at https://www.gromacs.org.
#
# To help us fund GROMACS development, we humbly ask that you cite
# the research papers on the package. Check out https://www.gromacs.org.
"""Regenerate the argon Lennard-Jones model used by metatomic.cpp.

The exported model is a truncated Lennard-Jones potential shifted so the
pair energy is zero at the cutoff. Install the exporter that still speaks
the NeighborListOptions interface this GROMACS build loads, for example:

    pip install 'git+https://github.com/metatensor/lj-test@a1464af'
"""

import pathlib

import metatomic_lj_test


def main():
    model = metatomic_lj_test.lennard_jones_model(
        atomic_type=18,
        cutoff=0.9,
        epsilon=0.978638,
        sigma=0.3401,
        length_unit="nm",
        energy_unit="kJ/mol",
        with_extension=False,
    )
    destination = pathlib.Path(__file__).with_name("model.pt")
    model.save(str(destination))
    print(destination)


if __name__ == "__main__":
    main()
