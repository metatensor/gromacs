#
# This file is part of the GROMACS molecular simulation package.
#
# Copyright 2025- The GROMACS Authors
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

if (GMX_USE_HDF5)
    gmx_add_unit_test(H5mdTests h5md-test
        CPP_SOURCE_FILES
            h5md.cpp
            h5mdtestbase.cpp
            h5md_attribute.cpp
            h5md_dataset.cpp
            h5md_datasetbuilder.cpp
            h5md_framedatasetbuilder.cpp
            h5md_group.cpp
            h5md_guard.cpp
            h5md_read.cpp
            h5md_type.cpp
            h5md_write.cpp
            h5md_util.cpp
        )
else()
    gmx_add_unit_test(H5mdTests h5md-test
        CPP_SOURCE_FILES
            h5md_disabled.cpp
        )
endif()

target_link_libraries(h5md-test PRIVATE h5md)
