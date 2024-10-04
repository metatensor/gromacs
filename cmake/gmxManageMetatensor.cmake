#
# This file is part of the GROMACS molecular simulation package.
#
# Copyright 2024- The GROMACS Authors
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


option(GMX_METATENSOR "Enable interface to metatensor atomistic models" OFF)

# if(TORCH_ALREADY_SEARCHED)
#     set(FIND_TORCH_QUIETLY ON)
# endif()

if(GMX_METATENSOR)
    # For the initial development, we just build the metatebsor library as part
    # of the gromacs build process. Later we'll add an option to use
    # pre-installed versions of the libraries

    include(FetchContent)

    set(URL_BASE "https://github.com/metatensor/metatensor/releases/download")

    set(METATENSOR_CORE_VERSION "0.1.10")
    FetchContent_Declare(metatensor
        URL ${URL_BASE}/metatensor-core-v${METATENSOR_CORE_VERSION}/metatensor-core-cxx-${METATENSOR_CORE_VERSION}.tar.gz
        URL_HASH SHA1=e89495c7ce425831257f66f0084ffb28caefc516
    )

    message(STATUS "Fetching metatensor v${METATENSOR_CORE_VERSION} from github")
    FetchContent_MakeAvailable(metatensor)


    set(METATENSOR_TORCH_VERSION "0.5.5")
    FetchContent_Declare(metatensor-torch
        URL ${URL_BASE}/metatensor-torch-v${METATENSOR_TORCH_VERSION}/metatensor-torch-cxx-${METATENSOR_TORCH_VERSION}.tar.gz
        URL_HASH SHA1=ad0c6ad5c8ea364b0b85fd96fd656822fd1b1443
    )

    message(STATUS "Fetching metatensor-torch v${METATENSOR_TORCH_VERSION} from github")
    FetchContent_MakeAvailable(metatensor-torch)

    list(APPEND GMX_COMMON_LIBRARIES metatensor_torch)
endif()
