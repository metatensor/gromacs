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


option(GMX_METATOMIC "Enable interface to metatomic atomistic models" OFF)

# if(TORCH_ALREADY_SEARCHED)
#     set(FIND_TORCH_QUIETLY ON)
# endif()

if(GMX_METATOMIC)
    # For the initial development, we just build the metatebsor library as part
    # of the gromacs build process. Later we'll add an option to use
    # pre-installed versions of the libraries

    include(FetchContent)

    set(URL_METATENSOR_BASE "https://github.com/metatensor/metatensor/releases/download")
    set(URL_METATOMIC_BASE "https://github.com/metatensor/metatomic/releases/download")

    set(METATENSOR_CORE_VERSION "0.1.17")
    FetchContent_Declare(metatensor
        URL ${URL_METATENSOR_BASE}/metatensor-core-v${METATENSOR_CORE_VERSION}/metatensor-core-cxx-${METATENSOR_CORE_VERSION}.tar.gz
        URL_HASH SHA256=42119e11908239915ccc187d7ca65449b461f1d4b5af4d6df1fb613d687da76a
    )

    message(STATUS "Fetching metatensor v${METATENSOR_CORE_VERSION} from github")
    FetchContent_MakeAvailable(metatensor)

    set(METATOMIC_CORE_VERSION "0.1.0")
    FetchContent_Declare(metatomic
        URL ${URL_METATOMIC_BASE}/metatomic-core-v${METATOMIC_CORE_VERSION}/metatomic-core-cxx-${METATOMIC_CORE_VERSION}.tar.gz
        URL_HASH SHA256=a5fc42a847ba8bb70e1ab9eb421784d1085b3472c7df1e2c1af2d490fd4ed3bd
    )

    message(STATUS "Fetching metatomic v${METATOMIC_CORE_VERSION} from github")
    FetchContent_MakeAvailable(metatomic)


    set(METATOMIC_TORCH_VERSION "0.1.4")
    FetchContent_Declare(metatomic-torch
        URL ${URL_METATOMIC_BASE}/metatomic-torch-v${METATOMIC_TORCH_VERSION}/metatomic-torch-cxx-${METATOMIC_TORCH_VERSION}.tar.gz
        URL_HASH SHA256=385ec8b8515d674b6a9f093f724792b2469e7ea2365ca596f574b64e38494f94
    )

    message(STATUS "Fetching metatomic-torch v${METATOMIC_TORCH_VERSION} from github")
    FetchContent_MakeAvailable(metatomic-torch)

    list(APPEND GMX_COMMON_LIBRARIES metatomic_torch)
endif()
