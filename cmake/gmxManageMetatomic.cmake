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
    include(FetchContent)

    find_package(metatensor-core)

    if(NOT metatensor-core_FOUND)
        message(STATUS "metatensor-core not found, fetching from git...")
        FetchContent_Declare(metatensor-core
            GIT_REPOSITORY "https://github.com/metatensor/metatensor.git"
            GIT_TAG "metatensor-core-v0.1.17"
        )
        FetchContent_MakeAvailable(metatensor-core)
    endif()

    find_package(metatensor-torch)

    if(NOT metatensor-torch_FOUND)
        message(STATUS "metatensor-torch not found, fetching from git...")
        FetchContent_Declare(metatensor-torch
            GIT_REPOSITORY "https://github.com/metatensor/metatensor.git"
            GIT_TAG "metatensor-torch-v0.8.0"
        )
        FetchContent_MakeAvailable(metatensor-torch)
    endif()

    find_package(metatomic-torch)

    if(NOT metatomic-torch_FOUND)
        message(STATUS "metatomic-torch not found, fetching from git...")
        FetchContent_Declare(metatomic-torch
            GIT_REPOSITORY "https://github.com/metatensor/metatomic.git"
            GIT_TAG "metatomic-torch-v0.1.4"
        )
        FetchContent_MakeAvailable(metatomic-torch)
    endif()

    list(APPEND GMX_COMMON_LIBRARIES
        metatensor-core
        metatensor-torch
        metatomic-torch
    )

endif()
