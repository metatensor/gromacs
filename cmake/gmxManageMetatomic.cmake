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


gmx_option_multichoice(GMX_METATOMIC
  "Enable interface to metatomic atomistic models"
    AUTO
    AUTO ON OFF
)

set(GMX_METATOMIC_ACTIVE OFF)

if(NOT GMX_METATOMIC STREQUAL "OFF")
    # Resolve metatensor first so missing dependencies disable AUTO gracefully.
    find_package(metatensor 0.2.4 CONFIG QUIET)
    if(metatensor_FOUND)
        find_package(metatomic CONFIG QUIET)
    endif()
    if(metatomic_FOUND)
        set(GMX_METATOMIC_ACTIVE ON)
        message(STATUS "Found metatomic ${metatomic_VERSION} (metatensor ${metatensor_VERSION}): "
                       "Metatomic potential support enabled.")

        list(APPEND GMX_COMMON_LIBRARIES metatomic::shared)
        # mdrun loads the plugins installed with this libmetatomic when the
        # metatomic-plugin mdp option is empty
        set(GMX_METATOMIC_PLUGINS_DIR "${METATOMIC_PLUGINS_DIR}")
    elseif(GMX_METATOMIC STREQUAL "ON")
        message(FATAL_ERROR
            "metatomic with metatensor >= 0.2.4 was not found. Install "
            "libmetatomic and add the directories containing "
            "metatomic-config.cmake and metatensor-config.cmake to "
            "CMAKE_PREFIX_PATH.")
    else() # AUTO
        message(STATUS "metatomic not found. Metatomic potential support will be disabled.")
    endif()
endif()
