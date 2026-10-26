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
    AUTO ON TORCH OFF
)

# TORCH was the name for "required" while the interface went through libtorch.
# It does not any more, so the value is ON. The old spelling keeps working:
# existing CI jobs and pixi tasks pass -DGMX_METATOMIC=TORCH.
if(GMX_METATOMIC STREQUAL "TORCH")
    message(STATUS
        "GMX_METATOMIC=TORCH is deprecated and now means ON; the metatomic "
        "interface uses the metatomic C API, not libtorch.")
    set(GMX_METATOMIC "ON")
endif()

# The C API does not use libtorch. Leave GMX_TORCH to NNPot.
set(GMX_METATOMIC_ACTIVE OFF)

if(NOT GMX_METATOMIC STREQUAL "OFF")
    # metatomic-config pulls metatensor. The two packages live in separate
    # prefixes when installed from wheels, so both cmake dirs must be on
    # CMAKE_PREFIX_PATH.
    # metatomic-config calls find_package(metatensor REQUIRED). Resolve
    # metatensor first so a missing dependency stays a soft failure in AUTO.
    # metatensor-core 0.3.0 has never been released; metatomic-core itself
    # requires 0.2.4 (REQUIRED_METATENSOR_VERSION), so match that floor.
    find_package(metatensor 0.2.4 CONFIG QUIET)
    if(metatensor_FOUND)
        # No version number in the request. metatomic-core is unreleased
        # (metatensor/metatomic#232), and its config file pairs
        # SameMinorVersion compatibility with a version that create_development_
        # version() derives by bumping the minor: a source build of the 0.1.0
        # tree reports 0.2.0-dev, so any number we could write here rejects
        # either every dev build or every release. The floor is the API check
        # below, and the STATUS line records what was actually found.
        # See src/gromacs/applied_forces/metatomic/docs/dependencies.md.
        find_package(metatomic CONFIG QUIET)
    endif()
    if(metatomic_FOUND)
        # The floor: metatomic-core 545d5976 (2026-10-19) made the pair lists and
        # inputs of a model depend on the outputs requested from it, and that is
        # the API this interface is written against. Pinned: 07cb8766.
        # Scope probe settings so toolchain flags and subsequent checks are preserved.
        function(_gmx_check_metatomic_api)
            include(CheckCXXSourceCompiles)
            # Re-run when reconfiguring: the dependency prefix can change in place.
            unset(GMX_METATOMIC_API_SUPPORTED CACHE)
            # List metatensor explicitly: the unqualified alias in metatomic's
            # link interface does not carry its includes into try_compile.
            set(CMAKE_REQUIRED_LIBRARIES metatomic::shared metatensor::shared)
            # the vendored nlohmann/json target of metatomic-config is not imported, so
            # it does not reach the test project: give its headers, and only compile
            set(CMAKE_REQUIRED_INCLUDES "${METATOMIC_INCLUDE}/metatomic/third-party")
            set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
            set(CMAKE_REQUIRED_QUIET ON)
            check_cxx_source_compiles("
#include <metatomic.hpp>
int main()
{
    std::vector<metatomic::PairListOptions> (metatomic::ExternalModel::*pairLists)(
                const std::vector<metatomic::Quantity>&) const = &metatomic::ExternalModel::requested_pair_lists;
    std::vector<metatomic::Quantity> (metatomic::ExternalModel::*inputs)(
                const std::vector<metatomic::Quantity>&) const = &metatomic::ExternalModel::requested_inputs;
    (void)pairLists;
    (void)inputs;
    return 0;
}" GMX_METATOMIC_API_SUPPORTED)
        endfunction()
        _gmx_check_metatomic_api()
        if(NOT GMX_METATOMIC_API_SUPPORTED)
            set(metatomic_FOUND FALSE)
            string(CONCAT _mta_reason "metatomic ${metatomic_VERSION} API compile check failed. "
                "This interface requires requested_pair_lists(outputs) and requested_inputs(outputs) "
                "(metatomic-core 545d5976 or later; tested with 07cb8766). "
                "The installed API may be incompatible, or its headers/dependencies may be unavailable. "
                "See CMakeFiles/CMakeConfigureLog.yaml for the compiler diagnostic.")
            if(GMX_METATOMIC STREQUAL "ON")
                message(FATAL_ERROR "${_mta_reason}")
            endif()
            message(STATUS "${_mta_reason} Metatomic potential support will be disabled.")
        endif()
    endif()
    if(metatomic_FOUND)
        set(GMX_METATOMIC_ACTIVE ON)
        # Record the versions: while metatomic-core is unreleased, the build log
        # is the only place the revision in use is written down.
        message(STATUS "Found metatomic ${metatomic_VERSION} (metatensor ${metatensor_VERSION}): "
                       "Metatomic potential support enabled.")

        list(APPEND GMX_COMMON_LIBRARIES metatomic::shared)

        # Pre-installed libraries outside the GROMACS prefix need their own
        # RPATH entry. Pip wheels put metatomic and metatensor in different
        # site-packages directories.
        foreach(_mta_target IN ITEMS metatomic::shared metatensor::shared)
            if(NOT TARGET ${_mta_target})
                continue()
            endif()
            unset(_mta_location)
            string(TOUPPER "${CMAKE_BUILD_TYPE}" _mta_config)
            foreach(_mta_property IN ITEMS IMPORTED_LOCATION_${_mta_config} IMPORTED_LOCATION IMPORTED_LOCATION_RELEASE
                                           IMPORTED_LOCATION_RELWITHDEBINFO IMPORTED_LOCATION_DEBUG)
                get_target_property(_mta_location ${_mta_target} ${_mta_property})
                if(_mta_location)
                    break()
                endif()
            endforeach()
            if(_mta_location)
                get_filename_component(_mta_libdir "${_mta_location}" DIRECTORY)
                list(APPEND CMAKE_INSTALL_RPATH "${_mta_libdir}")
            endif()
        endforeach()
        list(REMOVE_DUPLICATES CMAKE_INSTALL_RPATH)
    elseif(GMX_METATOMIC STREQUAL "ON")
        message(FATAL_ERROR
            "metatomic with metatensor >= 0.2.4 was not found. Install "
            "libmetatomic and add the directories containing "
            "metatomic-config.cmake and metatensor-config.cmake to "
            "CMAKE_PREFIX_PATH. Building metatomic-core is described in "
            "src/gromacs/applied_forces/metatomic/docs/dependencies.md.")
    elseif(NOT DEFINED _mta_reason) # AUTO
        message(STATUS "metatomic not found. Metatomic potential support will be disabled.")
    endif()
endif()
