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
    AUTO TORCH OFF
)

if(TORCH_ALREADY_SEARCHED)
    set(FIND_TORCH_QUIETLY ON)
endif()

set(GMX_TORCH OFF)
if(NOT GMX_METATOMIC STREQUAL "OFF")
    # TODO(rg): try to consolidate with the managennpot
    if(GMX_GPU_CUDA AND NOT TORCH_CUDA_ARCH_LIST)
        set(TORCH_CUDA_ARCH_LIST)
        foreach(_arch IN LISTS GMX_CUDA_ARCHITECTURES)
            if(_arch MATCHES "^[0-9]+[a-z]?(-virtual)?$")
                # Convert _arch from 75 or 75-virtual to 7.5+PTX
                string(REGEX REPLACE "^([0-9]+)([0-9][a-z]?)(-virtual)?$" "\\1.\\2+PTX" arch_ptx "${_arch}")
            elseif(_arch MATCHES "^[0-9]+[a-z]?-real$")
                # Convert _arch from 75-real to 7.5
                string(REGEX REPLACE "^([0-9]+)([0-9][a-z]?)-real$" "\\1.\\2" arch_ptx "${_arch}")
            else()
                message(FATAL_ERROR "Unknown CUDA architecture: ${_arch}")
            endif()
            set(TORCH_CUDA_ARCH_LIST "${TORCH_CUDA_ARCH_LIST} ${arch_ptx}")
        endforeach()
    endif()

    # Mirror gmxManageNNPot's CUDA-architectures workaround: torch unsets
    # CMAKE_CUDA_ARCHITECTURES inside its config, so save+restore around the
    # find_package call.
    set(_cmake_cuda_architectures_bak "${CMAKE_CUDA_ARCHITECTURES}")
    unset(CMAKE_CUDA_ARCHITECTURES CACHE)
    if (NOT GMX_USE_NVTX)
        # Stub nvToolsExt to prevent torch CMake from failing on systems
        # without it. Same workaround as gmxManageNNPot.
        if(NOT TARGET CUDA::nvToolsExt)
            add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
        endif()
    endif()

    # Bring the `torch` target in scope to allow evaluation of cmake
    # generator expressions from `metatensor_torch`. Use QUIET (not REQUIRED)
    # so that GMX_METATOMIC=AUTO can silently disable when torch is absent
    # (e.g. on Windows / macOS GitHub runners). The else()/elseif() arms
    # below honor the AUTO/TORCH distinction.
    find_package(Torch 2.0.0 QUIET)
    set(CMAKE_CUDA_ARCHITECTURES "${_cmake_cuda_architectures_bak}" CACHE STRING "")
    set(TORCH_ALREADY_SEARCHED TRUE CACHE BOOL "True if a search for libtorch has already been done")
    mark_as_advanced(TORCH_ALREADY_SEARCHED)

    if(Torch_FOUND)
        # This toggle exists anyway
        # Filter out nvToolsExt from TORCH_LIBRARIES to prevent build failures
        if (NOT GMX_USE_NVTX)
        set(_filtered_torch_libs "")
        # TORCH_LIBRARIES contain imported target "torch" that will set all flags and include paths etc
        foreach(_lib IN LISTS TORCH_LIBRARIES)
            if(NOT _lib MATCHES "nvToolsExt")
                list(APPEND _filtered_torch_libs "${_lib}")
            endif()
        endforeach()
        endif()
        if(NOT FIND_TORCH_QUIETLY)
            message(STATUS "Found Torch: Metatomic potential support enabled.")
        endif()

        # Check if the Torch version uses the correct ABI
        if (${TORCH_CXX_FLAGS} MATCHES "-D_GLIBCXX_USE_CXX11_ABI=0")
            message(FATAL_ERROR "Torch was compiled with the pre-cxx11 ABI. Please use a libtorch version "
                                "compiled with the cxx11 ABI, which is required for building GROMACS.")
        endif()

        set(GMX_TORCH ON)

        # Ensure the torch library directory is in RPATH so that libtorch.so,
        # libc10.so, etc. can be found at runtime. CMAKE_INSTALL_RPATH_USE_LINK_PATH
        # doesn't always extract paths from imported targets, so we add it explicitly.
        # Guard prevents duplicate additions if gmxManageNNPot already added it.
        if(TORCH_INSTALL_PREFIX AND NOT _torch_rpath_added)
            list(APPEND CMAKE_INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib")
            set(_torch_rpath_added TRUE)

            # Use RPATH instead of RUNPATH. Modern linkers default to RUNPATH
            # (--enable-new-dtags), but RUNPATH doesn't propagate to transitive
            # dependencies: gmx -> libgromacs.so -> libtorch.so -> libc10.so.
            include(CheckLinkerFlag)
            check_linker_flag(CXX "-Wl,--disable-new-dtags" _linker_supports_disable_new_dtags)
            if(_linker_supports_disable_new_dtags)
                add_link_options("-Wl,--disable-new-dtags")
            endif()
        endif()

        ################ definition of metatensor and metatomic targets ################
        # These are torch-dependent: only declare when Torch was found, so
        # AUTO mode on platforms without torch leaves the build clean.

        set(METATENSOR_CORE_VERSION "0.1.17")
        set(METATENSOR_CORE_SHA256 "42119e11908239915ccc187d7ca65449b461f1d4b5af4d6df1fb613d687da76a")

        set(METATENSOR_TORCH_VERSION "0.8.0")
        set(METATENSOR_TORCH_SHA256 "61d383ce958deafe0e3916088185527680c9118588722b17ec5c39cfbaa6da55")

        set(METATOMIC_TORCH_VERSION "0.1.7")
        set(METATOMIC_TORCH_SHA256 "726f5711b70c4b8cc80d9bc6c3ce6f3449f31d20acc644ab68dab083aa4ea572")

        set(DOWNLOAD_METATENSOR_DEFAULT ON)
        find_package(metatensor_torch ${METATENSOR_TORCH_VERSION} QUIET)
        if (metatensor_torch_FOUND)
            set(DOWNLOAD_METATENSOR_DEFAULT OFF)
        endif()

        set(DOWNLOAD_METATOMIC_DEFAULT ON)
        find_package(metatomic_torch ${METATOMIC_TORCH_VERSION} QUIET)
        if (metatomic_torch_FOUND)
            set(DOWNLOAD_METATOMIC_DEFAULT OFF)
        endif()


        option(DOWNLOAD_METATENSOR "Download metatensor package instead of using an already installed one" ${DOWNLOAD_METATENSOR_DEFAULT})
        option(DOWNLOAD_METATOMIC "Download metatomic package instead of using an already installed one" ${DOWNLOAD_METATOMIC_DEFAULT})

        include(FetchContent)

        if (DOWNLOAD_METATENSOR)
            set(URL_BASE "https://github.com/metatensor/metatensor/releases/download")
            FetchContent_Declare(metatensor
                URL ${URL_BASE}/metatensor-core-v${METATENSOR_CORE_VERSION}/metatensor-core-cxx-${METATENSOR_CORE_VERSION}.tar.gz
                URL_HASH SHA256=${METATENSOR_CORE_SHA256}
            )

            message(STATUS "Fetching metatensor v${METATENSOR_CORE_VERSION} from github")
            FetchContent_MakeAvailable(metatensor)

            FetchContent_Declare(metatensor-torch
                URL ${URL_BASE}/metatensor-torch-v${METATENSOR_TORCH_VERSION}/metatensor-torch-cxx-${METATENSOR_TORCH_VERSION}.tar.gz
                URL_HASH SHA256=${METATENSOR_TORCH_SHA256}
            )

            message(STATUS "Fetching metatensor-torch v${METATENSOR_TORCH_VERSION} from github")
            FetchContent_MakeAvailable(metatensor-torch)
        else()
            # make sure to fail the configuration if cmake can not find metatensor-torch
            find_package(metatensor_torch REQUIRED ${METATENSOR_TORCH_VERSION})
        endif()

        if (DOWNLOAD_METATOMIC)
            set(URL_BASE "https://github.com/metatensor/metatomic/releases/download")
            FetchContent_Declare(metatomic-torch
                URL ${URL_BASE}/metatomic-torch-v${METATOMIC_TORCH_VERSION}/metatomic-torch-cxx-${METATOMIC_TORCH_VERSION}.tar.gz
                URL_HASH SHA256=${METATOMIC_TORCH_SHA256}
            )

            message(STATUS "Fetching metatomic-torch v${METATOMIC_TORCH_VERSION} from github")
            FetchContent_MakeAvailable(metatomic-torch)
        else()
            # make sure to fail the configuration if cmake can not find metatomic-torch
            find_package(metatomic_torch REQUIRED ${METATOMIC_TORCH_VERSION})
        endif()

        list(APPEND GMX_COMMON_LIBRARIES
            metatensor
            metatomic_torch
            metatensor_torch
        )

    elseif(GMX_METATOMIC STREQUAL "TORCH")
        message(FATAL_ERROR "Torch not found. Please install libtorch and add its installation prefix"
                            " to CMAKE_PREFIX_PATH or set Torch_DIR to a directory containing "
                            "a TorchConfig.cmake or torch-config.cmake file.")
    else() # "AUTO"
        if(NOT FIND_TORCH_QUIETLY)
            message(STATUS "Torch not found. Metatomic potential support will be disabled.")
        endif()
    endif()

endif()
