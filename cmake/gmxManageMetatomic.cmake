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
    # Taken near verbatim from LAMMPS
    # https://github.com/metatensor/lammps/blob/metatomic/cmake/Modules/Packages/ML-METATOMIC.cmake
    if (BUILD_OMP AND APPLE)
        message(FATAL_ERROR
            "Can not enable both BUILD_OMP and PGK_ML-METATOMIC on Apple systems, "
            "since this results in two different versions of the OpenMP library (one "
            "from the system and one from Torch) being linked to the final "
            "executable, which then crashes"
        )
    endif()

    # Bring the `torch` target in scope to allow evaluation
    # of cmake generator expression from `metatensor_torch`
    find_package(Torch REQUIRED)

    # The caffe2::mkl target contains MKL_INCLUDE_DIR in it's
    # INTERFACE_INCLUDE_DIRECTORIES even if MKL was not found, causing a build
    # failure with "Imported target "torch" includes non-existent path" down the
    # line. This code removes the missing path from INTERFACE_INCLUDE_DIRECTORIES,
    # allowing the build to continue further.
    if (TARGET caffe2::mkl)
        get_target_property(CAFFE2_MKL_INCLUDE_DIRECTORIES caffe2::mkl INTERFACE_INCLUDE_DIRECTORIES)
        set(MKL_INCLUDE_DIR_NOTFOUND "")
        foreach(_include_dir_ ${CAFFE2_MKL_INCLUDE_DIRECTORIES})
            if ("${_include_dir_}" MATCHES "MKL_INCLUDE_DIR-NOTFOUND")
                set(MKL_INCLUDE_DIR_NOTFOUND "${_include_dir_}")
            endif()
        endforeach()

        if (NOT "${MKL_INCLUDE_DIR_NOTFOUND}" STREQUAL "")
            list(REMOVE_ITEM CAFFE2_MKL_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIR_NOTFOUND}")
        endif()
        set_target_properties(caffe2::mkl PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${CAFFE2_MKL_INCLUDE_DIRECTORIES}"
        )
    endif()

    ################ definition of metatensor and metatomic targets ################

    set(METATENSOR_CORE_VERSION "0.1.17")
    set(METATENSOR_CORE_SHA256 "42119e11908239915ccc187d7ca65449b461f1d4b5af4d6df1fb613d687da76a")

    set(METATENSOR_TORCH_VERSION "0.8.0")
    set(METATENSOR_TORCH_SHA256 "61d383ce958deafe0e3916088185527680c9118588722b17ec5c39cfbaa6da55")

    set(METATOMIC_TORCH_VERSION "0.1.4")
    set(METATOMIC_TORCH_SHA256 "385ec8b8515d674b6a9f093f724792b2469e7ea2365ca596f574b64e38494f94")

    set(VESIN_VERSION "0.3.7")
    set(VESIN_GIT_TAG "d0036631d52b75dca9352d80f028a6383335d6d2")

    set(DOWNLOAD_VESIN_DEFAULT ON)
    find_package(vesin QUIET ${VESIN_VERSION})
    if (vesin_FOUND)
        set(DOWNLOAD_VESIN_DEFAULT OFF)
    endif()

    set(DOWNLOAD_METATENSOR_DEFAULT ON)
    find_package(metatensor_torch QUIET ${METATENSOR_TORCH_VERSION})
    if (metatensor_torch_FOUND)
        set(DOWNLOAD_METATENSOR_DEFAULT OFF)
    endif()

    set(DOWNLOAD_METATOMIC_DEFAULT ON)
    find_package(metatomic_torch QUIET ${METATOMIC_TORCH_VERSION})
    if (metatomic_torch_FOUND)
        set(DOWNLOAD_METATOMIC_DEFAULT OFF)
    endif()


    option(DOWNLOAD_METATENSOR "Download metatensor package instead of using an already installed one" ${DOWNLOAD_METATENSOR_DEFAULT})
    option(DOWNLOAD_METATOMIC "Download metatomic package instead of using an already installed one" ${DOWNLOAD_METATOMIC_DEFAULT})

    if (DOWNLOAD_METATENSOR)
        include(FetchContent)

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
        include(FetchContent)

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

    if (DOWNLOAD_VESIN)
        include(FetchContent)

        FetchContent_Declare(
            vesin
            GIT_REPOSITORY https://github.com/Luthaf/vesin.git
            GIT_TAG ${VESIN_GIT_TAG}
        )

        FetchContent_MakeAvailable(vesin)
    else()
        # make sure to fail the configuration if cmake can not find vesin
        find_package(vesin REQUIRED ${VESIN_VERSION})
    endif()

    list(APPEND GMX_COMMON_LIBRARIES
        vesin
        metatomic_torch
        metatensor_torch
    )

endif()
