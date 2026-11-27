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

gmx_add_libgromacs_sources(
    metatomic_options.cpp
)

# The objects of the sources that include LibTorch reach libgromacs through
# this interface target, see src/gromacs/CMakeLists.txt
add_library(metatomic_torch_module INTERFACE)

# GMX_METATOMIC is the user-facing AUTO/TORCH/OFF multichoice; it stays truthy
# even when torch was not found in AUTO mode. GMX_TORCH is set only
# when Torch was located. Without them, link
# the stub force provider.
if (GMX_TORCH AND NOT GMX_METATOMIC STREQUAL "OFF")
    # These two files are the only ones that include the LibTorch, metatensor
    # and metatomic headers. LibTorch can require a newer C++ standard than
    # GROMACS uses (GMX_METATOMIC_CXX_STANDARD, see cmake/gmxManageMetatomic.cmake),
    # so they are built in an object library of their own with that standard.
    add_library(metatomic_torch_objects OBJECT
                metatomic_forceprovider.cpp
                metatomic_mdmodule.cpp)
    if (BUILD_SHARED_LIBS)
        set_target_properties(metatomic_torch_objects PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif()
    set_target_properties(metatomic_torch_objects PROPERTIES
                          CXX_STANDARD ${GMX_METATOMIC_CXX_STANDARD}
                          CXX_SCAN_FOR_MODULES OFF)
    gmx_target_compile_options(metatomic_torch_objects)
    target_compile_definitions(metatomic_torch_objects PRIVATE HAVE_CONFIG_H)
    target_link_libraries(metatomic_torch_objects PRIVATE
                          common
                          legacy_api
                          legacy_modules
                          domdec
                          hardware
                          math
                          mdrunutility
                          mdtypes
                          pbcutil
                          topology
                          utility
                          ${TORCH_LIBRARIES}
                          metatensor_torch
                          metatomic_torch)
    if (GMX_GPU_CUDA)
        target_link_libraries(metatomic_torch_objects PRIVATE CUDA::toolkit)
    elseif (GMX_GPU_HIP)
        target_link_libraries(metatomic_torch_objects PRIVATE hip::host)
    elseif (GMX_GPU_SYCL)
        add_sycl_to_target(TARGET metatomic_torch_objects)
    endif()
    target_link_libraries(metatomic_torch_module INTERFACE
                          $<BUILD_INTERFACE:metatomic_torch_objects>
                          $<TARGET_OBJECTS:metatomic_torch_objects>)
else()
    gmx_add_libgromacs_sources(
        metatomic_mdmodule.cpp
        metatomic_forceprovider_stub.cpp
    )
endif()

if (BUILD_TESTING)
    add_subdirectory(tests)
endif()
