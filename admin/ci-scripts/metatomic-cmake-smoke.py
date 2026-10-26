#!/usr/bin/env python3
"""Exercise metatomic dependency discovery with isolated CMake package fixtures."""
from pathlib import Path
import subprocess
import tempfile


def main():
    module = Path(__file__).resolve().parents[2] / 'cmake/gmxManageMetatomic.cmake'
    with tempfile.TemporaryDirectory(prefix='gromacs-capi-cmake-') as temporary:
        root = Path(temporary)
        source = root / 'source'
        source.mkdir()
        (source / 'CMakeLists.txt').write_text('''cmake_minimum_required(VERSION 3.28)
project(MetatomicDiscovery LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
function(gmx_option_multichoice name description default)
    set(${name} "${default}" CACHE STRING "${description}")
endfunction()
set(CMAKE_REQUIRED_LIBRARIES sentinel)
set(CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE)
include("''' + module.as_posix() + '''")
if(NOT "${GMX_METATOMIC_ACTIVE}" STREQUAL "${EXPECT_ACTIVE}")
    message(FATAL_ERROR "Unexpected activation: ${GMX_METATOMIC_ACTIVE}")
endif()
if(NOT CMAKE_REQUIRED_LIBRARIES STREQUAL "sentinel" OR
   NOT CMAKE_TRY_COMPILE_TARGET_TYPE STREQUAL "EXECUTABLE")
    message(FATAL_ERROR "API probe changed caller's CMake settings")
endif()
''')
        prefix = root / 'prefix'
        prefix.mkdir()
        headers = root / 'metatensor-include'
        headers.mkdir()
        # A unique transitive header prevents globally installed metatensor headers
        # from hiding a missing include directory in the compile probe.
        (headers / 'gmx_probe_metatensor_dependency.h').write_text('#pragma once\n')
        (prefix / 'metatensor-config.cmake').write_text('''set(metatensor_VERSION 0.2.6)
add_library(metatensor::shared INTERFACE IMPORTED)
set_target_properties(metatensor::shared PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/../metatensor-include")
add_library(metatensor ALIAS metatensor::shared)
''')
        (prefix / 'metatensor-config-version.cmake').write_text(
            'set(PACKAGE_VERSION 0.2.6)\nset(PACKAGE_VERSION_COMPATIBLE TRUE)\n')
        (prefix / 'metatomic-config.cmake').write_text('''set(metatomic_VERSION 0.2.0-dev)
set(METATOMIC_INCLUDE "${CMAKE_CURRENT_LIST_DIR}")
add_library(metatomic::shared INTERFACE IMPORTED)
set_target_properties(metatomic::shared PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}"
    INTERFACE_LINK_LIBRARIES metatensor)
''')
        header = prefix / 'metatomic.hpp'
        supported = '''#include <gmx_probe_metatensor_dependency.h>
#include <vector>
namespace metatomic {
struct PairListOptions {};
struct Quantity {};
struct ExternalModel {
    std::vector<PairListOptions> requested_pair_lists(const std::vector<Quantity>&) const;
    std::vector<Quantity> requested_inputs(const std::vector<Quantity>&) const;
};
}
'''
        header.write_text(supported)

        def configure(name, mode, active, missing=False, success=True):
            args = ['cmake', '-S', str(source), '-B', str(root / name),
                    f'-DCMAKE_PREFIX_PATH={prefix}', f'-DGMX_METATOMIC={mode}',
                    f'-DEXPECT_ACTIVE={active}',
                    f'-DCMAKE_DISABLE_FIND_PACKAGE_metatensor={"TRUE" if missing else "FALSE"}']
            result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            assert (result.returncode == 0) == success, result.stdout
            if not success:
                assert 'was not found' in result.stdout or 'API compile check failed' in result.stdout, result.stdout
            print(f'{name} ({mode}): PASS')

        configure('off', 'OFF', 'OFF', missing=True)
        configure('auto-missing', 'AUTO', 'OFF', missing=True)
        configure('on-missing', 'ON', 'OFF', missing=True, success=False)
        configure('supported', 'ON', 'ON')
        configure('alias', 'TORCH', 'ON')
        header.write_text(supported.replace('requested_inputs(const std::vector<Quantity>&)', 'requested_inputs()'))
        configure('unsupported-auto', 'AUTO', 'OFF')
        configure('unsupported-on', 'ON', 'OFF', success=False)
        # The same build directory previously passed: a cached success must not hide the change.
        configure('supported', 'ON', 'OFF', success=False)
        header.write_text(supported)
        configure('supported', 'ON', 'ON')


if __name__ == '__main__':
    main()
