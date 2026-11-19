#!/usr/bin/env python3
"""Exercise metatomic package discovery and the public CMake option values."""
from pathlib import Path
import subprocess
import tempfile


def main():
    cmake = Path(__file__).resolve().parents[2] / 'cmake'
    with tempfile.TemporaryDirectory(prefix='gromacs-capi-cmake-') as temporary:
        root = Path(temporary)
        source = root / 'source'
        source.mkdir()
        (source / 'CMakeLists.txt').write_text('''cmake_minimum_required(VERSION 3.28)
project(MetatomicDiscovery LANGUAGES CXX)
include("''' + (cmake / 'gmxOptionUtilities.cmake').as_posix() + '''")
include("''' + (cmake / 'gmxManageMetatomic.cmake').as_posix() + '''")
if(NOT "${GMX_METATOMIC_ACTIVE}" STREQUAL "${EXPECT_ACTIVE}")
    message(FATAL_ERROR "Unexpected activation: ${GMX_METATOMIC_ACTIVE}")
endif()
''')
        prefix = root / 'prefix'
        prefix.mkdir()
        (prefix / 'metatensor-config.cmake').write_text('''set(metatensor_VERSION 0.2.6)
add_library(metatensor::shared INTERFACE IMPORTED)
''')
        (prefix / 'metatensor-config-version.cmake').write_text(
            'set(PACKAGE_VERSION 0.2.6)\nset(PACKAGE_VERSION_COMPATIBLE TRUE)\n')
        (prefix / 'metatomic-config.cmake').write_text('''set(metatomic_VERSION 0.2.0-dev)
add_library(metatomic::shared INTERFACE IMPORTED)
''')

        def configure(name, mode, active, missing=None, success=True):
            args = ['cmake', '-S', str(source), '-B', str(root / name),
                    f'-DCMAKE_PREFIX_PATH={prefix}', f'-DGMX_METATOMIC={mode}',
                    f'-DEXPECT_ACTIVE={active}']
            for package in ['metatensor', 'metatomic']:
                args.append(f'-DCMAKE_DISABLE_FIND_PACKAGE_{package}={"TRUE" if package == missing else "FALSE"}')
            result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            assert (result.returncode == 0) == success, result.stdout
            if not success:
                assert 'was not found' in result.stdout or 'Invalid value for GMX_METATOMIC' in result.stdout, result.stdout
            print(f'{name} ({mode}): PASS')

        configure('off', 'OFF', 'OFF', missing='metatensor')
        for package in ['metatensor', 'metatomic']:
            configure(f'auto-without-{package}', 'AUTO', 'OFF', missing=package)
            configure(f'on-without-{package}', 'ON', 'OFF', missing=package, success=False)
        configure('on', 'ON', 'ON')
        configure('auto', 'AUTO', 'ON')
        configure('reconfigure', 'AUTO', 'ON')
        configure('reconfigure', 'OFF', 'OFF')
        configure('reconfigure', 'AUTO', 'ON')


if __name__ == '__main__':
    main()
