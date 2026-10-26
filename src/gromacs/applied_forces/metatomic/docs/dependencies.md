# Build dependencies of the metatomic interface

The force provider talks to metatomic through its **C API**, so this module
does not link libtorch. `GMX_TORCH` stays with NNPot; `GMX_METATOMIC` resolves
`metatomic` and `metatensor` on its own (`cmake/gmxManageMetatomic.cmake`).

| package | requirement | state |
| --- | --- | --- |
| metatomic-core (`libmetatomic`) | the API of 545d5976 or later; pinned to **07cb8766** | **unreleased**: branch `metatomic-core` of `metatensor/metatomic`, PR [#232](https://github.com/metatensor/metatomic/pull/232) (draft). `Cargo.toml` has `publish = false`. |
| metatensor-core (`libmetatensor`) | >= 0.2.4 | released 2026-07-30. Also metatomic-core's own `REQUIRED_METATENSOR_VERSION`. |

Both must be on `CMAKE_PREFIX_PATH`, as separate prefixes when they come from
wheels. `metatomic-config.cmake` calls `find_package(metatensor REQUIRED)`, so
GROMACS resolves metatensor first to keep a missing dependency a soft failure
under `GMX_METATOMIC=AUTO`.

## Building metatomic-core

```bash
git clone https://github.com/metatensor/metatomic && cd metatomic
git checkout 07cb8766af9c28fed4e232f911a24ec3b11a3840
cmake -S metatomic-core -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=$PREFIX -DCMAKE_PREFIX_PATH=<metatensor prefix>
cmake --build build && cmake --install build
```

This needs cargo >= 1.88. It installs `libmetatomic`, the headers, the CMake
package and the Lennard-Jones test plugin (`libexec/metatomic/`). Then
configure GROMACS with `-DGMX_METATOMIC=ON -DCMAKE_PREFIX_PATH="$PREFIX;<metatensor prefix>"`.

## The floor is an API check, not a version number

Asking `find_package` for a metatomic version is a trap here, and the trap is
worth stating because it looks like an obvious thing to add.

metatomic-core's config file is written with `COMPATIBILITY SameMinorVersion`,
so a request is satisfied only by the *same* `0.x`. The version it compares
against is not the one in `metatomic-core/Cargo.toml` (`0.1.0`): the build runs
that through `create_development_version()`, which bumps the minor for an
untagged tree. A source build of the 0.1.0 branch therefore installs a package
reporting `0.2.0-dev<N>`. Asking for `0.1.0` rejects every such build; asking
for `0.2.0` would reject `0.1.0` once it is released.

So `gmxManageMetatomic.cmake` asks for no version and instead compiles a check
against the installed headers: `ExternalModel::requested_pair_lists` must take
the requested outputs. That signature arrived in 545d5976 ("Requested inputs
and pair list can depend on which outputs will be computed", 2026-10-19), and
the interface is written against it. An older metatomic-core disables the
interface under `GMX_METATOMIC=AUTO` and stops the configuration under `ON`,
with the version found in the message. The configure log always records the
versions found:

```
-- Found metatomic 0.2.0-dev1226+git.07cb876 (metatensor 0.2.6): Metatomic potential support enabled.
```

## Units, and metatensor/metatomic#369

The force provider hands `execute_model` its system in the **model's own
length unit** (positions, cell and pair vectors are scaled from nm), requests
the energy in kJ/mol, and converts the positions gradient from kJ/mol per model
length unit to kJ/mol/nm itself.

This makes the forces independent of a change in metatomic-core. Up to
07cb8766, `execute_model` converts a system given in another length unit to the
model's unit but scales outputs and gradients by the energy factor only, so the
positions gradient stays per model length unit. Open PR
[#369](https://github.com/metatensor/metatomic/pull/369) also applies the length
factor to the positions gradient. A system in nm with a GROMACS-side factor of
1 / (model unit in nm), as the C-API branch did before (`617316faab`), is right
before #369 and 10x too large for an Angstrom model after it. With the system
in the model's unit, metatomic converts no length either way.
`tests/metatomic_forceprovider.cpp` checks the forces of a model in Angstrom
against the analytic values; run against a metatomic-core with #369 it passes,
and with the earlier conversion it fails with forces 10x too large.

## The metatensor floor has to track metatomic-core's

`metatomic-config.cmake` runs its own `find_package(metatensor
${REQUIRED_METATENSOR_VERSION} CONFIG REQUIRED)`. At 07cb8766 that number is
`0.2.4`, the same as the floor here. If metatomic-core raises it, GROMACS's
looser floor lets metatensor pass, `find_package(metatomic)` runs, and
metatomic-config aborts on its own `REQUIRED`: a hard configure error, also
under `GMX_METATOMIC=AUTO`. If you hit `Could not find a configuration file for
package "metatensor" ... version: 0.3` from inside metatomic-config, match the
metatensor your metatomic-core was built against.

## API surface in use

An API break upstream shows up in these names only:

- `<metatomic.hpp>`: `ExternalModel`, `load_model`, `load_plugin`,
  `execute_model`, `ModelCapabilities`, `PairListOptions`, `System`,
  `SampleKind`, `Quantity`, `Gradients`, `DLPackTensor`,
  `unit_conversion_factor`, `Error`
- `<metatensor.hpp>`: `Labels`, `TensorMap`, `TensorBlock`, `SimpleDataArray`,
  `DataArrayBase`
- `<metatensor/dlpack/dlpack.h>`: the DLPack structs the model inputs are
  handed over as

`tests/metatomic_forceprovider.cpp` drives the force provider with a pair model
registered as an in-process plugin (`mta_register_plugin`).

## Pinning

Build against **`metatensor/metatomic` @ `07cb8766`** ("Numpy-based
LennardJonesModel for testing", 22 Oct 2026) with metatensor-core 0.2.4 or
later (0.2.6 tested). The branch was force-pushed on 2026-10-22: the earlier
pin `a245aed8` no longer exists upstream (`9e1de9a4` has the same
`metatomic-core/` tree, and predates the API above). Since then:

- 545d5976 gives `requested_pair_lists()` and `requested_inputs()` the
  requested outputs as a parameter. The force provider builds its list of
  outputs first and asks for the pair lists and inputs of exactly those.
- 9e7312dc only re-indents the C++ headers.
- a370eeed validates the units of standard quantities: the force provider
  requests `kJ/mol`, `kJ/mol/nm` and `kJ/mol/nm^3`.

This is the reason the build prints the versions it found: the dependency is a
branch, and it moved under this module after the module was written.

CI (`.github/workflows/metatomic-capi.yml`) builds metatomic-core at the same
commit, configures with `GMX_METATOMIC=ON` and runs the unit tests: bump the
commit there and here together.

## Validating the C API migration

The `metatomic-capi.yml` workflow builds the pinned core with metatensor-core
0.2.6 on Linux/GCC/OpenMP and macOS/Clang. It replaces the libtorch-only pixi
workflow. The following commands use an already configured GROMACS build:

```bash
cmake --build build --target gmx mta_applied_forces-test mdrunutility-test metatomic-test-plugin
build/bin/mta_applied_forces-test
build/bin/mdrunutility-test
python3 admin/ci-scripts/metatomic-cmake-smoke.py
python3 admin/ci-scripts/metatomic-capi-smoke.py build/bin/gmx build/lib/metatomic-test-plugin.so
```

The unit tests compare energy, forces and virial with analytic Lennard-Jones
values, including nm/kJ/mol, Angstrom/eV, triclinic cell shifts, nonperiodic
and XY systems, and rejection of screw periodicity and unavailable inputs.
The CMake fixture checks AUTO/ON/OFF, the TORCH alias, incompatible APIs,
reconfiguration after a dependency change, and preservation of caller settings.

The end-to-end test loads the same analytic model from a shared test plugin,
runs `grompp` and single-rank `mdrun`, and checks the resulting energy and forces
in both nm and Angstrom. This tests model-option serialization, dynamic loading,
setup notifications and the actual GROMACS pair list. The shared plugin is a
test fixture, not an installed runtime dependency. The LJ plugin shipped with
the pinned core does not advertise strain gradients, which this conservative
force provider requires.

These checks cover the CPU, single-rank migration. They do not establish
correctness for arbitrary production models, accelerator plugins or domain
decomposition. ONIOM/link atoms, site batching, electrostatic corrections and
new distributed algorithms belong in follow-up changes after this interface
has been reviewed. Existing domain-decomposition code remains in the provider;
the new single-rank smoke test is not evidence of MPI correctness.
