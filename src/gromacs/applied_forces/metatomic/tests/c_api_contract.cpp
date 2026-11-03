/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2024- The GROMACS Authors
 * and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
 * Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * https://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at https://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out https://www.gromacs.org.
 */
/*! \internal \file
 * \brief
 * Recorded contracts for the metatomic C API engine path.
 *
 * Each test names the assertion the C API port must satisfy. The bodies
 * skip until the helper they name exists, so this file links before that
 * port does. The commented EXPECT_* lines are the contract.
 *
 * The checks that can run without a GROMACS build are also in
 * docs/c_api_contracts.ipynb, rewritten by docs/generate_c_api_notebook.py.
 *
 * \author Metatensor developers <https://github.com/metatensor>
 * \ingroup module_applied_forces
 */

#include "gmxpre.h"

#include <gtest/gtest.h>

namespace gmx
{
namespace test
{

// Part 1. Contracts

TEST(CApiEveryCall, ChecksTypesAndScalesUnitsWithoutConsistency)
{
    // capabilities.atomic_types = {1, 8}, length_unit = "angstrom",
    // declared energy unit = "eV". System length unit = "nm",
    // types = {1, 7}, positions in nm, requested energy unit = "kJ/mol".
    // check_consistency = false.
    //
    // EXPECT_THROW(execute_model(...), metatomic::Error);  // type 7
    // A second system with types {1, 8} must succeed, and the returned
    // energy must already be in kJ/mol (eV * unit_conversion_factor).
    // positions_ before the call must be bitwise unchanged.
    GTEST_SKIP() << "stub: needs a tiny BaseModel that echoes positions";
}

TEST(CApiCheckConsistency, RejectsDtypeDeviceAndExtraSamples)
{
    // check_consistency = true.
    // EXPECT_THROW on float64 positions when capabilities.dtype is float32.
    // EXPECT_THROW when selected_atoms names ("system","atom") row (0, 99)
    //     on a 4-atom system.
    // EXPECT_THROW when the energy block has a "strain" gradient that was
    //     not in the requested Quantity.
    // A model that omits the requested "positions" gradient must NOT throw
    // inside execute_model. The engine check is separate:
    // EXPECT_THROW(require_positions_gradient(block), gmx::APIError);
    // requested_inputs() == {spin} is rejected at init.
    // requested_inputs() == {charge} is accepted and attached as custom data.
    GTEST_SKIP() << "stub: needs BaseModel fixtures for dtype, samples, gradients";
}

TEST(CApiSystem, OwnsTensorsAndRejectsNonZeroVacuumRow)
{
    // metatomic::System("nm", types(4), positions(4,3), cell, pbc{true,true,false})
    // with a non-zero third cell row:
    // EXPECT_THROW(..., metatomic::Error);
    // Same call with a zero third row must return size() == 4.
    // After construction the caller's DLPack pointers have been released;
    // destroying the System must not free the GROMACS buffer (custom deleter
    // frees only the DLPack header).
    GTEST_SKIP() << "stub: wrap_dlpack helper not written";
}

TEST(CApiPairs, ChecksLayoutAndNotDistances)
{
    // add_pairs with sample name "atom_i" instead of "first_atom":
    // EXPECT_THROW(..., metatomic::Error);
    // A block whose vector length is 2 * cutoff, options.strict() == true:
    // EXPECT_NO_THROW. strict is an engine promise.
    // Adding the same cutoff/full_list/strict twice:
    // EXPECT_THROW, even if requestors differ.
    // GROMACS shift (1,0,0) must be stored as cell_shift (-1,0,0),
    // with r_ij = x[j] - x[i] + shift_box.
    GTEST_SKIP() << "stub: pair_block helper not written";
}

TEST(CApiEnergyBlock, SumsByLabelAndKeepsGradientSign)
{
    // Energy samples are (0,2), (0,0) with values 1.5 and 2.5 kJ/mol.
    // EXPECT_DOUBLE_EQ(sum_energy(block), 4.0);
    // values[0] must not be treated as atom 0.
    // Positions gradient row (sample=0, system=0, atom=3) value +1.0
    // becomes force[3] == -1.0 after the engine negate.
    // Strain gradient value 8.0 becomes virial contribution 4.0 (factor 1/2).
    // Requesting "energy/pbe0" when only "energy" is declared, with
    // check_consistency true: EXPECT_THROW.
    GTEST_SKIP() << "stub: scatter helpers not written";
}

TEST(CApiGromacsOnly, HomeRangeAndHaloExchange)
{
    // numHomeMta_ = 2, numLocalMta_ = 4.
    // selected_atoms rows must be (0,0) and (0,1) only.
    // A positions-gradient atom index 3 is passed to distributeNonHomeForces.
    // A non_conservative_force sample with atom index 3 is not: that output
    // is home-only, and an index >= numHomeMta_ is EXPECT_THROW / a warning
    // path, not a halo exchange.
    GTEST_SKIP() << "stub: needs the force-provider test double, no MPI";
}

// Part 2. Load and requested outputs

TEST(CApiLoad, LoadsPluginAndKeepsModelAlive)
{
    // extensionsDirectory non-empty calls metatomic::load_plugin once.
    // load_model failure becomes gmx::APIError, not a torch exception.
    // ExternalModel destructor calls unload exactly once.
    // requested_pair_lists() is cached on the force provider; it is not
    // fetched again inside calculateForces.
    GTEST_SKIP() << "stub: needs a test plugin .so exporting MTA_REGISTER_PLUGIN";
}

TEST(CApiLoad, ConvertsModelCutoffToNm)
{
    // length_unit "angstrom", cutoff 5.0 -> cutoffNm == 0.5.
    // length_unit "nm", cutoff 1.2 -> cutoffNm == 1.2.
    // EXPECT_DOUBLE_EQ(metatomic::unit_conversion_factor("angstrom", "nm") * 5.0, 0.5);
    GTEST_SKIP() << "stub";
}

TEST(CApiLoad, BuildsQuantityListInGromacsUnits)
{
    // Conservative: requested[0] is "energy", "kJ/mol", SampleKind::Atom,
    // gradients {Positions, Strain}. explicit_gradients is not empty.
    // Uncertainty auto threshold:
    // EXPECT_DOUBLE_EQ(threshold, 0.1 * unit_conversion_factor("eV", "kJ/mol"));
    // Non-conservative: the name is "non_conservative_force", not
    // "non_conservative_forces". Unit "kJ/mol/nm", no gradients.
    // Stress, when declared: "non_conservative_stress", "kJ/mol/nm^3",
    // SampleKind::System.
    // Variant "pbe0" requests "energy/pbe0", not "energy".
    GTEST_SKIP() << "stub";
}

// Part 3. Each MD step

TEST(CApiStep, WrapsGromacsBuffersWithoutCopyingWhenDtypesMatch)
{
    // GROMACS real is double, model dtype is float64.
    // System positions data pointer equals positions_.data().
    // Destroying the System leaves positions_ readable.
    // When real is float and the model is float64, the wrapper copies
    // first; the System pointer must not alias the GROMACS buffer.
    // No strain tensor is built. requires_grad is not set.
    GTEST_SKIP() << "stub: wrap_dlpack not written";
}

TEST(CApiStep, FiltersCutoffOnlyWhenStrict)
{
    // strict true, cutoffNm 0.5: a pair at 0.6 nm is absent from the block.
    // strict false: that pair is present.
    // full_list true: the block contains (j, i) with negated cell shifts
    // and a negated vector. full_list false: one row per GROMACS pair.
    // There is no register_autograd_neighbors call.
    GTEST_SKIP() << "stub";
}

TEST(CApiStep, PassesHomeAtomsAndIndexesOutputsByRequestOrder)
{
    // numHomeMta_ = 2. selected_atoms count is 2, rows (0,0) and (0,1).
    // outputs.size() == requested.size().
    // outputs[energyIndex].block_by_id(0) is the energy block.
    // There is no lookup by the string "energy" on the return value.
    // Rebuilding selected_atoms is skipped when numHomeMta_ is unchanged.
    GTEST_SKIP() << "stub";
}

// Part 4. Forces and virial

TEST(CApiForces, AccumulatesPositionsGradientByAtomLabel)
{
    // Two rows both name atom 1, values -F of (1,0,0) and (0,2,0).
    // forces[1] == (1, 2, 0). Row order is not atom order.
    // A row with atom >= numHomeMta_ is handed to distributeNonHomeForces
    // and is not written into the local force buffer on this rank.
    GTEST_SKIP() << "stub";
}

TEST(CApiForces, HalvesStrainGradientForGromacsVirial)
{
    // One strain sample, value V*sigma = 2 on the xx element.
    // virial[XX][XX] == 1. The other elements stay 0.
    // The factor is +0.5, not -0.5.
    GTEST_SKIP() << "stub";
}

TEST(CApiForces, ScattersNonConservativeForcesWithoutNegating)
{
    // Sample atom 1, xyz value (0.5, 0, -0.25) kJ/mol/nm.
    // force_[mtaToGmxLocal_[1]] gains that vector unchanged.
    // No call to distributeNonHomeForces.
    // Stress xx = 4 kJ/mol/nm^3, volume = 2 nm^3 -> virial xx == 4
    // (0.5 * stress * volume).
    GTEST_SKIP() << "stub";
}

// Part 5. Build

TEST(CApiBuild, DoesNotLinkTorch)
{
    // The applied_forces link line contains metatomic and metatensor.
    // It does not contain torch, metatensor_torch, or metatomic_torch.
    // c_api_contract.cpp is listed in tests/CMakeLists.txt.
    // This is a configure-time check of the CMake diff, not a GTest of the .so.
    GTEST_SKIP() << "stub: recorded against the CMake diff";
}

// Part 6. ONIOM and neighbor-list capacity

TEST(OniomTopology, KeepsBoundaryAnglesAndDihedrals)
{
    // alanine_vacuo, ML indices {8, 9, 10, 11, 12, 13}.
    // An angle with 3 ML atoms is removed.
    // An angle with 2 ML + 1 MM is kept.
    // A dihedral with 3 ML + 1 MM is kept.
    // A bond is removed only when both atoms are ML.
    GTEST_SKIP() << "stub: makeMtopFromFile(alanine_vacuo) already used by the PR tests";
}

TEST(OniomLinks, FindsFrontierInEveryMoleculeOfABlock)
{
    // One molblock, 4 copies, one boundary bond per copy.
    // EXPECT_EQ(linkFrontiers.size(), 4);
    // A walk that uses only globalAtomStart returns 1.
    GTEST_SKIP() << "stub";
}

TEST(NlCapacity, PairCountMatchesBruteForce)
{
    // 4x4x4 lattice, 64 atoms, box 1.2 nm, cutoff 0.5 nm, PBC.
    // No bonds, so excludedPairlist_ is empty.
    // EXPECT_EQ(pairs.size(), bruteForceMinimumImageCount(...));
    // EXPECT_GT(pairs.size(), 0);
    // Repeat with a triclinic box; cell shifts must match the brute-force image.
    // If PlainPairlistRanges is registered below the model cutoff, the count drops.
    // setPairlist must append signal.pairlist_ and signal.excludedPairlist_.
    GTEST_SKIP() << "stub: shared by the full-system, DD, and ONIOM branches";
}

TEST(OniomMd, ConservedEnergyDriftBound)
{
    // ACE-ALA-NME, ML = ALA only, link-atoms=yes, nstlist=1, 200 steps, vacuum.
    // |conserved(last) - conserved(first)| stays within a few kJ/mol.
    // The thread-MPI failure mode is a drift above 2000 kJ/mol inside ~100 steps.
    // Run under thread-MPI and under real MPI. Not a neighbor-list capacity test.
    GTEST_SKIP() << "stub: integration, needs a tiny model and an mdp";
}

} // namespace test
} // namespace gmx
