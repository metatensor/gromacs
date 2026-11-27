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
 * Declares the Metatomic Force Provider class
 *
 * \author Metatensor developers <https://github.com/metatensor>
 * \ingroup module_applied_forces
 */

#pragma once

#include <unordered_map>

#include "gromacs/mdtypes/iforceprovider.h"

#include "metatomic_options.h"

namespace gmx
{

struct MetatomicParameters;
struct MetatomicData;
struct MDModulesAtomsRedistributedSignal;
struct MDModulesPairlistConstructedSignal;

class MDLogger;
class MpiComm;

/*! \brief \internal
 * MetatomicForceProvider class
 *
 * Implements the IForceProvider interface for the Metatomic force provider.
 *
 * ## Domain decomposition
 *
 * With more than one rank every rank evaluates the model on all atoms of
 * the input group, since a model can depend on atoms several cutoffs away.
 * The ranks gather the atoms they own (gatherRemoteAtoms()) and the pairs of
 * their GROMACS pairlists (gatherRemotePairs()), which hold each pair on
 * exactly one rank. The energy is that of the home atoms only
 * (`selected_atoms`), so the sum over ranks counts each atom once.
 *
 * **Forces**: the backward pass gives forces on all atoms of the group.
 * ForceWithVirial is not communicated by dd_move_f, so forces on home atoms
 * are applied directly and forces on the other atoms go to the ranks that
 * own them (distributeNonHomeForces()), as an indexed gather or, for small
 * systems (N_total_mta < 1000), a dense reduction.
 *
 * **Shift convention**: GROMACS shifts atom I (first):
 * d = x[I]+shift - x[J]. Metatensor convention: r_ij = x[J] + cell_shift*box
 * - x[I]. Therefore metatensor cell shifts = negated GROMACS cell shifts.
 * A pair received from another rank comes with its pair vector, and its
 * shift is the one that gives that vector between the local copies of the
 * two atoms.
 */
class MetatomicForceProvider final : public IForceProvider
{
public:
    MetatomicForceProvider(const MetatomicOptions&, const MDLogger&, const MpiComm&);
    ~MetatomicForceProvider();

    /*! \brief Calculate forces for Metatomic.
     *
     * Prepares the input for the neural network model triggers model inference.
     * \param[in] fInput input for force provider
     * \param[out] fOutput output for force provider
     */
    void calculateForces(const ForceProviderInput& inputs, ForceProviderOutput* outputs) override;

    //! Gather atom numbers and indices. Triggered on AtomsRedistributed signal.
    void gatherAtomNumbersIndices(const MDModulesAtomsRedistributedSignal& signal);

    //! Store GROMACS pairlist and convert to MTA model indices.
    void setPairlist(const MDModulesPairlistConstructedSignal& signal);

private:
    //! Gather atom positions for MTA input (local only, no MPI).
    void gatherAtomPositions(ArrayRef<const RVec> positions);

    /*! \brief Adds the model atoms that other ranks own to the local system.
     *
     * A model can depend on atoms several cutoffs away, so in parallel each
     * rank evaluates it on all atoms. Every rank sends its home atoms and
     * appends the ones it does not have to positions_, atomNumbers_ and
     * mtaToGlobalMta_, and increases numLocalMta_. The caller restores
     * numLocalMta_ after the model evaluation.
     *
     * Called once per step, after gatherAtomPositions() and before
     * gatherRemotePairs(). Does nothing on a single rank.
     */
    void gatherRemoteAtoms();

    /*! \brief Adds the pairs that other ranks hold to the local pair list.
     *
     * With domain decomposition the GROMACS pairlist holds each pair on
     * exactly one rank. Every rank sends its pairs as global atom indices
     * with the pair vector, and stores the pairs of the other ranks in
     * backwardPairsMta_ with the periodic shift that gives the same vector
     * between its own copies of the two atoms.
     *
     * Called after gatherRemoteAtoms(), which makes every atom of every pair
     * local. Does nothing on a single rank.
     *
     * \param[in] box  Current simulation box.
     */
    void gatherRemotePairs(const matrix box);

    //! Distribute non-home forces via sparse indexed exchange (allgatherv pattern).
    //! Home forces are applied directly; only halo forces are communicated.
    //! \param[in] forces   Flat force array [numLocalMta_ * 3], row-major (fx,fy,fz).
    //! \param[out] outputs Force provider output to accumulate into.
    void distributeNonHomeForces(const double* forces, ForceProviderOutput* outputs);

    const MetatomicOptions& options_;
    const MDLogger&         logger_;
    const MpiComm&          mpiComm_;

    //! Positions of local MTA atoms, indexed by model index [0, numLocalMta_).
    //! Home atoms occupy [0, numHomeMta_), halo atoms [numHomeMta_, numLocalMta_).
    std::vector<RVec> positions_;

    //! Atomic numbers of local MTA atoms, same indexing as positions_.
    std::vector<int32_t> atomNumbers_;

    //! Number of home (owned by this rank) MTA atoms.
    int32_t numHomeMta_ = 0;
    //! Number of unique local MTA atoms (home + halo, after deduplication).
    int32_t numLocalMta_ = 0;

    //! Maps model index [0, numLocalMta_) -> GROMACS local buffer index (first occurrence).
    //! Used for position gathering and force scattering.
    std::vector<int32_t> mtaToGmxLocal_;
    //! Maps model index [0, numLocalMta_) -> global MTA index [0, N_total_mta).
    //! Used for scatter/gather in the global force buffer during MPI all-reduce.
    std::vector<int32_t> mtaToGlobalMta_;
    //! Maps ANY GROMACS local buffer index -> MTA model index.
    //! Includes ALL periodic ghost images of each atom (not just the first).
    //! Needed because excludedPairlist_ entries can reference any image.
    //! Initialized to -1 for non-MTA atoms.
    std::vector<int32_t> gmxLocalToMtaIdx_;

    //! Maps global MTA index -> local home model index (only home atoms).
    //! Built in gatherAtomNumbersIndices(), used by distributeNonHomeForces()
    //! to identify incoming forces destined for this rank's home atoms.
    std::unordered_map<int32_t, int32_t> globalMtaToLocalHome_;

    //! Pairlist in MTA model indices, flat [i0,j0, i1,j1, ...].
    //! Built from GROMACS excludedPairlist_ with negated cell shifts.
    //! Used in both NL modes.
    std::vector<int32_t> pairlistMta_;
    //! Cell shifts for each pair (metatensor convention: shift applied to second atom).
    std::vector<IVec> cellShiftsMta_;

    //! Pairs from the pairlists of other ranks, found by gatherRemotePairs().
    //! Flat [i0,j0, i1,j1, ...] with MTA local indices.
    std::vector<int32_t> backwardPairsMta_;
    //! Cell shifts for backward pairs (recomputed from local positions).
    std::vector<IVec> backwardShiftsMta_;

    //! Pre-allocated raw buffers for NL construction (used by both NL modes).
    //! Filled inline from pairlistMta_/cellShiftsMta_ (+ backward pairs in full mode).
    //! Wrapped with torch::from_blob (zero-cost) before passing to metatensor.
    std::vector<int32_t> nlSamplesBuffer_; //!< flat [n_pairs * 5]: i, j, cs_a, cs_b, cs_c
    std::vector<double>  nlVectorsBuffer_; //!< flat [n_pairs * 3]: dx, dy, dz

    //! local copy of simulation box
    matrix box_;

    //! Data required for metatomic calculations
    std::unique_ptr<MetatomicData> data_;
};

} // namespace gmx
