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
 * ## Domain decomposition (DD) strategy
 *
 * Each rank evaluates the model on its local (home + halo) MTA atoms.
 * The GROMACS excluded pairlist serves as the pair source; a backward ghost
 * atom exchange followed by a backward pair exchange ensures every home atom
 * has ALL its pairs (the "newton pair ON" pattern from LAMMPS pair_metatomic).
 * Distance vectors and shifts are recomputed from local positions.
 * The energy is summed over home atoms only (`selected_atoms = home_only`).
 * This approach is safe for all model architectures.
 *
 * **Atoms**: In DD, GROMACS partitions atoms into "home" atoms (owned by
 * this rank) and "halo" atoms (copies from neighboring ranks needed for
 * short-range interactions). The same global atom may appear as multiple
 * periodic ghost images in the halo. We deduplicate these so each atom has
 * one model index, but record ALL GROMACS local buffer indices in
 * gmxLocalToMtaIdx_ so that the pairlist (which may reference any image) can
 * be resolved.
 *
 * **Forces**: backward() produces forces on all local atoms (home + halo).
 * Since ForceWithVirial is not communicated by dd_move_f, home forces are
 * applied directly and non-home (halo) forces are exchanged via sparse
 * indexed communication (allgatherv pattern). Falls back to dense allreduce
 * for small systems (N_total_mta < 1000).
 *
 * **Shift convention**: GROMACS shifts atom I (first):
 * d = x[I]+shift - x[J]. Metatensor convention: r_ij = x[J] + cell_shift*box
 * - x[I]. Therefore metatensor cell shifts = negated GROMACS cell shifts.
 * Backward pairs have shifts recomputed from local positions using the
 * minimum-image convention.
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

    /*! \brief Exchange backward-direction pairs via ring-based MPI_Sendrecv.
     *
     * In GROMACS DD, the excluded pairlist assigns each pair to exactly one
     * rank (the rank whose home atom is the i-atom). For newton mode, each
     * rank needs ALL pairs involving its home atoms, including those assigned
     * to other ranks. This method discovers those missing pairs by
     * circulating packed (globalI, globalJ) pair buffers in P-1 ring rounds,
     * then adds them to backwardPairsMta_ with shifts recomputed from local
     * positions. Complexity: O(total_pairs) communication, O(max_pairs/rank)
     * memory without an O(N²) global pair table.
     *
     * Must be called after exchangeBackwardGhosts() (so all atom positions
     * are available) and before the NL building loop.
     *
     * \param[in] box        Current simulation box.
     * \param[in] maxRounds  Maximum number of ring exchange rounds (capped by
     *                       ceil(cutoff/minCellSize) when DD info is available).
     */
    void exchangeBackwardPairs(const matrix box, int maxRounds);

    /*! \brief Exchange backward ghost MTA atoms via DD to fill the backward gap.
     *
     * In the standard GROMACS DD halo exchange, atoms are communicated
     * via forward pulses only.  When the number of forward pulses does
     * not cover all cells in a dimension (npulse < numCells - 1), atoms
     * from the backward PBC neighbor are missing from the local halo.
     * This method performs one additional forward exchange per DD dimension
     * to fill this gap.
     *
     * Protocol: each rank identifies home MTA atoms within \p cutoff of its
     * forward cell boundary, then calls \c ddSendrecv(dd, d, dddirForward).
     * The receiving rank gets atoms from its backward neighbor.  Received
     * positions are PBC-shifted when the backward neighbor wraps around.
     *
     * Called once per step from \c calculateForces(), after
     * \c gatherAtomPositions() and before \c exchangeBackwardPairs().
     * Extends positions_, atomNumbers_, mtaToGlobalMta_, and numLocalMta_
     * with the backward ghost atoms.  The caller must save and restore the
     * original numLocalMta_ after model evaluation.
     *
     * \param[in] dd      Domain decomposition structure.
     * \param[in] box     Current simulation box.
     * \param[in] cutoff  Maximum model cutoff (nm).
     * \returns Number of backward ghost atoms added.
     */
    int32_t exchangeBackwardGhosts(const gmx_domdec_t* dd, const matrix box, double cutoff);

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

    //! Backward-direction pairs discovered by exchangeBackwardPairs().
    //! These are pairs from other ranks' pairlists that involve this rank's
    //! home atoms. Flat [i0,j0, i1,j1, ...] with MTA local indices.
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
