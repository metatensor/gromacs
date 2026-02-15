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
 * ## Domain decomposition strategy
 *
 * Each rank evaluates the model on its local (home + halo) MTA atoms.
 * The neighbor list comes from the GROMACS plain pairlist (excludedPairlist),
 * which assigns each pair to exactly one rank — no double counting.
 *
 * **Atoms**: In DD, GROMACS partitions atoms into "home" atoms (owned by this
 * rank) and "halo" atoms (copies from neighboring ranks needed for short-range
 * interactions). The same global atom may appear as multiple periodic ghost
 * images in the halo. We deduplicate these so each atom has one model index,
 * but record ALL GROMACS local buffer indices in gmxLocalToMtaIdx_ so that
 * the pairlist (which may reference any image) can be resolved.
 *
 * **Pairs**: MTA-MTA pairs are excluded from classical nonbonded interactions
 * via intermolecularExclusionGroup (set by addEmbeddedNBExclusions). The
 * GROMACS pairlist builder reports these excluded pairs in excludedPairlist_,
 * filtered to the plainPairlistRange (= model cutoff). Each pair appears on
 * exactly one rank.
 *
 * **Energy**: With per_atom=true, the model decomposes energy per atom.
 * selected_atoms is always nullopt: we sum ALL per-atom energies (home + halo)
 * on each rank. Since each pair is on one rank, the per-pair energy
 * (V_ij/2 on atom i + V_ij/2 on atom j) sums to V_ij on that rank.
 * GROMACS global_stat sums across ranks for the total.
 *
 * **Forces**: backward() produces forces on all local atoms (home + halo).
 * Since ForceWithVirial is not communicated by dd_move_f, we scatter forces
 * into a global buffer and MPI all-reduce, then apply only to home atoms.
 *
 * **Shift convention**: GROMACS shifts atom I (first): d = x[I]+shift - x[J].
 * Metatensor convention: r_ij = x[J] + cell_shift*box - x[I]. Therefore
 * metatensor cell shifts = negated GROMACS cell shifts.
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
    std::unordered_map<int32_t, int32_t> gmxLocalToMtaIdx_;

    //! Global force buffer [N_total_mta] for MPI all-reduce of forces.
    //! Each rank scatters its local forces here, all-reduce sums them,
    //! then home forces are read back.
    std::vector<RVec> globalForceBuffer_;

    //! Pairlist in MTA model indices, flat [i0,j0, i1,j1, ...].
    //! Built from GROMACS excludedPairlist_ with negated cell shifts.
    std::vector<int32_t> pairlistMta_;
    //! Cell shifts for each pair (metatensor convention: shift applied to second atom).
    std::vector<IVec> cellShiftsMta_;

    //! local copy of simulation box
    matrix box_;

    //! Data required for metatomic calculations
    std::unique_ptr<MetatomicData> data_;
};

} // namespace gmx
