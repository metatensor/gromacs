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
 * \author Eric Boittier <eric.boittier@epfl.ch>
 * \author Rohit Goswami <r95g10@gmail.com>
 * \author Guillaume Fraux <guillaume.fraux@epfl.ch>
 * \ingroup module_applied_forces
 */

#pragma once

#include <unordered_map>
#include <vector>

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
class PlainPairlistRanges;

/*! \brief \internal
 * MetatomicForceProvider class
 *
 * Implements the IForceProvider interface for the Metatomic force provider.
 *
 * MTA atoms are the atoms of metatomic-input-group, the atoms the model sees.
 *
 * ## Domain decomposition (DD) strategy
 *
 * The energy of a home atom depends on the atoms within the interaction range
 * of the model, in every direction, and on the pairs among them (message
 * passing). The DD halo of GROMACS is one-sided and about rlist wide, so it is
 * not used: the positions of the home MTA atoms of all ranks are all-reduced,
 * and each rank takes the MTA atoms within the interaction range of its home
 * atoms, and finds the pairs among them itself (gatherEnvironment). The model
 * runs on these atoms and returns the outputs of the home atoms only
 * (`selected_atoms`), so each energy is counted once.
 *
 * **Forces**: `execute_model` returns the positions gradient (dE/dx) on every
 * atom the model saw (home + environment). The force is its negative.
 * ForceWithVirial is not communicated by dd_move_f, so home forces are applied
 * directly and the others are sent to their home ranks. Below 1000 metatomic
 * atoms the exchange is a dense all-reduce.
 *
 * **Units**: systems and pair vectors use nm, with energy requested in kJ/mol.
 * Metatomic converts the model's outputs and gradients to these units.
 *
 * **Shift convention**: GROMACS shifts atom I (first):
 * d = x[I]+shift - x[J]. Metatensor convention: r_ij = x[J] + cell_shift*box
 * - x[I]. Therefore metatensor cell shifts = negated GROMACS cell shifts.
 * With DD, pairs are minimum-image pairs (findPairs).
 */
class MetatomicForceProvider final : public IForceProvider
{
public:
    MetatomicForceProvider(const MetatomicOptions&, const MDLogger&, const MpiComm&);
    ~MetatomicForceProvider();

    /*! \brief Evaluate the model and add its energy, forces and virial.
     *
     * \param[in] fInput input for force provider
     * \param[out] fOutput output for force provider
     */
    void calculateForces(const ForceProviderInput& inputs, ForceProviderOutput* outputs) override;

    //! Gather atom numbers and indices. Triggered on AtomsRedistributed signal.
    void gatherAtomNumbersIndices(const MDModulesAtomsRedistributedSignal& signal);

    //! Store GROMACS pairlist and convert to MTA model indices, without DD only.
    void setPairlist(const MDModulesPairlistConstructedSignal& signal);

    /*! \brief The plain pairlist range this model needs, in nm.
     *
     * The largest pair-list cutoff plus a buffer for the motion between pairlist
     * updates (twice the RMS displacement, at least 0.1 nm), capped at the
     * normal pairlist cut-off and never below the pair-list cutoff. A model
     * without pair lists needs no plain pairlist: the range is then 0, as with
     * DD, where the force provider finds the pairs itself.
     */
    real plainPairlistRange(const PlainPairlistRanges& ranges) const;

private:
    //! Gather atom positions for MTA input (local only, no MPI).
    void gatherAtomPositions(ArrayRef<const RVec> positions);

    /*! \brief Add the environment of the home atoms, and the pairs among all, with DD.
     *
     * All-reduces the positions of the home MTA atoms of all ranks, appends the MTA
     * atoms within the environment range of the home atoms (the caller removes them
     * after the evaluation), and fills pairlistMta_ and cellShiftsMta_ with the
     * minimum-image pairs among the home and appended atoms.
     *
     * \param[in] box  Current simulation box.
     */
    void gatherEnvironment(const matrix box);

    //! Distribute non-home forces via sparse indexed exchange (allgatherv pattern).
    //! Home forces are applied directly; only the forces on other atoms are communicated.
    //! \param[in] forces   Flat force array [numLocalMta_ * 3], row-major (fx,fy,fz).
    //! \param[out] outputs Force provider output to accumulate into.
    void distributeNonHomeForces(const double* forces, ForceProviderOutput* outputs);

    const MetatomicOptions& options_;
    const MDLogger&         logger_;
    const MpiComm&          mpiComm_;

    //! Positions of the MTA atoms the model sees, indexed by model index [0, numLocalMta_).
    //! Home atoms occupy [0, numHomeMta_), with DD their environment [numHomeMta_, numLocalMta_).
    std::vector<RVec> positions_;

    //! Atomic numbers of local MTA atoms, same indexing as positions_.
    std::vector<int32_t> atomNumbers_;

    //! Number of home (owned by this rank) MTA atoms.
    int32_t numHomeMta_ = 0;
    //! Number of MTA atoms the model sees: the home atoms, and with DD their environment.
    int32_t numLocalMta_ = 0;

    //! Maps home model index [0, numHomeMta_) -> GROMACS local buffer index.
    //! Used for position gathering and force scattering.
    std::vector<int32_t> mtaToGmxLocal_;
    //! Maps model index [0, numLocalMta_) -> global MTA index [0, N_total_mta).
    //! Used for scatter/gather in the global force buffer during MPI all-reduce.
    std::vector<int32_t> mtaToGlobalMta_;
    //! Maps GROMACS local buffer index -> MTA model index, -1 for other atoms.
    //! Without DD only, to read the GROMACS pairlist.
    std::vector<int32_t> gmxLocalToMtaIdx_;

    //! Maps global MTA index -> local home model index (only home atoms).
    //! Built in gatherAtomNumbersIndices(), used by distributeNonHomeForces()
    //! to identify incoming forces destined for this rank's home atoms.
    std::unordered_map<int32_t, int32_t> globalMtaToLocalHome_;

    //! Pairlist in MTA model indices, flat [i0,j0, i1,j1, ...]. Without DD, built from the
    //! GROMACS interacting and excluded pairlists with negated cell shifts; with DD by findPairs.
    std::vector<int32_t> pairlistMta_;
    //! Cell shifts for each pair (metatensor convention: shift applied to second atom).
    std::vector<IVec> cellShiftsMta_;

    //! Samples of the pairs given to the model, flat [n_pairs * 5]: i, j, cs_a, cs_b, cs_c.
    //! Filled from pairlistMta_/cellShiftsMta_, kept to reuse the allocation.
    std::vector<int32_t> nlSamplesBuffer_;

    //! local copy of simulation box
    matrix box_;

    //! Data required for metatomic calculations
    std::unique_ptr<MetatomicData> data_;
};

} // namespace gmx
