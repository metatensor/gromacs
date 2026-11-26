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

#include <vector>

#include "gromacs/mdtypes/iforceprovider.h"
#include "gromacs/topology/embedded_system_preprocessing.h"

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
 * passing). The DD halo of GROMACS is one-sided and about rlist wide, so the
 * force provider exchanges its own (gatherEnvironment): the ranks share the
 * region of their home MTA atoms (7 numbers each), and each rank sends to the
 * others the home atoms that may be within range of their regions. The receiver
 * keeps the atoms within range of its home atoms, and finds the pairs among all
 * of these itself. The model runs on these atoms and returns the outputs of the
 * home atoms only (`selected_atoms`), so each energy is counted once.
 *
 * **Forces**: `execute_model` returns the positions gradient (dE/dx) on every
 * atom the model saw (home + environment). The force is its negative.
 * ForceWithVirial is not communicated by dd_move_f, so home forces are applied
 * directly and the others go back to their home ranks (returnEnvironmentForces),
 * reversing the exchange of the positions.
 *
 * **Units**: systems and pair vectors use nm, with energy requested in kJ/mol.
 * Metatomic converts the model's outputs and gradients to these units.
 *
 * **Shift convention**: GROMACS shifts atom I (first):
 * d = x[I]+shift - x[J]. Metatensor convention: r_ij = x[J] + cell_shift*box
 * - x[I]. Therefore metatensor cell shifts = negated GROMACS cell shifts.
 * With DD, pairs are minimum-image pairs (findPairs).
 *
 * ## Link atoms (metatomic-link-atoms)
 *
 * Each bond between an embedded atom and an MM atom gets a hydrogen cap, at the link
 * distance from the embedded atom along the bond. The MM atoms of these bonds are MTA
 * atoms after those of the input group: the first cap on an MM atom takes its row in the
 * model input, further caps on the same atom get rows after the local atoms. The pairs of
 * the caps are found from their positions. The force on a cap is spread onto the two atoms
 * of its bond, and the virial gets the term of the fixed cap distance. With DD, the
 * environment reaches twice the longest capped bond further, so that the rank has both atoms
 * of every cap it needs, and a cap beyond the local atoms belongs to the home rank of its MM
 * atom.
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
     * Exchanges the home MTA atoms near the regions of the other ranks, appends the
     * received atoms within the environment range of the home atoms (the caller removes
     * them after the evaluation), and fills pairlistMta_ and cellShiftsMta_ with the
     * minimum-image pairs among the home and appended atoms.
     *
     * \param[in] box  Current simulation box.
     */
    void gatherEnvironment(const matrix box);

    /*! \brief Add the forces on the home atoms, with those computed on other ranks, with DD.
     *
     * \param[in] forces   Forces on the atoms the model saw, flat [numLocalMta_ * 3].
     * \param[out] outputs Force provider output to accumulate into.
     */
    void returnEnvironmentForces(ArrayRef<const double> forces, ForceProviderOutput* outputs);

    /*! \brief Place the link caps of the bonds whose two atoms the model sees.
     *
     * Fills caps_, the types and positions the model sees (modelTypes_, modelPositions_), and
     * its pairs (modelPairs_, modelShifts_): those of pairlistMta_ without the MM atoms of the
     * caps, and the pairs of the caps within the largest pair-list cutoff.
     *
     * \param[in] box  Current simulation box.
     */
    void placeLinkCaps(const matrix box);

    /*! \brief Spread the forces on the caps onto the atoms of their bonds.
     *
     * \param[in]     box     Current simulation box.
     * \param[in,out] forces  Forces on the rows of the model input, flat; the rows of the caps
     *                        beyond the local atoms are removed.
     * \param[in,out] virial  With a value, gets the term of the fixed cap distance.
     */
    void spreadLinkCapForces(const matrix box, std::vector<double>* forces, matrix* virial) const;

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
    std::vector<int32_t> mtaToGlobalMta_;
    //! Maps GROMACS local buffer index -> MTA model index, -1 for other atoms.
    //! Without DD only, to read the GROMACS pairlist.
    std::vector<int32_t> gmxLocalToMtaIdx_;

    //! With DD, the home atoms sent to other ranks, ordered by rank, and how many to each.
    std::vector<int32_t> sendAtoms_;
    std::vector<int>     sendCounts_;
    //! With DD, how many atoms came from each rank, and for each environment atom (model
    //! index numHomeMta_ and up) its place among them, for sending its force back.
    std::vector<int>     receivedCounts_;
    std::vector<int32_t> environmentSources_;

    //! Pairlist in MTA model indices, flat [i0,j0, i1,j1, ...]. Without DD, built from the
    //! GROMACS interacting and excluded pairlists with negated cell shifts; with DD by findPairs.
    std::vector<int32_t> pairlistMta_;
    //! Cell shifts for each pair (metatensor convention: shift applied to second atom).
    std::vector<IVec> cellShiftsMta_;

    //! A link cap of the current step
    struct LinkCap
    {
        //! Row of the embedded atom of the bond
        int32_t embedded;
        //! Row of the MM atom of the bond
        int32_t mm;
        //! Row of the cap: the MM atom's for its first cap, after the local atoms for others
        int32_t row;
        //! Cell shift of the MM atom to the minimum image of the bond
        IVec shift;
        //! Distance of the cap from the embedded atom
        real distance;
    };
    //! The bonds capped by link atoms, with the indices in mtaIndices_ of their atoms
    std::vector<LinkFrontierAtom> linkBonds_;
    //! The caps of the current step
    std::vector<LinkCap> caps_;
    //! Row of each MTA atom the model sees, -1 for the others
    std::vector<int32_t> rowOfMta_;
    //! Whether each row of the model input is a cap
    std::vector<char> isCapRow_;
    //! With caps, the types and positions of the rows of the model input
    std::vector<int32_t> modelTypes_;
    std::vector<RVec>    modelPositions_;
    //! With caps, the pairs of the model input and their cell shifts, as pairlistMta_
    std::vector<int32_t> modelPairs_;
    std::vector<IVec>    modelShifts_;

    //! Samples of the pairs given to the model, flat [n_pairs * 5]: i, j, cs_a, cs_b, cs_c.
    //! Filled from pairlistMta_/cellShiftsMta_, kept to reuse the allocation.
    std::vector<int32_t> nlSamplesBuffer_;

    //! local copy of simulation box
    matrix box_;

    //! Data required for metatomic calculations
    std::unique_ptr<MetatomicData> data_;
};

} // namespace gmx
