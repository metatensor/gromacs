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
 * Each rank evaluates the model on its local (home + halo) MTA atoms.
 * The neighbor list comes from the GROMACS pairlist (excludedPairlist),
 * which assigns each pair to exactly one rank — no double counting.
 * Forces are combined via MPI all-reduce on a global force buffer.
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

    //! vector storing local MTA atom positions (home + halo)
    std::vector<RVec> positions_;

    //! vector storing local MTA atomic numbers (home + halo)
    std::vector<int32_t> atomNumbers_;

    //! Number of home MTA atoms on this rank
    int32_t numHomeMta_ = 0;
    //! Number of home + halo MTA atoms on this rank
    int32_t numLocalMta_ = 0;

    //! Maps local model index [0, numLocalMta_) -> GROMACS local buffer index
    std::vector<int32_t> mtaToGmxLocal_;
    //! Maps local model index [0, numLocalMta_) -> global MTA index [0, N_total_mta)
    std::vector<int32_t> mtaToGlobalMta_;
    //! Maps ANY GROMACS local buffer index (including periodic ghosts) -> MTA model index
    //! Used by setPairlist to resolve pairlist entries that reference ghost images.
    std::unordered_map<int32_t, int32_t> gmxLocalToMtaIdx_;

    //! Global force buffer sized [N_total_mta] for MPI all-reduce
    std::vector<RVec> globalForceBuffer_;

    //! Pairlist from GROMACS (MTA model indices), flat [A0,B0,A1,B1,...]
    std::vector<int32_t> pairlistMta_;
    //! Integer cell shifts for each pair in pairlistMta_
    std::vector<IVec> cellShiftsMta_;

    //! local copy of simulation box
    matrix box_;

    //! Data required for metatomic calculations
    std::unique_ptr<MetatomicData> data_;
};

} // namespace gmx
