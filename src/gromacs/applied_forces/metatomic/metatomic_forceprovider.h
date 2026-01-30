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

/*! For compatibility with pairlist data structure in MDModulesPairlistConstructedSignal.
 * Contains pairs like ((atom1, atom2), shiftIndex).
 */
using PairlistEntry = std::pair<std::pair<int, int>, int>;

/*! \brief \internal
 * MetatomicForceProvider class
 *
 * Implements the IForceProvider interface for the Metatomic force provider.
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

    //! Set pairlist from notification and filter to MTA atom pairs.
    void setPairlist(const MDModulesPairlistConstructedSignal& signal);

private:
    //! Gather atom positions for MTA input.
    void gatherAtomPositions(ArrayRef<const RVec> globalPositions);

    //! Prepare pairlist input for model
    void preparePairlistInput();

    const MetatomicOptions& options_;
    const MDLogger&         logger_;
    const MpiComm&          mpiComm_;

    //! vector storing all MTA atom positions
    std::vector<RVec> positions_;

    //! vector storing all atomic numbers
    std::vector<int> atomNumbers_;

    //! lookup table to map model input indices [0...numInput) to local atom indices
    std::vector<int> inputToLocalIndex_;

    //! lookup table to map model input indices to global atom indices
    std::vector<int> inputToGlobalIndex_;

    //! Full pairlist from MDModules notification
    std::vector<PairlistEntry> fullPairlist_;

    //! Interacting pairs of MTA atoms within cutoff, for model input
    std::vector<int> pairlistForModel_;

    //! Shift vectors for each atom pair in pairlistForModel_
    std::vector<RVec> shiftVectors_;

    //! local copy of simulation box
    matrix box_;

    //! Data required for metatomic calculations
    std::unique_ptr<MetatomicData> data_;

    //! flag to check if pairlist data should be prepared
    bool doPairlist_ = false;
};

} // namespace gmx
