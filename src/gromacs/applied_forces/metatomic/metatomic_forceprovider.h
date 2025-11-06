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

#include "metatensor.hpp"
#ifdef DIM
#    undef DIM
#endif
#include "metatensor/torch.hpp"
#include "metatomic/torch.hpp"
#ifdef DIM
// XXX(rg): ask gromacs folks to do GMX_DIM
#    undef DIM
#endif

#include "metatomic_options.h"

namespace gmx
{

struct MetatomicParameters;

class MDLogger;
class MpiComm;

/*! \brief \internal
 * TODO
 */
class MetatomicForceProvider final : public IForceProvider
{
public:
    MetatomicForceProvider(const MetatomicOptions&, const MDLogger&, const MpiComm&);
    ~MetatomicForceProvider();

    /*! TODO
     */
    void calculateForces(const ForceProviderInput& inputs, ForceProviderOutput* outputs) override;
    void updateLocalAtoms();
    void gatherAtomPositions(ArrayRef<const RVec> globalPositions);
    void gatherAtomNumbersIndices();

    metatensor_torch::TensorBlock computeNeighbors(metatomic_torch::NeighborListOptions request,
                                                   long                                 n_atoms,
                                                   const float*                         positions,
                                                   const matrix                         box,
                                                   bool                                 periodic);

private:
    /// From NNPot
    const MetatomicOptions& options_;
    const MDLogger&         logger_;
    const MpiComm&          mpiComm_;
    torch::Device           device_;
    //! vector storing all atom positions
    std::vector<RVec> positions_;

    //! vector storing all atomic numbers
    std::vector<int> atomNumbers_;

    //! global index lookup table to map indices from model input to global atom indices
    std::vector<int> idxLookup_;

    //! local copy of simulation box
    matrix box_;
    /// From EON
    torch::jit::Module                                model_;
    metatomic_torch::ModelCapabilities                capabilities_;
    std::vector<metatomic_torch::NeighborListOptions> nl_requests_;
    metatomic_torch::ModelEvaluationOptions           evaluations_options_;
    torch::ScalarType                                 dtype_;
    bool                                              check_consistency_;
};

} // namespace gmx
